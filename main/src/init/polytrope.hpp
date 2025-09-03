//
// Created by Noah Kubli on 15.02.2025.
//

#pragma once

#include <map>

#include "cstone/sfc/box.hpp"
#include "cstone/tree/continuum.hpp"
#include "sph/eos.hpp"

#include "isim_init.hpp"
#include "early_sync.hpp"
#include "grid.hpp"
#include "utils.hpp"
#include "polytrope/bisect.hpp"
#include "polytrope/polytrope_profile.hpp"

namespace sphexa
{

std::map<std::string, double> polytropeConstants()
{
    constexpr double r            = 4.72108762739756E-01;
    constexpr double mTotal       = 1e-6;
    constexpr double gravConstant = 1.0;
    const double     t_relax      = std::sqrt(r * r * r / (gravConstant * mTotal)) / 3.;

    return {{"gravConstant", gravConstant}, // {"r", 0.47},
            {"polytrope::r", r},
            {"polytrope::mTotal", mTotal},
            {"polytropic_index", 5. / 3.},
            {"minDt", 1e-4},
            {"minDt_m1", 1e-4},
            {"ng0", 100},
            {"ngmax", 150},
            {"eosChoice", sph::EosType::polytropic},
            {"relaxationTimescale", t_relax}};
}

template<class Dataset>
void initPolytropeFields(Dataset& d, const std::map<std::string, double>& constants, double m_part)
{
    using T = typename Dataset::RealType;

    std::fill(d.m.begin(), d.m.end(), m_part);
    std::fill(d.du_m1.begin(), d.du_m1.end(), 0.0);
    std::fill(d.mui.begin(), d.mui.end(), d.muiConst);
    std::fill(d.alpha.begin(), d.alpha.end(), d.alphamin);

    std::fill(d.vx.begin(), d.vx.end(), 0.0);
    std::fill(d.vy.begin(), d.vy.end(), 0.0);
    std::fill(d.vz.begin(), d.vz.end(), 0.0);

    std::fill(d.x_m1.begin(), d.x_m1.end(), 0.0);
    std::fill(d.y_m1.begin(), d.y_m1.end(), 0.0);
    std::fill(d.z_m1.begin(), d.z_m1.end(), 0.0);

    std::fill(d.u.begin(), d.u.end(), 0.0);

    generateParticleIDs(d.id);
}

template<class Vector>
void contractRadialProfile(Vector& x, Vector& y, Vector& z, double rho_uniform, auto radiusOfEnclosedMass)
{
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x.size(); i++)
    {
        const auto old_radius    = std::sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
        const auto old_volume    = 4. * M_PI / 3. * old_radius * old_radius * old_radius;
        const auto enclosed_mass = old_volume * rho_uniform;
        const auto new_radius    = radiusOfEnclosedMass(enclosed_mass);
        const auto factor        = new_radius / old_radius;

        x[i] *= factor;
        y[i] *= factor;
        z[i] *= factor;
    }
}

template<typename Dataset>
void estimateSmoothingLengths(auto rhoAtRadius, Dataset& d, double m_part, size_t ng0, double r_total)
{
    d.h.resize(d.x.size());

    auto smoothing_length = [rhoAtRadius, m_part, ng0](double r)
    { return 0.5 * std::cbrt(3. * ng0 * m_part / (4. * M_PI * rhoAtRadius(r))); };

    auto boundary_overlap = [&](double r) { return 2. * smoothing_length(r) + r - r_total; };
    const auto [converged, r_resolved] =
        polytrope::find_zero_bisect(boundary_overlap, r_total / 2., r_total, 1e-6 * r_total, 1e-6 * r_total);
    if (!converged) throw std::runtime_error("Find zero not converged");

    const double h_max = smoothing_length(r_resolved);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < d.x.size(); i++)
    {
        const auto radius = std::sqrt(d.x[i] * d.x[i] + d.y[i] * d.y[i] + d.z[i] * d.z[i]);
        d.h[i]            = std::min(h_max, smoothing_length(radius));
    }
}

template<class Dataset>
class Polytrope : public ISimInitializer<Dataset>
{
    std::string          glassBlock;
    mutable InitSettings settings_;

public:
    explicit Polytrope(std::string initBlock, std::string settingsFile, IFileReader* reader)
        : glassBlock(std::move(initBlock))
    {
        Dataset d;
        settings_ = buildSettings(d, polytropeConstants(), settingsFile, reader);
    }

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cbrtNumPart, Dataset& simData,
                                                 IFileReader* reader) const override
    {
        auto& d       = simData.hydro;
        using KeyType = typename Dataset::KeyType;
        using T       = typename Dataset::RealType;

        const double polytropic_index = settings_.at("polytropic_index");
        const double n_polytropic        = 1. / (settings_.at("polytropic_index") - 1.);
        const double m_total             = settings_.at("polytrope::mTotal");
        const double r_total             = settings_.at("polytrope::r");
        const double G                   = settings_.at("gravConstant");
        const size_t ng0                 = settings_.at("ng0");

        auto [rho_r, M_r, polytropic_const] = polytrope::computePolytropeProfile(n_polytropic, m_total, r_total, G);
        settings_["polytropic_const"]       = polytropic_const;

        if (rank == 0)
        {
            std::printf("polytropic constant: %lf\tpolytropic exponent: %lf\n", polytropic_const, polytropic_index);
            std::printf("r_total: %lf\tachieved r: %lf\n", r_total, M_r.y_values.back());
        }
        const auto globalBox = createUniformSphere(rank, numRanks, cbrtNumPart, simData, reader, r_total);

        const double rho_original = m_total / (4. / 3. * M_PI * r_total * r_total * r_total);

        contractRadialProfile(d.x, d.y, d.z, rho_original, M_r);

        syncAndLoadAttributes(rank, numRanks, simData, globalBox);
        const double m_part = m_total / d.numParticlesGlobal;

        estimateSmoothingLengths(rho_r, d, m_part, ng0, r_total);

        initPolytropeFields(d, settings_, m_part);

        return globalBox;
    }

    void syncAndLoadAttributes(int rank, int numRanks, Dataset& simData, const auto& globalBox) const
    {
        auto& d       = simData.hydro;
        using KeyType = typename Dataset::KeyType;

        size_t numParticlesGlobal = d.x.size();
        MPI_Allreduce(MPI_IN_PLACE, &numParticlesGlobal, 1, MpiType<size_t>{}, MPI_SUM, simData.comm);

        auto t0 = std::chrono::high_resolution_clock::now();
        transferToDevice(d, 0, d.x.size(), {"x", "y", "z"});
        syncCoords<KeyType>(rank, numRanks, numParticlesGlobal, get<"x">(d), get<"y">(d), get<"z">(d), globalBox);
        transferToHost(d, 0, get<"x">(d).size(), {"x", "y", "z"});
        auto t1 = std::chrono::high_resolution_clock::now();
        if (rank == 0) std::cout << "earlySync " << std::chrono::duration<float>(t1 - t0).count() << std::endl;

        d.resize(d.x.size());

        settings_["numParticlesGlobal"] = double(numParticlesGlobal);
        BuiltinWriter attributeSetter(settings_);
        d.loadOrStoreAttributes(&attributeSetter);
    }

    auto createUniformSphere(int rank, int numRanks, size_t cbrtNumPart, Dataset& simData, IFileReader* reader,
                             const double r_total) const
    {
        using T       = typename Dataset::RealType;
        using KeyType = typename Dataset::KeyType;
        auto& d       = simData.hydro;

        std::vector<T> xBlock, yBlock, zBlock;
        readTemplateBlock(glassBlock, reader, xBlock, yBlock, zBlock);
        size_t blockSize = xBlock.size();

        int               multi1D      = rint(cbrtNumPart / std::cbrt(blockSize));
        cstone::Vec3<int> multiplicity = {multi1D, multi1D, multi1D};

        cstone::Box<T> globalBox(-r_total, r_total, cstone::BoundaryType::open);

        auto [keyStart, keyEnd] = equiDistantSfcSegments<KeyType>(rank, numRanks, 100);

        auto t0 = std::chrono::high_resolution_clock::now();
        assembleCuboid<T>(keyStart, keyEnd, globalBox, multiplicity, xBlock, yBlock, zBlock, d.x, d.y, d.z);
        cutSphere(r_total, d.x, d.y, d.z);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (rank == 0) std::cout << "assembly " << std::chrono::duration<float>(t1 - t0).count() << std::endl;

        return globalBox;
    }

    void resetConstants(InitSettings newSettings) { settings_ = std::move(newSettings); }

    [[nodiscard]] const InitSettings& constants() const override { return settings_; }
};
} // namespace sphexa
