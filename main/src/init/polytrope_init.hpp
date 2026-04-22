//
// Created by Noah Kubli on 15.02.2025.
//

#pragma once

#include <map>

#include "sph/eos.hpp"

#include "polytrope/bisect.hpp"
#include "polytrope/polytrope_profile.hpp"
#include "radial_profile.hpp"

namespace sphexa
{

InitSettings polytropeConstants()
{
    return {{"gravConstant", 1.0},
            {"polytrope::r", 4.72108762739756E-01},
            {"polytrope::mTotal", 1e-6},
            {"polytropic_index", 5. / 3.},
            {"minDt", 1e-4},
            {"minDt_m1", 1e-4},
            {"ng0", 100},
            {"ngmax", 150},
            {"eosChoice", sph::EosType::polytropic}};
}

template<class Dataset>
void initPolytropeFields(Dataset& d, const std::map<std::string, double>& constants, double m_part)
{
    constexpr bool gpu = cstone::HaveGpu<typename Dataset::AcceleratorType>{};

    cstone::fill<gpu>(d.m.begin(), d.m.end(), m_part);
    cstone::fill<gpu>(d.du_m1.begin(), d.du_m1.end(), 0.0);
    cstone::fill<gpu>(d.mui.begin(), d.mui.end(), d.muiConst);
    cstone::fill<gpu>(d.alpha.begin(), d.alpha.end(), d.alphamin);

    cstone::fill<gpu>(d.vx.begin(), d.vx.end(), 0.0);
    cstone::fill<gpu>(d.vy.begin(), d.vy.end(), 0.0);
    cstone::fill<gpu>(d.vz.begin(), d.vz.end(), 0.0);

    cstone::fill<gpu>(d.x_m1.begin(), d.x_m1.end(), 0.0);
    cstone::fill<gpu>(d.y_m1.begin(), d.y_m1.end(), 0.0);
    cstone::fill<gpu>(d.z_m1.begin(), d.z_m1.end(), 0.0);

    cstone::fill<gpu>(d.u.begin(), d.u.end(), 0.0);

    generateParticleIDs<gpu>(d.id);
}

template<typename Dataset>
void estimateSmoothingLengths(auto rhoAtRadius, Dataset& d, double m_part, size_t ng0, double r_total)
{
    auto x = toHost(d.x);
    auto y = toHost(d.y);
    auto z = toHost(d.z);

    using Th = decltype(d.h)::value_type;
    std::vector<Th> h(x.size());
    auto            smoothing_length = [rhoAtRadius, m_part, ng0](double r)
    { return 0.5 * std::cbrt(3. * ng0 * m_part / (4. * M_PI * rhoAtRadius(r))); };

    // Find the resolved radius, from which on the smoothing length is capped, such that it does not diverge
    auto boundary_overlap = [&](double r) { return 2. * smoothing_length(r) + r - r_total; };
    const auto [converged, r_resolved] =
        polytrope::find_zero_bisect(boundary_overlap, r_total / 2., r_total, 1e-6 * r_total, 1e-6 * r_total);
    if (!converged) throw std::runtime_error("Find zero not converged");

    const double h_max = smoothing_length(r_resolved);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < d.x.size(); i++)
    {
        const auto radius = std::sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
        h[i]              = std::min(h_max, smoothing_length(radius));
    }
    d.h = std::move(h);
}

template<class Dataset>
class Polytrope : public RadialProfile<Dataset>
{
    using Base = RadialProfile<Dataset>;
    using Base::settings_;

public:
    explicit Polytrope(std::string initBlock, std::string settingsFile, IFileReader* reader)
        : Base(std::move(initBlock), polytropeConstants(), std::move(settingsFile), reader)
    {
        if (not settings_.contains("relaxationTimescale"))
        {
            const double r                   = settings_["polytrope::r"];
            const double mTotal              = settings_["polytrope::mTotal"];
            const double gravConstant        = settings_["gravConstant"];
            const double t_relax             = std::sqrt(r * r * r / (gravConstant * mTotal)) / 3.;
            settings_["relaxationTimescale"] = t_relax;
        }
    }

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cbrtNumPart, Dataset& simData,
                                                 IFileReader* reader) const override
    {
        const double polytropic_index = settings_.at("polytropic_index");
        const double n_polytropic     = 1. / (settings_.at("polytropic_index") - 1.);
        const double m_total          = settings_.at("polytrope::mTotal");
        const double r_total          = settings_.at("polytrope::r");

        auto [rho_r, M_r, polytropic_const] =
            polytrope::computePolytropeProfile(n_polytropic, m_total, r_total, settings_.at("gravConstant"));
        settings_["polytropic_const"] = polytropic_const;

        if (rank == 0)
        {
            std::printf("polytropic constant: %lf\tpolytropic exponent: %lf\n", polytropic_const, polytropic_index);
            std::printf("r_total: %lf\tachieved r: %lf\n", r_total, M_r.y_values.back());
        }

        const double rho_old = m_total / (4. / 3. * M_PI * r_total * r_total * r_total);

        auto polytrope_transformation = [M_r, rho_old](auto old_radius)
        {
            const auto old_volume    = 4. * M_PI / 3. * old_radius * old_radius * old_radius;
            const auto enclosed_mass = old_volume * rho_old;
            const auto new_radius    = M_r(enclosed_mass);
            const auto factor        = new_radius / old_radius;
            return factor;
        };

        auto globalBox = Base::init(rank, numRanks, cbrtNumPart, simData, reader, r_total, polytrope_transformation);

        const double m_part = m_total / settings_.at("numParticlesGlobal");
        estimateSmoothingLengths(rho_r, simData.hydro, m_part, settings_.at("ng0"), r_total);
        initPolytropeFields(simData.hydro, settings_, m_part);

        return globalBox;
    }
};
} // namespace sphexa
