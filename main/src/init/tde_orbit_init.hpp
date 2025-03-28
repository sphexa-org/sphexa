//
// Created by Noah Kubli on 10.03.2025.
//

#pragma once

#include "isim_init.hpp"
#include "file_init.hpp"
#include "settings.hpp"
#include "utils.hpp"

namespace sphexa
{
InitSettings tdeOrbitConstants()
{
    InitSettings ret{{"tde-orbit::beta_impact", 1.0},
                     {"tde-orbit::r0_per_periapsis", 5.0},
                     {"star::potentialType", disk::StarPotentialType::einstein_precession},
                     {"star::m", 1.0},
                     {"star::inner_size", 0.0},
                     {"star::fixed_star", 1},
                     {"star::x", 0.},
                     {"star::y", 0.},
                     {"star::z", 0.},
                     {"star::x_m1", 0.},
                     {"star::y_m1", 0.},
                     {"star::z_m1", 0.}};
    return ret;
}

/*! @brief Compute the position and the velocity of an object in a parabolic orbit.
 * The function assumes that the orbital plane is the xy-plane and the periapsis is at y = 0.
 * @param m_b Mass of the central object (black hole)
 * @param pos_b Position of the centrla object
 * @param r_periapsis Periapsis distance of the orbit
 * @param r0_per_periapsis Initial distance from the central object measured in periapsis distances
 * @param G Gravitational constant
 * */
auto computeParabolicOrbit(double m_b, const cstone::Vec3<double>& pos_b, double r_periapsis, double r0_per_periapsis,
                           double G)
{
    //! @brief 1 + cos(theta0); while theta is the angle of the object in the orbital (xy) plane, measured from
    //! periapsis counter-clock wise
    const double theta0_factor = 2. / r0_per_periapsis;

    const double r0 = r0_per_periapsis * r_periapsis;

    // I choose the negative solution as a starting point; so the object moves from negative to positive theta.
    const double theta0 = -std::acos(theta0_factor - 1.);

    // Periapsis is chosen to be at y = 0.
    const double x0 = r0 * std::cos(theta0);
    const double y0 = r0 * std::sin(theta0);

    //! @brief The angle of the velocity measured from the orbiter. 0 if in the tangential direction (circular orbit),
    //! counter-clock wise
    const double phi0 = -theta0 / 2.;
    //! @brief The angle of the velocity measured from the central object.
    const double phi0_centre = theta0 + std::numbers::pi_v<double> / 2. + phi0;

    const double v = std::sqrt(G * m_b / r_periapsis * theta0_factor);

    const double vx = v * std::cos(phi0_centre);
    const double vy = v * std::sin(phi0_centre);

    return std::tuple{cstone::Vec3<double>{x0, y0, 0.} + pos_b, cstone::Vec3<double>{vx, vy, 0.}};
}

void printMap(const auto& map)
{
    for (const auto& elem : map)
    {
        std::cout << elem.first << " " << elem.second << "\n";
    }
}

//! @brief Displace system by a position and velocity
template<typename Dataset>
void displaceSystem(Dataset& simData, const cstone::Vec3<double>& X, const cstone::Vec3<double>& V)
{
    auto& d = simData.hydro;
#pragma omp parallel for
    for (size_t i = 0; i < d.x.size(); i++)
    {
        d.x[i] += X[0];
        d.y[i] += X[1];
        d.z[i] += X[2];
        d.vx[i] += V[0];
        d.vy[i] += V[1];
        d.vz[i] += V[2];
        d.x_m1[i] = d.vx[i] * d.minDt;
        d.y_m1[i] = d.vy[i] * d.minDt;
        d.z_m1[i] = d.vz[i] * d.minDt;
    }
}
template<typename Dataset>
class TDEOrbitInit : public ISimInitializer<Dataset>
{
    InitSettings settings_;
    std::string  h5_fname;
    int          initStep = -1;

public:
    explicit TDEOrbitInit(const std::string& filename, int initStep, IFileReader* reader)
        : h5_fname(filename)
        , initStep(initStep)
    {
        BuiltinReader extractor(settings_);
        // load default settings
        Dataset simData;
        simData.hydro.loadOrStoreAttributes(&extractor);
        simData.star.loadOrStoreAttributes(&extractor);

        // settings specified in tdeOrbitConstants()
        for (const auto& kv : tdeOrbitConstants())
        {
            settings_[kv.first] = kv.second;
        }

        // load settings from init file. Must contain mTotal and r fields of the orbiter.
        readFileAttributes(settings_, filename, reader, true);
        printMap(settings_);
        if (!settings_.contains("polytrope::r"))
        {
            throw std::runtime_error("init file must contain attribute polytrope::r");
        }
        if (!settings_.contains("polytrope::mTotal"))
        {
            throw std::runtime_error("init file must contain attribute polytrope::mTotal");
        }
    }

    [[nodiscard]] const InitSettings& constants() const override { return settings_; }

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t n, Dataset& simData,
                                                 IFileReader* reader) const override
    {
        BuiltinWriter attributeSetter(settings_);
        simData.hydro.loadOrStoreAttributes(&attributeSetter);
        simData.star.loadOrStoreAttributes(&attributeSetter);

        reader->setStep(h5_fname, initStep, FileMode::collective);
        auto box = restoreData(reader, simData);
        reader->closeStep();

        // step parameters that have to be overriden
        simData.hydro.relaxationTimescale = 0.;
        simData.hydro.iteration           = 0;
        simData.hydro.ttot                = 0.0;
        simData.hydro.minDt               = 1e-9;
        simData.hydro.minDt_m1            = 1e-9;

        const double               mTotal = settings_.at("polytrope::mTotal");
        const double               r      = settings_.at("polytrope::r");
        const double               m_b    = settings_.at("star::m");
        const cstone::Vec3<double> pos_b  = {settings_.at("star::x"), settings_.at("star::y"), settings_.at("star::z")};

        const double r_tidal          = r * std::pow(m_b / mTotal, 1. / 3.);
        const double r_periapsis      = r_tidal / settings_.at("tde-orbit::beta_impact");
        const double r0_per_periapsis = settings_.at("tde-orbit::r0_per_periapsis");

        const auto [X, V] = computeParabolicOrbit(m_b, pos_b, r_periapsis, r0_per_periapsis, simData.hydro.g);
        displaceSystem(simData, X, V);
        std::printf("Placed orbiter: \n");
        std::printf("x: %lf, %lf, %lf\n", X[0], X[1], X[2]);
        std::printf("v: %lf, %lf, %lf\n", V[0], V[1], V[2]);

        return box;
    }
};
} // namespace sphexa
