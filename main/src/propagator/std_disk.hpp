
/*! @file
 * @brief A Propagator class for Protoplanetary disk
 *
 * @author Noah Kubli
 */

#pragma once

#include <cstdio>

#include "io/arg_parser.hpp"
#include "ipropagator.hpp"
#include "std_hydro.hpp"
#include "sph/particles_data.hpp"
#include "sph/sph.hpp"

#include "accretion.hpp"
#include "beta_cooling.hpp"
#include "central_force.hpp"
#include "exchange_star_position.hpp"
#include "star_data.hpp"

namespace sphexa
{

using namespace sph;

template<class DomainType, class DataType>
class HydroPropRelax : public HydroProp<DomainType, DataType>
{
    struct Params
    {
        double relaxationTimescale{0};

        template<class Archive>
        void loadOrStoreAttributes(Archive* ar)
        {
            ar->stepAttribute("relaxationTimescale", &relaxationTimescale, 1);
        }
    };
    Params params_;

public:
    HydroPropRelax(std::ostream& output, size_t rank, const InitSettings& settings)
        : HydroProp<DomainType, DataType>(output, rank)
    {
        BuiltinWriter attributeWriter(settings);
        params_.loadOrStoreAttributes(&attributeWriter);
    }

    void computeForces(DomainType& domain, DataType& simData) override
    {
        HydroProp<DomainType, DataType>::computeForces(domain, simData);
        relaxSystem(domain.startIndex(), domain.endIndex(), simData.hydro, params_.relaxationTimescale);
    }

    void load(const std::string& initCond, IFileReader* reader) override
    {
        const std::string path = removeModifiers(initCond);
        if (std::filesystem::exists(path))
        {
            int snapshotIndex = numberAfterSign(initCond, ":");
            reader->setStep(path, snapshotIndex, FileMode::independent);
            params_.loadOrStoreAttributes(reader);
            reader->closeStep();
        }
    }

    void save(IFileWriter* writer) override { params_.loadOrStoreAttributes(writer); }
};

template<class DomainType, class DataType>
class DiskProp : public HydroProp<DomainType, DataType>
{
protected:
    using Base = HydroProp<DomainType, DataType>;
    using Base::timer;

    using T = typename DataType::RealType;

    disk::StarData star;

public:
    DiskProp(std::ostream& output, size_t rank, const InitSettings& settings)
        : Base(output, rank)
    {
        BuiltinWriter attributeWriter(settings);
        star.loadOrStoreAttributes(&attributeWriter);
    }

    void load(const std::string& initCond, IFileReader* reader) override
    {
        const std::string path = removeModifiers(initCond);
        if (std::filesystem::exists(path))
        {
            int snapshotIndex = numberAfterSign(initCond, ":");
            reader->setStep(path, snapshotIndex, FileMode::independent);
            star.loadOrStoreAttributes(reader);
            reader->closeStep();
        }
    }

    void save(IFileWriter* writer) override { star.loadOrStoreAttributes(writer); }

    void computeForces(DomainType& domain, DataType& simData) override
    {
        Base::computeForces(domain, simData);

        auto&        d     = simData.hydro;
        const size_t first = domain.startIndex();
        const size_t last  = domain.endIndex();

        disk::betaCooling(first, last, d, star);
        timer.step("betaCooling");

        disk::computeCentralForce(first, last, d, star);
        timer.step("computeCentralForce");
    }

    void integrate(DomainType& domain, DataType& simData) override
    {
        const size_t first = domain.startIndex();
        const size_t last  = domain.endIndex();
        auto&        d     = simData.hydro;

        disk::duTimestep(first, last, d, star);
        timer.step("duTimestep");

        computeTimestep(first, last, d, star.t_du, d.etaAcc * star.t_star);
        timer.step("Timestep");

        computePositions(Base::groups_.view(), d, domain.box(), d.minDt, {float(d.minDt_m1)});

        bool haveUnconvergedParticles = updateSmoothingLength(Base::groups_.view(), d);
        if (haveUnconvergedParticles && not d.removeUnconvergedParticles)
        {
            throw std::runtime_error("Neighbor search did not converge\n");
        }
        timer.step("UpdateQuantities");

        disk::computeAndExchangeStarPosition(star, d.minDt, d.minDt_m1);
        timer.step("computeAndExchangeStarPosition");

        disk::computeAccretionCondition(first, last, d, star);
        timer.step("computeAccretionCondition");

        disk::exchangeAndAccreteOnStar(star, d.minDt_m1, Base::rank_);
        timer.step("exchangeAndAccreteOnStar");

        if (Base::rank_ == 0)
        {
            std::printf("star position: %lf\t%lf\t%lf\n", star.position[0], star.position[1], star.position[2]);
            std::printf("star mass: %lf\n", star.m);
            std::printf("additional pot. erg.: %lf\n", star.potential);
        }
    }
};

} // namespace sphexa
