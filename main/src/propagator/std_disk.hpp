
/*! @file
 * @brief A Propagator class for Protoplanetary disk
 *
 * @author Noah Kubli
 */

#pragma once

#include <cstdio>

#include "cstone/fields/field_get.hpp"
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
using util::FieldList;

template<class DomainType, class DataType>
class DiskProp : public HydroProp<DomainType, DataType>
{
protected:
    using Base = HydroProp<DomainType, DataType>;
    using Base::timer;

    using T = typename DataType::RealType;

public:
    DiskProp(std::ostream& output, size_t rank, const InitSettings& settings)
        : Base(output, rank)
    {
    }

    void activateFields(DataType& simData) override
    {
        simData.star.active = true;
        Base::activateFields(simData);
    }

    void computeForces(DomainType& domain, DataType& simData) override
    {
        Base::computeForces(domain, simData);

        auto&        d     = simData.hydro;
        const size_t first = domain.startIndex();
        const size_t last  = domain.endIndex();

        disk::betaCooling(first, last, d, simData.star);
        timer.step("betaCooling");

        disk::computeCentralForce(first, last, d, simData.star);
        timer.step("computeCentralForce");
    }

    void integrate(DomainType& domain, DataType& simData) override
    {
        const size_t first = domain.startIndex();
        const size_t last  = domain.endIndex();
        auto&        d     = simData.hydro;
        auto&        star  = simData.star;

        disk::duTimestep(first, last, d, star);
        timer.step("duTimestep");

        computeTimestep(first, last, d, star.t_du, d.etaAcc * star.t_star);
        timer.step("Timestep");

        computePositions(Base::groups_.view(), d, domain.box(), d.minDt, {float(d.minDt_m1)});
        updateSmoothingLength(Base::groups_.view(), d);
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
    void saveFields(IFileWriter* writer, size_t first, size_t last, DataType& simData,
                    const cstone::Box<T>& box) override
    {
        Base::saveFields(writer, first, last, simData, box);
        simData.star.loadOrStoreAttributes(writer);
    }
};

} // namespace sphexa
