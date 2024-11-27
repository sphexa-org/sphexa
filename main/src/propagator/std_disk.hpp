
/*! @file
 * @brief A Propagator class for Protoplanetary disk
 *
 * @author Noah Kubli
 */

#pragma once

#include <variant>

#include "cstone/fields/field_get.hpp"
#include "gravity_wrapper.hpp"
#include "io/arg_parser.hpp"
#include "ipropagator.hpp"
#include "sph/particles_data.hpp"
#include "sph/sph.hpp"

#include "accretion.hpp"
#include "betaCooling.hpp"
#include "computeCentralForce.hpp"
#include "eos_polytropic.hpp"
#include "exchangeStarPosition.hpp"
#include "star_data.hpp"

namespace sphexa
{

using namespace sph;
using util::FieldList;

template<class DomainType, class DataType>
class DiskProp : public Propagator<DomainType, DataType>
{
protected:
    using Base = Propagator<DomainType, DataType>;
    using Base::timer;

    using T             = typename DataType::RealType;
    using KeyType       = typename DataType::KeyType;
    using Tmass         = typename DataType::HydroData::Tmass;
    using MultipoleType = ryoanji::CartesianQuadrupole<Tmass>;

    using Acc       = typename DataType::AcceleratorType;
    using MHolder_t = typename cstone::AccelSwitchType<Acc, MultipoleHolderCpu, MultipoleHolderGpu>::template type<
        MultipoleType, DomainType, typename DataType::HydroData>;

    MHolder_t      mHolder_;
    GroupData<Acc> groups_;

    StarData star;

    /*! @brief the list of conserved particles fields with values preserved between iterations
     *
     * x, y, z, h and m are automatically considered conserved and must not be specified in this list
     */
    using ConservedFields = FieldList<"u", "vx", "vy", "vz", "x_m1", "y_m1", "z_m1", "du_m1">;

    //! @brief the list of dependent particle fields, these may be used as scratch space during domain sync
    using DependentFields =
        FieldList<"rho", "p", "c", "ax", "ay", "az", "du", "c11", "c12", "c13", "c22", "c23", "c33", "nc">;

public:
    DiskProp(std::ostream& output, size_t rank, const InitSettings& settings)
        : Base(output, rank)
    {
    }

    std::vector<std::string> conservedFields() const override
    {
        std::vector<std::string> ret{"x", "y", "z", "h", "m"};
        for_each_tuple([&ret](auto f) { ret.push_back(f.value); }, make_tuple(ConservedFields{}));
        return ret;
    }
    void load(const std::string& initCond, IFileReader* reader) override
    {
        // Read star position from hdf5 File
        std::string path = removeModifiers(initCond);
        if (std::filesystem::exists(path))
        {
            int snapshotIndex = numberAfterSign(initCond, ":");
            reader->setStep(path, snapshotIndex, FileMode::independent);
            star.loadOrStoreAttributes(reader);
            reader->closeStep();
            printf("star position: %lf\t%lf\t%lf\n", star.position[0], star.position[1], star.position[2]);
            printf("star mass: %lf\n", star.m);
        }
    }
    void save(IFileWriter* writer) override { star.loadOrStoreAttributes(writer); }

    void activateFields(DataType& simData) override
    {
        auto& d = simData.hydro;

        //! @brief Fields accessed in domain sync are not part of extensible lists.
        d.setConserved("x", "y", "z", "h", "m");
        d.setDependent("keys");
        std::apply([&d](auto... f) { d.setConserved(f.value...); }, make_tuple(ConservedFields{}));
        std::apply([&d](auto... f) { d.setDependent(f.value...); }, make_tuple(DependentFields{}));

        d.devData.setConserved("x", "y", "z", "h", "m");
        d.devData.setDependent("keys");
        std::apply([&d](auto... f) { d.devData.setConserved(f.value...); }, make_tuple(ConservedFields{}));
        std::apply([&d](auto... f) { d.devData.setDependent(f.value...); }, make_tuple(DependentFields{}));
    }

    void sync(DomainType& domain, DataType& simData) override
    {
        auto& d = simData.hydro;
        if (d.g != 0.0)
        {
            domain.syncGrav(get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d), get<"m">(d),
                            get<ConservedFields>(d), get<DependentFields>(d));
        }
        else
        {
            domain.sync(get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d),
                        std::tuple_cat(std::tie(get<"m">(d)), get<ConservedFields>(d)), get<DependentFields>(d));
        }
        d.treeView = domain.octreeProperties();
    }

    void eosSwitch(size_t first, size_t last, DataType& simData, const StarData& star)
    {
        if (star.use_polytropic_eos) { planet::computePolytropicEOS_HydroStd(first, last, simData.hydro, star); }
        else { computeEOS_HydroStd(first, last, simData.hydro); }
    }

    void computeForces(DomainType& domain, DataType& simData) override
    {
        timer.start();

        sync(domain, simData);
        timer.step("domain::sync");

        auto& d = simData.hydro;
        d.resize(domain.nParticlesWithHalos());
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        domain.exchangeHalos(std::tie(get<"m">(d)), get<"ax">(d), get<"ay">(d));

        resizeNeighbors(d, domain.nParticles() * d.ngmax);
        findNeighborsSfc(first, last, d, domain.box());
        computeGroups(first, last, d, domain.box(), groups_);
        timer.step("FindNeighbors");

        computeDensity(groups_.view(), d, domain.box());
        timer.step("Density");
        eosSwitch(first, last, simData, star);
        timer.step("EquationOfState");

        domain.exchangeHalos(get<"vx", "vy", "vz", "rho", "p", "c">(d), get<"ax">(d), get<"ay">(d));
        timer.step("mpi::synchronizeHalos");

        computeIAD(groups_.view(), d, domain.box());
        timer.step("IAD");

        domain.exchangeHalos(get<"c11", "c12", "c13", "c22", "c23", "c33">(d), get<"ax">(d), get<"ay">(d));
        timer.step("mpi::synchronizeHalos");

        computeMomentumEnergySTD(groups_.view(), d, domain.box());
        timer.step("MomentumEnergyIAD");

        if (d.g != 0.0)
        {
            auto groups = mHolder_.computeSpatialGroups(d, domain);
            mHolder_.upsweep(d, domain);
            timer.step("Upsweep");
            mHolder_.traverse(groups, d, domain);
            timer.step("Gravity");
        }

        planet::betaCooling(first, last, d, star);
        timer.step("betaCooling");

        planet::duTimestep(first, last, d, star);

        planet::computeCentralForce(first, last, d, star);
        timer.step("computeCentralForce");
    }

    void integrate(DomainType& domain, DataType& simData) override
    {
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();
        auto&  d     = simData.hydro;

        computeTimestep(first, last, d, star.t_du);
        timer.step("Timestep");

        computePositions(groups_.view(), d, domain.box(), d.minDt, {float(d.minDt_m1)});
        updateSmoothingLength(groups_.view(), d);
        timer.step("UpdateQuantities");

        planet::computeAndExchangeStarPosition(star, d.minDt, d.minDt_m1, Base::rank_);
        timer.step("computeAndExchangeStarPosition");

        planet::computeAccretionCondition(first, last, d, star);
        timer.step("computeAccretionCondition");

        planet::exchangeAndAccreteOnStar(star, d.minDt_m1, Base::rank_);
        timer.step("exchangeAndAccreteOnStar");

        if (Base::rank_ == 0)
        {
            printf("star position: %lf\t%lf\t%lf\n", star.position[0], star.position[1], star.position[2]);
            printf("star mass: %lf\n", star.m);
            printf("additional pot. erg.: %lf\n", star.potential);
        }
    }

    void saveFields(IFileWriter* writer, size_t first, size_t last, DataType& simData,
                    const cstone::Box<T>& /*box*/) override
    {
        auto output = [&](auto& d)
        {
            auto fieldPointers = d.data();
            auto indicesDone   = d.outputFieldIndices;
            auto namesDone     = d.outputFieldNames;

            for (int i = int(indicesDone.size()) - 1; i >= 0; --i)
            {
                int fidx = indicesDone[i];
                if (d.isAllocated(fidx))
                {
                    int column = std::find(d.outputFieldIndices.begin(), d.outputFieldIndices.end(), fidx) -
                                 d.outputFieldIndices.begin();
                    transferToHost(d, first, last, {d.fieldNames[fidx]});
                    std::visit([writer, c = column, key = namesDone[i]](auto field)
                               { writer->writeField(key, field->data(), c); },
                               fieldPointers[fidx]);
                    indicesDone.erase(indicesDone.begin() + i);
                    namesDone.erase(namesDone.begin() + i);
                }
            }

            if (!indicesDone.empty() && Base::rank_ == 0)
            {
                std::cout << "WARNING: the following fields are not in use and therefore not output: ";
                for (int fidx = 0; fidx < indicesDone.size() - 1; ++fidx)
                {
                    std::cout << d.fieldNames[fidx] << ",";
                }
                std::cout << d.fieldNames[indicesDone.back()] << std::endl;
            }
        };

        output(simData.hydro);
        output(simData.chem);
    }
};

} // namespace sphexa
