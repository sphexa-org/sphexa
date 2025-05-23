/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief A Propagator class for standard SPH including Grackle chemistry and cooling
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 * @author Noah Kubli <noah.kubli@uzh.ch>
 */

#pragma once

#include "cstone/fields/field_get.hpp"
#include "cstone/util/value_list.hpp"
#include "io/arg_parser.hpp"
#include "sph/particles_data.hpp"
#include "sph/sph.hpp"

#include "cooling/cooler.hpp"
#include "cooling/eos_cooling.hpp"

#include "std_hydro.hpp"
#include "gravity_wrapper.hpp"

namespace sphexa
{

using namespace sph;
using util::FieldList;

template<class DomainType, class DataType>
class HydroGrackleProp final : public HydroProp<DomainType, DataType>
{
    using Base = HydroProp<DomainType, DataType>;
    using Base::groups_;
    using Base::timer;

    using T        = typename DataType::RealType;
    using KeyType  = typename DataType::KeyType;
    using ChemData = typename DataType::ChemData;

    cooling::Cooler<T> cooling_data;

    /*! @brief the list of conserved particles fields with values preserved between iterations
     *
     * x, y, z, h and m are automatically considered conserved and must not be specified in this list
     */
    using ConservedFields = FieldList<"u", "vx", "vy", "vz", "x_m1", "y_m1", "z_m1", "du_m1", "id">;

    //! @brief the list of dependent particle fields, these may be used as scratch space during domain sync
    using DependentFields =
        FieldList<"rho", "p", "c", "ax", "ay", "az", "du", "c11", "c12", "c13", "c22", "c23", "c33", "nc">;

    //! @brief All fields listed in Chemistry data are used. This could be overridden with a sublist if desired
    using CoolingFields = typename cooling::Cooler<T>::CoolingFields;

public:
    HydroGrackleProp(std::ostream& output, size_t rank, const InitSettings& settings)
        : Base(output, rank)
    {
        BuiltinWriter attributeSetter(settings);
        cooling_data.loadOrStoreAttributes(&attributeSetter);
        cooling_data.init(0);
    }

    void load(const std::string& initCond, IFileReader* reader) override
    {
        std::string path = removeModifiers(initCond);
        if (std::filesystem::exists(path))
        {
            int snapshotIndex = numberAfterSign(initCond, ":");
            reader->setStep(path, snapshotIndex, FileMode::independent);
            cooling_data.loadOrStoreAttributes(reader);
            reader->closeStep();
        }
        else if (path != "evrard-cooling")
        {
            throw std::runtime_error("Cooling propagator has to be used with the evrard-cooling builtin test-case "
                                     "or a suitable init file");
        }
    }

    void save(IFileWriter* writer) override { cooling_data.loadOrStoreAttributes(writer); }

    std::vector<std::string> conservedFields() const override
    {
        std::vector<std::string> ret{"x", "y", "z", "h", "m"};
        for_each_tuple([&ret](auto f) { ret.push_back(f.value); }, make_tuple(ConservedFields{}));
        for_each_tuple([&ret](auto f) { ret.push_back(f.value); }, make_tuple(CoolingFields{}));
        return ret;
    }

    void activateFields(DataType& simData) override
    {
        auto& d = simData.hydro;

        //! @brief Fields accessed in domain sync are not part of extensible lists.
        d.setConserved("x", "y", "z", "h", "m");
        d.setDependent("keys");
        std::apply([&d](auto... f) { d.setConserved(f.value...); }, make_tuple(ConservedFields{}));
        std::apply([&d](auto... f) { d.setDependent(f.value...); }, make_tuple(DependentFields{}));
        std::apply([&simData](auto... f) { simData.chem.setConserved(f.value...); }, make_tuple(CoolingFields{}));

        d.devData.setConserved("x", "y", "z", "h", "m");
        d.devData.setDependent("keys");
        std::apply([&d](auto... f) { d.devData.setConserved(f.value...); }, make_tuple(ConservedFields{}));
        std::apply([&d](auto... f) { d.devData.setDependent(f.value...); }, make_tuple(DependentFields{}));
    }

    void sync(DomainType& domain, DataType& simData) override
    {
        using ChemRealType = typename DataType::ChemData::RealType;

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

        std::vector<ChemRealType> scratch1, scratch2;
        domain.reapplySync(get<CoolingFields>(simData.chem), scratch1, scratch2, get<"nc">(d));
        d.treeView = domain.octreeProperties();
    }

    void computeForces(DomainType& domain, DataType& simData) override
    {
        auto& d = simData.hydro;
        timer.start();

        sync(domain, simData);
        // halo exchange for masses, allows for particles with variable masses
        domain.exchangeHalos(std::tie(get<"m">(d)), get<"ax">(d), get<"ay">(d));
        timer.step("domain::sync");
        d.resize(domain.nParticlesWithHalos());
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        resizeNeighbors(d, domain.nParticles() * d.ngmax);
        findNeighborsSfc(first, last, d, domain.box());
        computeGroups(first, last, d, domain.box(), groups_);
        timer.step("FindNeighbors");

        computeDensity(groups_.view(), d, domain.box());
        timer.step("Density");

        transferToHost(d, first, last, {"rho", "u"});

        eos_cooling(first, last, d, simData.chem, cooling_data);
        transferToDevice(d, first, last, {"p", "c"});
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
            auto groups = Base::mHolder_.computeSpatialGroups(d, domain);
            Base::mHolder_.upsweep(d, domain);
            timer.step("Upsweep");
            Base::mHolder_.traverse(groups, d, domain);
            timer.step("Gravity");
        }
    }

    void integrate(DomainType& domain, DataType& simData) override
    {
        auto&  d     = simData.hydro;
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        auto minDtCooling = cooling::coolingTimestep(first, last, d, cooling_data, simData.chem);
        computeTimestep(first, last, d, minDtCooling);
        timer.step("Timestep");

        transferToHost(d, first, last, {"du"});

        cooling_data.cool_particles(T(d.minDt), d.rho.data(), d.u.data(),
                                    cstone::getPointers(get<CoolingFields>(simData.chem), 0), d.du.data(), first, last);

        transferToDevice(d, first, last, {"du"});
        timer.step("GRACKLE chemistry and cooling");

        computePositions(groups_.view(), d, domain.box(), d.minDt, {float(d.minDt_m1)});
        timer.step("UpdateQuantities");
        updateSmoothingLength(groups_.view(), d);
        timer.step("UpdateSmoothingLength");
    }
};

} // namespace sphexa
