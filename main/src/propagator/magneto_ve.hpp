/*
 * MIT License
 *
 * Copyright (c) 2024 CSCS, ETH Zurich, University of Basel, University of Zurich
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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUTh WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUTh NOTh LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENTh SHALL THE
 * AUTHORS OR COPYRIGHTh HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORTh OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file Propagator to be used with Magneto-Hydrodynamics
 *
 * @author Lukas Schmidt
 */

#pragma once

#include "ipropagator.hpp"

#include <variant>

#include "cstone/fields/field_get.hpp"
#include "sph/particles_data.hpp"
#include "sph/magneto_ve/magneto_data.hpp"
#include "sph/sph.hpp"

#include "gravity_wrapper.hpp"

namespace sphexa::magneto
{
using namespace sph;
using util::FieldList;

template<bool avClean, class DomainType, class DataType>
class MagnetoHydroProp : public Propagator<DomainType, DataType>
{
protected:
    using Base = Propagator<DomainType, DataType>;
    using Base::pmReader;
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

    /*! @brief the list of conserved particles fields with values preserved between iterations
     *
     * x, y, z, h and m are automatically considered conserved and must not be specified in this list
     */
    using ConservedFieldsHydro   = FieldList<"temp", "vx", "vy", "vz", "x_m1", "y_m1", "z_m1", "du_m1", "alpha">;
    using ConservedFieldsMagneto = FieldList<"Bx", "By", "Bz", "dBx", "dBy", "dBz", "dBx_m1", "dBy_m1", "dBz_m1",
                                             "psi_ch", "d_psi_ch", "d_psi_ch_m1">;

    //! @brief list of dependent fields, these may be used as scratch space during domain sync
    using DependentFieldsHydro = FieldList<"ax", "ay", "az", "prho", "c", "p", "du", "c11", "c12", "c13", "c22", "c23",
                                           "c33", "xm", "kx", "nc", "gradh">;
    using DependentFieldsMagneto = FieldList<"dvxdx", "dvxdy", " dvxdz", "dvydx", "dvydy", "dvydz", "dvzdx", "dvzdy",
                                             "dvzdz", "divB", "curlB_x", "curlB_y", "curlB_z">;

public:
    MagnetoHydroProp(std::ostream& output, size_t rank)
        : Base(output, rank)
    {
        if (avClean && rank == 0) { std::cout << "AV cleaning is activated" << std::endl; }
    }

    std::vector<std::string> conservedFields() const override
    {
        std::vector<std::string> ret{"x", "y", "z", "h", "m"};
        for_each_tuple([&ret](auto f) { ret.push_back(f.value); }, make_tuple(ConservedFieldsHydro{}));
        for_each_tuple([&ret](auto f) { ret.push_back(f.value); }, make_tuple(ConservedFieldsMagneto{}));
        return ret;
    }

    void activateFields(DataType& simData) override
    {
        auto& d  = simData.hydro;
        auto& md = simData.magneto;
        //! @brief Fields accessed in domain sync (x,y,z,h,m,keys) are not part of extensible lists.
        d.setConserved("x", "y", "z", "h", "m");
        d.setDependent("keys");
        std::apply([&d](auto... f) { d.setConserved(f.value...); }, make_tuple(ConservedFieldsHydro{}));
        std::apply([&d](auto... f) { d.setDependent(f.value...); }, make_tuple(DependentFieldsHydro{}));
        std::apply([&md](auto... f) { md.setConserved(f.value...); }, make_tuple(ConservedFieldsMagneto{}));
        std::apply([&md](auto... f) { md.setDependent(f.value...); }, make_tuple(DependentFieldsMagneto{}));

        d.devData.setConserved("x", "y", "z", "h", "m");
        d.devData.setDependent("keys");
        std::apply([&d](auto... f) { d.devData.setConserved(f.value...); }, make_tuple(ConservedFieldsHydro{}));
        std::apply([&d](auto... f) { d.devData.setDependent(f.value...); }, make_tuple(DependentFieldsHydro{}));
        std::apply([&md](auto... f) { md.devData.setConserved(f.value...); }, make_tuple(ConservedFieldsMagneto{}));
        std::apply([&md](auto... f) { md.devData.setDependent(f.value...); }, make_tuple(DependentFieldsMagneto{}));
    }

    void sync(DomainType& domain, DataType& simData) override
    {
        auto& d  = simData.hydro;
        auto& md = simData.magneto;
        if (d.g != 0.0)
        {
            domain.syncGrav(get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d), get<"m">(d),
                            std::tuple_cat(get<ConservedFieldsHydro>(d), get<ConservedFieldsMagneto>(md)),
                            std::tuple_cat(get<DependentFieldsHydro>(d), get<DependentFieldsMagneto>(md)));
        }
        else
        {
            domain.sync(
                get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d),
                std::tuple_cat(std::tie(get<"m">(d)), get<ConservedFieldsHydro>(d), get<ConservedFieldsMagneto>(md)),
                std::tuple_cat(get<DependentFieldsHydro>(d), get<DependentFieldsMagneto>(md)));
        }
        d.treeView = domain.octreeProperties();
    }

    void computeForces(DomainType& domain, DataType& simData) override
    {
        timer.start();
        pmReader.start();
        sync(domain, simData);
        timer.step("domain::sync");

        auto& d  = simData.hydro;
        auto& md = simData.magneto;
        d.resizeAcc(domain.nParticlesWithHalos());
        md.resizeAcc(domain.nParticlesWithHalos());
        resizeNeighbors(d, domain.nParticles() * d.ngmax);
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        transferToHost(d, first, first + 1, {"m"});
        fill(get<"m">(d), 0, first, d.m[first]);
        fill(get<"m">(d), last, domain.nParticlesWithHalos(), d.m[first]);

        findNeighborsSfc(first, last, d, domain.box());
        computeGroups(first, last, d, domain.box(), groups_);
        timer.step("FindNeighbors");
        pmReader.step();

        computeXMass(groups_.view(), d, domain.box());
        timer.step("XMass");
        domain.exchangeHalos(std::tie(get<"xm">(d)), get<"ax">(d), get<"keys">(d));
        timer.step("mpi::synchronizeHalos");

        release(d, "ay");
        computeVeDefGradh(groups_.view(), d, domain.box());
        timer.step("Normalization & Gradh");

        computeEOS(first, last, d);
        timer.step("EquationOfState");

        domain.exchangeHalos(get<"vx", "vy", "vz", "p", "c", "kx">(d), get<"ax">(d), get<"keys">(d));
        domain.exchangeHalos(get<"Bx", "By", "Bz">(md), get<"ax">(d), get<"keys">(d));
        timer.step("mpi::synchronizeHalos");

        release(d, "az");
        acquire(d, "divv", "curlv");
        sph::magneto::computeIadFullDivvCurlv(groups_.view(), simData, domain.box());
        d.minDtRho = rhoTimestep(first, last, d);
        timer.step("IadVelocityDivCurl");

        domain.exchangeHalos(get<"c11", "c12", "c13", "c22", "c23", "c33", "divv">(d), get<"ax">(d), get<"keys">(d));
        timer.step("mpi::synchronizeHalos");

        computeAVswitches(groups_.view(), d, domain.box());
        timer.step("AVswitches");

        domain.exchangeHalos(get<"alpha", "gradh">(d), get<"ax">(d), get<"keys">(d));
        domain.exchangeHalos(get<"dvxdx", "dvxdy", " dvxdz", "dvydx", "dvydy", "dvydz", "dvzdx", "dvzdy", "dvzdz">(md),
                             get<"ax">(d), get<"keys">(d));
        timer.step("mpi::synchronizeHalos");

        release(d, "divv", "curlv");
        acquire(d, "ay", "az");
        sph::magneto::computeMomentumEnergy<avClean>(groups_.view(), nullptr, simData, domain.box());
        timer.step("MomentumAndEnergy");

        domain.exchangeHalos(get<"divB", "curlB_x", "curlB_y", "curlB_z", "psi_ch">(md), get<"divv">(d),
                             get<"curlv">(d));
        timer.step("mpi::synchronizeHalos");

        sph::magneto::computeInductionAndDissipation(groups_.view(), simData, domain.box());
        timer.step("InductionAndDissipation");
        pmReader.step();

        if (d.g != 0.0)
        {
            auto groups = mHolder_.computeSpatialGroups(d, domain);
            mHolder_.upsweep(d, domain);
            timer.step("Upsweep");
            pmReader.step();
            mHolder_.traverse(groups, d, domain);
            timer.step("Gravity");
            pmReader.step();
        }
    }

    void integrate(DomainType& domain, DataType& simData) override
    {
        auto&  d     = simData.hydro;
        auto&  md    = simData.magneto;
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        computeTimestep(first, last, d);
        timer.step("Timestep");
        computePositions(groups_.view(), d, domain.box(), d.minDt, {float(d.minDt_m1)});
        sph::magneto::integrateMagneticQuantities(groups_.view(), md, d.minDt, d.minDt_m1);
        updateSmoothingLength(groups_.view(), d);
        timer.step("UpdateQuantities");
    }

    void saveFields(IFileWriter* writer, size_t first, size_t last, DataType& simData,
                    const cstone::Box<T>& box) override
    {

        auto& d  = simData.hydro;
        auto& md = simData.magneto;
        d.resize(d.accSize());
        md.resize(md.accSize());
        auto fieldPointersHydro   = d.data();
        auto indicesDoneHydro     = d.outputFieldIndices;
        auto namesDoneHydro       = d.outputFieldNames;
        auto fieldPointersMagneto = md.data();
        auto indicesDoneMagneto   = md.outputFieldIndices;
        auto namesDoneMagneto     = md.outputFieldNames;

        auto output = [&]()
        {
            for (int i = int(indicesDoneHydro.size()) - 1; i >= 0; --i)
            {
                int fidx = indicesDoneHydro[i];
                if (d.isAllocated(fidx))
                {
                    int column = std::find(d.outputFieldIndices.begin(), d.outputFieldIndices.end(), fidx) -
                                 d.outputFieldIndices.begin();
                    transferToHost(d, first, last, {d.fieldNames[fidx]});
                    std::visit([writer, c = column, key = namesDoneHydro[i]](auto field)
                               { writer->writeField(key, field->data(), c); }, fieldPointersHydro[fidx]);
                    indicesDoneHydro.erase(indicesDoneHydro.begin() + i);
                    namesDoneHydro.erase(namesDoneHydro.begin() + i);
                }
            }
            if (!indicesDoneMagneto.empty())
            {
                for (int i = int(indicesDoneMagneto.size()) - 1; i >= 0; --i)
                {
                    int fidx = indicesDoneMagneto[i];
                    if (md.isAllocated(fidx))
                    {
                        int column = std::find(md.outputFieldIndices.begin(), md.outputFieldIndices.end(), fidx) -
                                     md.outputFieldIndices.begin();
                        transferToHost(md, first, last, {md.fieldNames[fidx]});
                        std::visit([writer, c = column, key = namesDoneMagneto[i]](auto field)
                                   { writer->writeField(key, field->data(), c); }, fieldPointersMagneto[fidx]);
                        indicesDoneMagneto.erase(indicesDoneMagneto.begin() + i);
                        namesDoneMagneto.erase(namesDoneMagneto.begin() + i);
                    }
                }
            }
        };

        // first output pass: write everything allocated at the end of computeForces()
        output();

        // second output pass: write temporary quantities produced by the EOS
        release(d, "c11");
        acquire(d, "rho");
        computeEOS(first, last, d);
        output();
        release(d, "rho");
        acquire(d, "c11");

        // third output pass: recover temporary curlv and divv quantities
        release(d, "prho", "c");
        acquire(d, "divv", "curlv");
        // partial recovery of cij in range [first:last] without halos, which are not needed for divv and curlv
        if (!indicesDoneHydro.empty()) { computeIadDivvCurlv(groups_.view(), d, box); }
        output();
        release(d, "divv", "curlv");
        acquire(d, "prho", "c");

        /* The following data is now lost and no longer available in the integration step
         *  c11, c12, c12: halos invalidated
         *  prho, c: destroyed
         */

        if (!indicesDoneHydro.empty() && Base::rank_ == 0)
        {
            std::cout << "WARNING: the following fields are not in use and therefore not output: ";
            for (int fidx = 0; fidx < indicesDoneHydro.size() - 1; ++fidx)
            {
                std::cout << d.fieldNames[fidx] << ",";
            }
            std::cout << d.fieldNames[indicesDoneHydro.back()] << std::endl;
        }
        timer.step("FileOutput");
    }
};
} // namespace sphexa::magneto