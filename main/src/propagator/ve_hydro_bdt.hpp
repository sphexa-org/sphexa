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
 * @brief A Propagator class for modern SPH with generalized volume elements
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 * @author Jose A. Escartin <ja.escartin@gmail.com>
 */

#pragma once

#include <variant>

#include "cstone/cuda/device_vector.h"
#include "cstone/fields/field_get.hpp"
#include "io/arg_parser.hpp"
#include "sph/particles_data.hpp"
#include "sph/sph.hpp"
#include "sph/ts_rungs.hpp"

#include "ipropagator.hpp"
#include "gravity_wrapper.hpp"

namespace sphexa
{

using namespace sph;
using util::FieldList;

template<bool avClean, class DomainType, class DataType>
class HydroVeBdtProp : public Propagator<DomainType, DataType>
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
    template<class VType>
    using AccVector = typename cstone::AccelSwitchType<Acc, std::vector, cstone::DeviceVector>::template type<VType>;

    MHolder_t mHolder_;

    //! @brief groups sorted by ascending SFC keys
    GroupData<Acc>        groups_;
    AccVector<float>      groupDt_;
    AccVector<LocalIndex> groupIndices_;

    //! @brief groups sorted by ascending time-step
    GroupData<Acc>                               tsGroups_;
    std::array<GroupView, Timestep::maxNumRungs> rungs_;
    GroupView                                    activeRungs_;

    //! brief timestep information rungs
    Timestep timestep_, prevTimestep_;
    //! number of initial steps to disable block time-steps
    int safetySteps{0};

    //! @brief no dependent fields can be temporarily reused as scratch space for halo exchanges
    AccVector<LocalIndex> haloRecvScratch;

    /*! @brief the list of conserved particles fields with values preserved between iterations
     *
     * x, y, z, h and m are automatically considered conserved and must not be specified in this list
     */
    using ConservedFields = FieldList<"temp", "vx", "vy", "vz", "x_m1", "y_m1", "z_m1", "du_m1", "alpha", "rung", "id">;

    //! @brief list of dependent fields, these may be used as scratch space during domain sync
    using DependentFields_ = FieldList<"ax", "ay", "az", "prho", "c", "du", "c11", "c12", "c13", "c22", "c23", "c33",
                                       "xm", "kx", "nc", "divv", "gradh">;

    //! @brief velocity gradient fields will only be allocated when avClean is true
    using GradVFields = FieldList<"dV11", "dV12", "dV13", "dV22", "dV23", "dV33">;

    //! @brief what will be allocated based AV cleaning choice
    using DependentFields =
        std::conditional_t<avClean, decltype(DependentFields_{} + GradVFields{}), decltype(DependentFields_{})>;

    //! @brief Return rung of current block time-step
    static int activeRung(int substep, int numRungs)
    {
        if (substep == 0 || substep >= (1 << (numRungs - 1))) { return 0; }
        else { return cstone::butterfly(substep); }
    }

public:
    HydroVeBdtProp(std::ostream& output, size_t rank, const InitSettings& settings)
        : Base(output, rank)
    {
        if (not cstone::HaveGpu<Acc>{}) { throw std::runtime_error("This propagator is not supported on CPUs\n"); }
        if (avClean && rank == 0) { std::cout << "AV cleaning is activated" << std::endl; }
        try
        {
            timestep_.dt_m1[0] = settings.at("minDt");
        }
        catch (const std::out_of_range&)
        {
            std::cout << "Init settings miss the following parameter: minDt" << std::endl;
            throw;
        }
    }

    std::vector<std::string> conservedFields() const override
    {
        std::vector<std::string> ret{"x", "y", "z", "h", "m"};
        for_each_tuple([&ret](auto f) { ret.push_back(f.value); }, make_tuple(ConservedFields{}));
        return ret;
    }

    void activateFields(DataType& simData) override
    {
        auto& d = simData.hydro;
        //! @brief Fields accessed in domain sync (x,y,z,h,m,keys) are not part of extensible lists.
        d.setConserved("x", "y", "z", "h", "m");
        d.setDependent("keys");
        std::apply([&d](auto... f) { d.setConserved(f.value...); }, make_tuple(ConservedFields{}));
        std::apply([&d](auto... f) { d.setDependent(f.value...); }, make_tuple(DependentFields{}));

        d.devData.setConserved("x", "y", "z", "h", "m");
        d.devData.setDependent("keys");
        std::apply([&d](auto... f) { d.devData.setConserved(f.value...); }, make_tuple(ConservedFields{}));
        std::apply([&d](auto... f) { d.devData.setDependent(f.value...); }, make_tuple(DependentFields{}));
    }

    void save(IFileWriter* writer) override { timestep_.loadOrStore(writer, "ts::"); }

    void load(const std::string& initCond, IFileReader* reader) override
    {
        int         step = numberAfterSign(initCond, ":");
        std::string path = removeModifiers(initCond);
        // The file does not exist, we're starting from scratch. Nothing to do.
        if (!std::filesystem::exists(path)) { return; }

        reader->setStep(path, step, FileMode::independent);
        timestep_.loadOrStore(reader, "ts::");
        reader->closeStep();

        int numSplits = numberAfterSign(initCond, ",");
        if (numSplits > 0) { timestep_.dt_m1[0] /= 100 * numSplits; }
        if (numSplits > 0) { safetySteps = 1000; }
    }

    void fullSync(DomainType& domain, DataType& simData)
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

        d.resizeAcc(domain.nParticlesWithHalos());
        resizeNeighbors(d, domain.nParticles() * d.ngmax);

        computeGroups(domain.startIndex(), domain.endIndex(), d, domain.box(), groups_);
        activeRungs_ = groups_.view();

        reallocate(groups_.numGroups, d.getAllocGrowthRate(), groupDt_, groupIndices_);
        fill(groupDt_, 0, groupDt_.size(), std::numeric_limits<float>::max());
    }

    void partialSync(DomainType& domain, DataType& simData)
    {
        auto& d = simData.hydro;
        domain.exchangeHalos(get<"x", "y", "z", "h">(d), get<"keys">(d), haloRecvScratch);
        if (d.g != 0.0)
        {
            domain.updateExpansionCenters(get<"x">(d), get<"y">(d), get<"z">(d), get<"m">(d), get<"keys">(d),
                                          haloRecvScratch);
        }

        //! @brief increase tree-cell search radius for each substep to account for particles drifting out of cells
        d.treeView.searchExtFactor *= 1.012;

        int highestRung = cstone::butterfly(timestep_.substep);
        activeRungs_    = makeSlicedView(tsGroups_.view(), timestep_.rungRanges[0], timestep_.rungRanges[highestRung]);
    }

    void sync(DomainType& domain, DataType& simData) override
    {
        domain.setTreeConv(true);
        domain.setHaloFactor(1.0 + float(timestep_.numRungs) / 40);

        if (activeRung(timestep_.substep, timestep_.numRungs) == 0) { fullSync(domain, simData); }
        else { partialSync(domain, simData); }
    }

    bool isSynced() override { return activeRung(timestep_.substep, timestep_.numRungs) == 0; }

    void computeForces(DomainType& domain, DataType& simData) override
    {
        timer.start();
        pmReader.start();
        sync(domain, simData);
        timer.step("domain::sync");

        auto&  d     = simData.hydro;
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        transferToHost(d, first, first + 1, {"m"});
        fill(get<"m">(d), 0, first, d.m[first]);
        fill(get<"m">(d), last, domain.nParticlesWithHalos(), d.m[first]);

        findNeighborsSfc(first, last, d, domain.box());
        timer.step("FindNeighbors");
        pmReader.step();

        computeXMass(activeRungs_, d, domain.box());
        timer.step("XMass");
        domain.exchangeHalos(std::tie(get<"xm">(d)), get<"keys">(d), haloRecvScratch);
        timer.step("mpi::synchronizeHalos");

        computeVeDefGradh(activeRungs_, d, domain.box());
        timer.step("Normalization & Gradh");

        computeEOS(first, last, d);
        timer.step("EquationOfState");

        domain.exchangeHalos(get<"vx", "vy", "vz", "prho", "c", "kx">(d), get<"keys">(d), haloRecvScratch);
        timer.step("mpi::synchronizeHalos");

        computeIadDivvCurlv(activeRungs_, d, domain.box());
        groupDivvTimestep(activeRungs_, rawPtr(groupDt_), d);
        timer.step("IadVelocityDivCurl");

        domain.exchangeHalos(get<"c11", "c12", "c13", "c22", "c23", "c33", "divv">(d), get<"keys">(d), haloRecvScratch);
        timer.step("mpi::synchronizeHalos");

        computeAVswitches(activeRungs_, d, domain.box());
        timer.step("AVswitches");

        if (avClean)
        {
            domain.exchangeHalos(get<"dV11", "dV12", "dV22", "dV23", "dV33", "alpha">(d), get<"keys">(d),
                                 haloRecvScratch);
        }
        else { domain.exchangeHalos(std::tie(get<"alpha">(d)), get<"keys">(d), haloRecvScratch); }
        timer.step("mpi::synchronizeHalos");

        computeMomentumEnergy<avClean>(activeRungs_, rawPtr(groupDt_), d, domain.box());
        timer.step("MomentumAndEnergy");
        pmReader.step();

        if (d.g != 0.0)
        {
            bool      isNewHierarchy = activeRung(timestep_.substep, timestep_.numRungs) == 0;
            GroupView gravGroup      = isNewHierarchy ? mHolder_.computeSpatialGroups(d, domain) : activeRungs_;

            mHolder_.upsweep(d, domain);
            timer.step("Upsweep");
            pmReader.step();
            mHolder_.traverse(gravGroup, d, domain);
            timer.step("Gravity");
            pmReader.step();
        }
        groupAccTimestep(activeRungs_, rawPtr(groupDt_), d);
    }

    void computeRungs(DataType& simData)
    {
        auto& d        = simData.hydro;
        int   highRung = activeRung(timestep_.substep, timestep_.numRungs);

        if (highRung == 0)
        {
            prevTimestep_ = timestep_;
            float maxDt   = timestep_.dt_m1[0] * d.maxDtIncrease;
            timestep_ = rungTimestep(rawPtr(groupDt_), rawPtr(groupIndices_), groups_.numGroups, maxDt, get<"keys">(d));

            if (safetySteps > 0)
            {
                timestep_.numRungs = 1;
                std::fill(timestep_.rungRanges.begin() + 1, timestep_.rungRanges.end(), groups_.numGroups);
                safetySteps--;
            }
        }
        else
        {
            auto [dt, rungRanges] = minimumGroupDt(timestep_, rawPtr(groupDt_), rawPtr(groupIndices_),
                                                   timestep_.rungRanges[highRung], get<"keys">(d));
            timestep_.nextDt      = dt;
            std::copy(rungRanges.begin(), rungRanges.begin() + highRung, timestep_.rungRanges.begin());
        }

        if (highRung == 0 || highRung > 1)
        {
            if (highRung > 1) { swap(groups_, tsGroups_); }
            if constexpr (cstone::HaveGpu<Acc>{})
            {
                extractGroupGpu(groups_.view(), rawPtr(groupIndices_), 0, timestep_.rungRanges.back(), tsGroups_);
            }
        }

        for (int r = 0; r < timestep_.numRungs; ++r)
        {
            rungs_[r] = makeSlicedView(tsGroups_.view(), timestep_.rungRanges[r], timestep_.rungRanges[r + 1]);
        }
    }

    void integrate(DomainType& domain, DataType& simData) override
    {
        computeRungs(simData);
        printTimestepStats(timestep_);
        timer.step("Timestep");

        auto  driftBack       = [](int subStep, int rung) { return subStep % (1 << rung); };
        auto& d               = simData.hydro;
        int   lowestDriftRung = cstone::butterfly(timestep_.substep + 1);
        bool  isLastSubstep   = activeRung(timestep_.substep + 1, timestep_.numRungs) == 0;
        auto  substepBox      = isLastSubstep ? domain.box() : cstone::Box<T>(0, 1, cstone::BoundaryType::open);

        for (int i = 0; i < timestep_.numRungs; ++i)
        {
            bool useRung = timestep_.substep == driftBack(timestep_.substep, i); // if drift back to start of hierarchy
            bool advance = i < lowestDriftRung;

            float          dt    = timestep_.nextDt;
            auto           dt_m1 = useRung ? prevTimestep_.dt_m1 : timestep_.dt_m1;
            const uint8_t* rung  = rawPtr(get<"rung">(d));

            if (advance)
            {
                if (timestep_.dt_drift[i] > 0) { driftPositions(rungs_[i], d, 0, timestep_.dt_drift[i], dt_m1, rung); }
                computePositions(rungs_[i], d, substepBox, timestep_.dt_drift[i] + dt, dt_m1, rung);
                timestep_.dt_m1[i]    = timestep_.dt_drift[i] + dt;
                timestep_.dt_drift[i] = 0;
                if constexpr (cstone::HaveGpu<Acc>{}) { storeRungGpu(rungs_[i], i, rawPtr(get<"rung">(d))); }
            }
            else
            {
                driftPositions(rungs_[i], d, timestep_.dt_drift[i] + dt, timestep_.dt_drift[i], dt_m1, rung);
                timestep_.dt_drift[i] += dt;
            }
        }

        updateSmoothingLength(activeRungs_, d);

        timestep_.substep++;
        timestep_.elapsedDt += timestep_.nextDt;

        d.ttot += timestep_.nextDt;
        d.minDt_m1 = d.minDt;
        d.minDt    = timestep_.nextDt;
        timer.step("UpdateQuantities");
    }

    void saveFields(IFileWriter* writer, size_t first, size_t last, DataType& simData,
                    const cstone::Box<T>& box) override
    {
        auto& d = simData.hydro;
        d.resize(d.accSize());
        auto fieldPointers = d.data();
        auto indicesDone   = d.outputFieldIndices;
        auto namesDone     = d.outputFieldNames;

        auto output = [&]()
        {
            for (int i = int(indicesDone.size()) - 1; i >= 0; --i)
            {
                int fidx = indicesDone[i];
                if (d.isAllocated(fidx))
                {
                    int column = std::find(d.outputFieldIndices.begin(), d.outputFieldIndices.end(), fidx) -
                                 d.outputFieldIndices.begin();
                    transferToHost(d, first, last, {d.fieldNames[fidx]});
                    std::visit([writer, c = column, key = namesDone[i]](auto field)
                               { writer->writeField(key, field->data(), c); }, fieldPointers[fidx]);
                    indicesDone.erase(indicesDone.begin() + i);
                    namesDone.erase(namesDone.begin() + i);
                }
            }
        };

        // first output pass: write everything allocated at the end of the step
        output();

        // second output pass: write temporary quantities produced by the EOS
        release(d, "ay", "az");
        acquire(d, "rho", "p");
        computeEOS(first, last, d);
        output();
        release(d, "rho", "p");

        // third output pass: recover temporary curlv and divv quantities
        acquire(d, "curlv");
        if (!indicesDone.empty()) { computeIadDivvCurlv(groups_.view(), d, box); }
        output();
        release(d, "curlv");
        acquire(d, "ay", "az");

        // restore destroyed accelerations
        computeMomentumEnergy<avClean>(groups_.view(), nullptr, d, box);

        if (!indicesDone.empty() && Base::rank_ == 0)
        {
            std::cout << "WARNING: the following fields are not in use and therefore not output: ";
            for (int fidx = 0; fidx < indicesDone.size() - 1; ++fidx)
            {
                std::cout << d.fieldNames[fidx] << ",";
            }
            std::cout << d.fieldNames[indicesDone.back()] << std::endl;
        }
        timer.step("FileOutput");
    }

private:
    void printTimestepStats(Timestep ts)
    {
        int highRung = activeRung(timestep_.substep, timestep_.numRungs);
        if (Base::rank_ == 0)
        {
            util::array<LocalIndex, 4> numRungs = {ts.rungRanges[1], ts.rungRanges[2] - ts.rungRanges[1],
                                                   ts.rungRanges[3] - ts.rungRanges[2],
                                                   ts.rungRanges[4] - ts.rungRanges[3]};

            LocalIndex numActiveGroups = 0;
            for (int i = 0; i < highRung; ++i)
            {
                numActiveGroups += rungs_[i].numGroups;
            }
            if (highRung == 0) { std::cout << "# New block-TS " << ts.numRungs << " rungs, "; }
            else
            {
                std::cout << "# Substep " << timestep_.substep << "/" << (1 << (timestep_.numRungs - 1)) << ", "
                          << numActiveGroups << " active groups, ";
            }

            // clang-format off
            std::cout << "R0: " << numRungs[0] << " (" << (100. * numRungs[0] / groups_.numGroups) << "%) "
                      << "R1: " << numRungs[1] << " (" << (100. * numRungs[1] / groups_.numGroups) << "%) "
                      << "R2: " << numRungs[2] << " (" << (100. * numRungs[2] / groups_.numGroups) << "%) "
                      << "R3: " << numRungs[3] << " (" << (100. * numRungs[3] / groups_.numGroups) << "%) "
                      << "All: " << groups_.numGroups << " (100%)" << std::endl;
            // clang-format on
        }
    }
};

} // namespace sphexa
