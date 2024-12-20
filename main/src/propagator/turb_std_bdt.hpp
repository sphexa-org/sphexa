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
 */

#pragma once

#include <filesystem>
#include <variant>

#include "cstone/util/constexpr_string.hpp"
#include "cstone/fields/field_get.hpp"
#include "io/arg_parser.hpp"
#include "sph/sph.hpp"
#include "sph/hydro_turb/turbulence_data.hpp"
#include "sph/ts_rungs.hpp"

#include "gravity_wrapper.hpp"

namespace sphexa
{

using namespace sph;
using util::FieldList;

//! @brief VE hydro propagator that adds turbulence stirring to the acceleration prior to position update
template<class DomainType, class DataType>
class TurbStdBdtProp final : public Propagator<DomainType, DataType>
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

    AccVector<LocalIndex>                                                                haloRecvScratch;
    sph::TurbulenceData<typename DataType::RealType, typename DataType::AcceleratorType> turbulenceData;
    using ConservedFields = FieldList<"c", "vx", "vy", "vz", "x_m1", "y_m1", "z_m1", "du_m1", "rung", "id">;

    //! @brief list of dependent fields, these may be used as scratch space during domain sync
    using DependentFields =
        FieldList<"rho", "p", "c", "ax", "ay", "az", "du", "c11", "c12", "c13", "c22", "c23", "c33", "nc">;

    //! @brief Return rung of current block time-step
    static int activeRung(int substep, int numRungs)
    {
        if (substep == 0 || substep >= (1 << (numRungs - 1))) { return 0; }
        else { return cstone::butterfly(substep); }
    }

public:
    TurbStdBdtProp(std::ostream& output, size_t rank, const InitSettings& settings)
        : Base(output, rank)
        , turbulenceData(settings, rank == 0)
    {
        if (not cstone::HaveGpu<Acc>{}) { throw std::runtime_error("This propagator is not supported on CPUs\n"); }
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

    void save(IFileWriter* writer) override
    {
        timestep_.loadOrStore(writer, "ts::");
        turbulenceData.loadOrStore(writer);
    }

    void load(const std::string& initCond, IFileReader* reader) override
    {
        int         step = numberAfterSign(initCond, ":");
        std::string path = removeModifiers(initCond);
        // The file does not exist, we're starting from scratch. Nothing to do.
        if (!std::filesystem::exists(path)) { return; }

        reader->setStep(path, step, FileMode::independent);
        turbulenceData.loadOrStore(reader);
        timestep_.loadOrStore(reader, "ts::");

        if (Base::rank_ == 0) { std::cout << "Restored turbulence state from " << path << ":" << step << std::endl; }
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

        computeDensity(activeRungs_, d, domain.box());
        timer.step("Density");
        computeIsothermalEosStd(first, last, d);
        timer.step("EquationOfState");

        domain.exchangeHalos(get<"vx", "vy", "vz", "rho", "p", "c">(d), get<"keys">(d), haloRecvScratch);
        timer.step("mpi::synchronizeHalos");

        computeIAD(activeRungs_, d, domain.box());
        timer.step("IAD");

        domain.exchangeHalos(get<"c11", "c12", "c13", "c22", "c23", "c33">(d), get<"keys">(d), haloRecvScratch);
        timer.step("mpi::synchronizeHalos");

        computeMomentumEnergySTD(activeRungs_, rawPtr(groupDt_), d, domain.box());
        timer.step("MomentumEnergyIAD");

        if (d.g != 0.0)
        {
            bool      isNewHierarchy = activeRung(timestep_.substep, timestep_.numRungs) == 0;
            GroupView gravGroup = isNewHierarchy ? mHolder_.computeSpatialGroups(simData.hydro, domain) : activeRungs_;

            mHolder_.upsweep(d, domain);
            timer.step("Upsweep");
            pmReader.step();
            mHolder_.traverse(gravGroup, d, domain);
            timer.step("Gravity");
            pmReader.step();
        }
        groupAccTimestep(activeRungs_, rawPtr(groupDt_), d);
        driveTurbulence(activeRungs_, simData.hydro, turbulenceData);
        timer.step("Turbulence Stirring");
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
        Base::outputAllocatedFields(writer, first, last, simData);
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
