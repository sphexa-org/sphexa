/*
 * MIT License
 *
 * Copyright (c) 2026 CSCS, ETH Zurich
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
 * @brief Automatic load-balancing criterion from Boulmier et al. (JPDC 2022)
 *
 * Triggers a full domain sync when tau * u(tau) - integral u >= C, where u is the
 * imbalance time metric (t_max - t_mean) and C is the measured cost of the last sync.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mpi.h>

namespace sphexa
{

//! @brief Conservative upper bound on max particle speed over [first, last)
template<class T>
float localMaxParticleSpeed(size_t first, size_t last, const T* vx, const T* vy, const T* vz)
{
    float maxSpeed = 0;
    if (first >= last) { return maxSpeed; }

#pragma omp parallel for reduction(max : maxSpeed)
    for (size_t i = first; i < last; ++i)
    {
        float speed = std::abs(vx[i]) + std::abs(vy[i]) + std::abs(vz[i]);
        maxSpeed    = std::max(maxSpeed, speed);
    }
    return maxSpeed;
}

//! @brief Sum of smoothing lengths over locally owned particles
template<class T>
double localSumSmoothingLength(size_t first, size_t last, const T* h)
{
    if (first >= last) { return 0; }

    double sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (size_t i = first; i < last; ++i)
    {
        sum += h[i];
    }
    return sum;
}

class BoulmierLoadBalanceCriterion
{
public:
    //! @param haloSafetyFactor trigger sync when particle travel exceeds this fraction of the halo reach
    explicit BoulmierLoadBalanceCriterion(float haloSafetyFactor = 0.5f)
        : haloSafetyFactor_(haloSafetyFactor)
    {
    }

    [[nodiscard]] bool enabled() const { return enabled_; }
    void               setEnabled(bool flag) { enabled_ = flag; }

    [[nodiscard]] bool isFullySynced() const { return fullySynced_; }
    void               setFullySynced(bool flag) { fullySynced_ = flag; }

    [[nodiscard]] size_t stepsSinceSync() const { return stepsSinceSync_; }

    //! @brief Decide whether a full MPI domain sync is required at the start of an iteration
    [[nodiscard]] bool needsFullSync(size_t iteration, float haloSearchExt) const
    {
        if (!enabled_) { return true; }
        if (iteration == 0 || loadBalanceCost_ <= 0) { return true; }

        if (meanSmoothingLength_ > 0 && maxSpeedSinceSync_ > 0)
        {
            const float haloReach = haloSearchExt * 2.0f * meanSmoothingLength_;
            const float travel    = dtSinceLastSync_ * maxSpeedSinceSync_;
            if (travel >= haloSafetyFactor_ * haloReach) { return true; }
        }

        if (stepsSinceSync_ == 0) { return false; }
        return stepsSinceSync_ * lastImbalance_ - cumulativeImbalance_ >= loadBalanceCost_;
    }

    void resetAfterFullSync(float localSyncTime, MPI_Comm comm)
    {
        float syncCost = localSyncTime;
        MPI_Allreduce(&localSyncTime, &syncCost, 1, MPI_FLOAT, MPI_MAX, comm);

        loadBalanceCost_       = syncCost;
        stepsSinceSync_        = 0;
        cumulativeImbalance_   = 0;
        lastImbalance_         = 0;
        dtSinceLastSync_       = 0;
        maxSpeedSinceSync_     = 0;
        meanSmoothingLength_   = 0;
        fullySynced_           = true;
    }

    void recordComputeImbalance(float localComputeTime, MPI_Comm comm)
    {
        if (!enabled_) { return; }

        float tMax = localComputeTime;
        float tSum = localComputeTime;
        MPI_Allreduce(&localComputeTime, &tMax, 1, MPI_FLOAT, MPI_MAX, comm);
        MPI_Allreduce(&localComputeTime, &tSum, 1, MPI_FLOAT, MPI_SUM, comm);

        int numRanks = 1;
        MPI_Comm_size(comm, &numRanks);
        lastImbalance_       = tMax - tSum / float(numRanks);
        cumulativeImbalance_ += lastImbalance_;
        stepsSinceSync_++;
    }

    void recordMotion(float dt, float localMaxSpeed, double localSumH, size_t localCount, MPI_Comm comm)
    {
        if (!enabled_) { return; }

        float globalMaxSpeed = localMaxSpeed;
        MPI_Allreduce(&localMaxSpeed, &globalMaxSpeed, 1, MPI_FLOAT, MPI_MAX, comm);

        uint64_t globalCount = localCount;
        double   globalSumH  = localSumH;
        MPI_Allreduce(MPI_IN_PLACE, &globalCount, 1, MPI_UINT64_T, MPI_SUM, comm);
        MPI_Allreduce(MPI_IN_PLACE, &globalSumH, 1, MPI_DOUBLE, MPI_SUM, comm);
        if (globalCount > 0) { meanSmoothingLength_ = float(globalSumH / double(globalCount)); }

        maxSpeedSinceSync_ = std::max(maxSpeedSinceSync_, globalMaxSpeed);
        dtSinceLastSync_ += dt;
    }

    [[nodiscard]] float loadBalanceCost() const { return loadBalanceCost_; }
    [[nodiscard]] float lastImbalance() const { return lastImbalance_; }
    [[nodiscard]] float cumulativeImbalance() const { return cumulativeImbalance_; }

private:
    bool  enabled_{false};
    bool  fullySynced_{true};
    float haloSafetyFactor_{0.5f};

    size_t stepsSinceSync_{0};
    float  cumulativeImbalance_{0};
    float  lastImbalance_{0};
    float  loadBalanceCost_{0};

    float dtSinceLastSync_{0};
    float maxSpeedSinceSync_{0};
    float meanSmoothingLength_{0};
};

} // namespace sphexa
