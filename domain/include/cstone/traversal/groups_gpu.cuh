/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Device-side particle target group splitting helpers.
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * The split criterion defined here (evalGroupSplitMask/findSplits) is used fused into
 * ryoanji's traverseKernel and in the unit tests of the standalone split pipeline.
 */

#include "cstone/cuda/cuda_runtime.hpp"
#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/cuda/gpu_config.cuh"
#include "cstone/primitives/warpscan.cuh"
#include "cstone/sfc/box.hpp"
#include "cstone/sfc/sfc.hpp"
#include "cstone/tree/definitions.h"
#include "cstone/util/array.hpp"

namespace cstone
{

/*! @brief Computes splitting patterns of target particle groups
 *
 * @tparam    T            float or double
 * @tparam    N            number of particles per thread
 * @param[in] pos          particle input
 * @param[in] distCritSq   maximum allowed distance^2 between two consecutive particles
 * @return                 a thread mask indicating the lanes between splits with 1-bits
 *                         all lanes return the same result
 */
template<class T, size_t N>
__device__ util::array<GpuConfig::ThreadMask, N> findSplits(util::array<Vec4<T>, N> pos, T distCritSq)
{
    unsigned laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);

    util::array<Vec3<T>, N> Xlane;
    for (std::size_t k = 0; k < N; ++k)
    {
        Xlane[k] = {pos[k][0], pos[k][1], pos[k][2]};
    }

    util::array<Vec3<T>, N> Xnext = Xlane;
    for (std::size_t k = 0; k < N; ++k)
    {
        for (int j = 0; j < 3; ++j)
        {
            Xnext[k][j] = shflDownSync(Xlane[k][j], 1);
        }
        if (k + 1 < N)
        {
            Vec3<T> spill;
            for (int j = 0; j < 3; ++j)
            {
                spill[j] = shflSync(Xlane[k + 1][j], 0);
            }
            if (laneIdx == GpuConfig::warpSize - 1) { Xnext[k] = spill; }
        }
    }

    // The last difference in the warp (last lane of last segment N-1) will always be zero,
    // therefore the MSB of the return value will always be zero.
    util::array<GpuConfig::ThreadMask, N> splits;
    for (std::size_t k = 0; k < N; ++k)
    {
        T distSq   = norm2(Xnext[k] - Xlane[k]);
        bool split = distSq > stl::min(distCritSq, pos[k][3] * pos[k][3]);
        splits[k]  = ballotSync(split);
    }

    return splits;
}

/*! @brief evaluate the spatial compactness criterion for one fixed-size target group
 *
 * Determines the split boundaries of the fixed-size group starting at particle index @p bodyBegin:
 * two consecutive particles of the group are split when they are farther apart than the smaller of
 * (tolFactor times the edge length of the smallest leaf cell occupied by any particle of the group)
 * or twice their smoothing length. Particle loads beyond @p last - 1 are clamped, such that the
 * trailing lanes of the last fixed group of a particle range re-load the last particle.
 *
 * This is the single source of truth for the split criterion, shared between the standalone
 * split kernels and fused traversals that compute the split in-register.
 *
 * @tparam groupSize   number of particles per fixed group, a multiple of GpuConfig::warpSize
 * @tparam Tc          float or double, type of the particle coordinates
 * @tparam Th          float or double, type of the particle smoothing lengths
 * @tparam KeyType     32- or 64-bit unsigned integer
 * @param[in] bodyBegin  particle index of the first particle of the fixed group
 * @param[in] last       index one past the last particle of the range
 * @param[in] x          local x particle coordinates
 * @param[in] y          local y particle coordinates
 * @param[in] z          local z particle coordinates
 * @param[in] h          local smoothing lengths, may be nullptr
 * @param[in] leaves     cornerstone leaf-cell array, size is @p numLeaves + 1
 * @param[in] numLeaves  number of leaves in @p leaves
 * @param[in] layout     layout[i] is the particle index of the first particle in leaf i
 * @param[in] box        global coordinate bounding box
 * @param[in] tolFactor  maximum distance between consecutive particles is
 *                       tolFactor * cbrt(smallest leaf-cell volume in group)
 * @return               a warp-uniform thread mask with a 1-bit at the position of the last
 *                       particle of each subgroup
 */
template<unsigned groupSize, class Tc, class Th, class KeyType>
__device__ util::array<GpuConfig::ThreadMask, groupSize / GpuConfig::warpSize> evalGroupSplitMask(
    LocalIndex bodyBegin, LocalIndex last, const Tc* x, const Tc* y, const Tc* z, const Th* h, const KeyType* leaves,
    TreeNodeIndex numLeaves, const LocalIndex* layout, const Box<Tc> box, float tolFactor)
{
    constexpr LocalIndex nwt = groupSize / GpuConfig::warpSize;

    LocalIndex laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);

    LocalIndex bodyIdx[nwt];
    for (LocalIndex k = 0; k < nwt; ++k)
    {
        bodyIdx[k] = imin(bodyBegin + k * GpuConfig::warpSize + laneIdx, last - 1);
    }

    TreeNodeIndex leafIdx[nwt];
    for (LocalIndex k = 0; k < nwt; ++k)
    {
        leafIdx[k] = stl::upper_bound(layout, layout + numLeaves, bodyIdx[k]) - layout - 1;
    }

    const auto axesBits = box.getBoxDimBits(maxTreeLevel<KeyType>{});
    Box<Th> unitBox(0, 1 / (1 << (maxTreeLevel<KeyType>{} - axesBits[0])), 0,
                    1 / (1 << (maxTreeLevel<KeyType>{} - axesBits[1])), 0,
                    1 / (1 << (maxTreeLevel<KeyType>{} - axesBits[2])));
    Th nodeVolume = 1;
    for (LocalIndex k = 0; k < nwt; ++k)
    {
        auto nodeIBox = sfcIBox(sfcKey<KeyType>(leaves[leafIdx[k]]), sfcKey<KeyType>(leaves[leafIdx[k] + 1]), axesBits);
        auto [nodeCenter, nodeSize] = centerAndSize<KeyType>(nodeIBox, unitBox);
        Th vol                       = 8 * nodeSize[0] * nodeSize[1] * nodeSize[2];
        nodeVolume                  = vol > 0 ? min(vol, nodeVolume) : nodeVolume;
    }
    nodeVolume  = warpMin(nodeVolume);
    Tc distCrit = std::cbrt(nodeVolume) * tolFactor;

    // load target coordinates scaled to the unit cube
    util::array<Vec4<Tc>, nwt> pos_i;
    for (LocalIndex k = 0; k < nwt; k++)
    {
        pos_i[k] = {x[bodyIdx[k]] * box.ilx(), y[bodyIdx[k]] * box.ily(), z[bodyIdx[k]] * box.ilz(),
                    h ? Tc(2) * h[bodyIdx[k]] / box.minExtent() : Tc(1)};
    }

    return findSplits(pos_i, distCrit * distCrit);
}

} // namespace cstone
