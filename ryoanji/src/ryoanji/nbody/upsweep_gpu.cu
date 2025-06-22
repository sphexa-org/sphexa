/*
 * Ryoanji N-body solver
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Upsweep for multipole and source center computation
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "cstone/cuda/cuda_runtime.hpp"
#include "cstone/primitives/math.hpp"

#include "ryoanji/nbody/cartesian_qpole.hpp"
#include "ryoanji/nbody/kernel.hpp"

#include "upsweep_gpu.h"

namespace ryoanji
{

struct UpsweepConfig
{
    static constexpr int numThreads = 256;
};

template<class Tc, class Tm, class Tf, class MType>
__global__ void computeLeafMultipolesKernel(const Tc* x, const Tc* y, const Tc* z, const Tm* m,
                                            const TreeNodeIndex* leafToInternal, TreeNodeIndex numLeaves,
                                            const LocalIndex* layout, const Vec4<Tf>* centers, MType* multipoles)
{
    unsigned leafIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leafIdx < numLeaves)
    {
        TreeNodeIndex i = leafToInternal[leafIdx];
        P2M(x, y, z, m, layout[leafIdx], layout[leafIdx + 1], centers[i], multipoles[i]);
    }
}

template<class Tc, class Tm, class Tf, class MType>
void computeLeafMultipoles(const Tc* x, const Tc* y, const Tc* z, const Tm* m, const TreeNodeIndex* leafToInternal,
                           TreeNodeIndex numLeaves, const LocalIndex* layout, const Vec4<Tf>* centers,
                           MType* multipoles)
{
    constexpr int numThreads = UpsweepConfig::numThreads;
    if (numLeaves)
    {
        computeLeafMultipolesKernel<<<cstone::iceil(numLeaves, numThreads), numThreads>>>(
            x, y, z, m, leafToInternal, numLeaves, layout, centers, multipoles);
    }
}

#define COMPUTE_LEAF_MULTIPOLES(Tc, Tm, Tf, MType)                                                                     \
    template void computeLeafMultipoles(const Tc* x, const Tc* y, const Tc* z, const Tm* m,                            \
                                        const TreeNodeIndex* leafToInternal, TreeNodeIndex numLeaves,                  \
                                        const LocalIndex* layout, const Vec4<Tf>* centers, MType* multipoles)

template<class T, class MType>
__global__ void upsweepMultipolesKernel(TreeNodeIndex firstCell, TreeNodeIndex lastCell,
                                        const TreeNodeIndex* childOffsets, const Vec4<T>* centers, MType* multipoles)
{
    const int cellIdx = blockIdx.x * blockDim.x + threadIdx.x + firstCell;
    if (cellIdx >= lastCell) return;

    TreeNodeIndex firstChild = childOffsets[cellIdx];

    // firstChild is zero if the cell is a leaf
    if (firstChild) { M2M(firstChild, firstChild + 8, centers[cellIdx], centers, multipoles, multipoles[cellIdx]); }
}

template<class T, class MType>
void upsweepMultipoles(TreeNodeIndex firstCell, TreeNodeIndex lastCell, const TreeNodeIndex* childOffsets,
                       const Vec4<T>* centers, MType* multipoles)
{
    constexpr int numThreads = UpsweepConfig::numThreads;
    if (lastCell > firstCell)
    {
        upsweepMultipolesKernel<<<cstone::iceil(lastCell - firstCell, numThreads), numThreads>>>(
            firstCell, lastCell, childOffsets, centers, multipoles);
    }
}

#define UPSWEEP_MULTIPOLES(T, MType)                                                                                   \
    template void upsweepMultipoles(TreeNodeIndex firstCell, TreeNodeIndex lastCell,                                   \
                                    const TreeNodeIndex* childOffsets, const Vec4<T>* centers, MType* multipoles)

#define INSTANTIATE_MULTIPOLE(MType)                                                                                   \
    COMPUTE_LEAF_MULTIPOLES(double, double, double, MType<double>);                                                    \
    COMPUTE_LEAF_MULTIPOLES(double, float, double, MType<float>);                                                      \
    COMPUTE_LEAF_MULTIPOLES(float, float, float, MType<float>);                                                        \
    UPSWEEP_MULTIPOLES(double, MType<double>);                                                                         \
    UPSWEEP_MULTIPOLES(double, MType<float>);                                                                          \
    UPSWEEP_MULTIPOLES(float, MType<float>);

INSTANTIATE_MULTIPOLE(CartesianQuadrupole)
INSTANTIATE_MULTIPOLE(CartesianMDQpole)

} // namespace ryoanji
