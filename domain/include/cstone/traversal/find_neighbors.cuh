/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Neighbor search on GPU with breadth-first warp-aware octree traversal
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/sfc/box.hpp"
#include "cstone/tree/definitions.h"

namespace cstone
{

template<bool UsePbc, class T, std::enable_if_t<UsePbc, int> = 0>
__device__ __forceinline__ bool cellOverlap(const Vec3<T>& curSrcCenter,
                                            const Vec3<T>& curSrcSize,
                                            const Vec3<T>& targetCenter,
                                            const Vec3<T>& targetSize,
                                            const Box<T>& box)
{
    return norm2(minDistance(curSrcCenter, curSrcSize, targetCenter, targetSize, box)) == T(0.0);
}

template<bool UsePbc, class T, std::enable_if_t<!UsePbc, int> = 0>
__device__ __forceinline__ bool cellOverlap(const Vec3<T>& curSrcCenter,
                                            const Vec3<T>& curSrcSize,
                                            const Vec3<T>& targetCenter,
                                            const Vec3<T>& targetSize,
                                            const Box<T>& /*box*/)
{
    return norm2(minDistance(curSrcCenter, curSrcSize, targetCenter, targetSize)) == T(0.0);
}

} // namespace cstone
