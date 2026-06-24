/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief compute global minima and maxima of array ranges
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 * @author Noah Kubli <noah.kubli@uzh.ch>
 *
 */

#pragma once

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cmath>

#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/primitives/primitives_gpu.h"
#include "cstone/execution.hpp"
#include "box.hpp"

namespace cstone
{

/*! @brief compute global bounding box for local x,y,z arrays
 *
 * @tparam     T            float or double
 * @tparam     Execution    execution policy (Cpu or Gpu)
 * @param[in]  x            x coordinate array start
 * @param[in]  y            y coordinate array start
 * @param[in]  z            z coordinate array start
 * @param[in]  numElements  length of @a x,y,z arrays
 * @param[in]  comm         MPI communicator
 * @param[in]  exec         execution policy
 * @param[in]  previousBox  previous coordinate bounding box, default open-boundary box with limits ignored
 * @return                  the new bounding box
 *
 * For each periodic dimension, limits are fixed and will not be modified.
 * For non-periodic dimensions, limits are determined by global min/max.
 */
template<execution::Policy Exec, class T>
auto makeGlobalBox(Exec exec,
                   const T* x,
                   const T* y,
                   const T* z,
                   size_t numElements,
                   MPI_Comm comm,
                   const Box<T>& previousBox = Box<T>(0, 1))
{
    bool keepX = previousBox.boundaryX() == BoundaryType::periodic || previousBox.boundaryX() == BoundaryType::fixed;
    bool keepY = previousBox.boundaryY() == BoundaryType::periodic || previousBox.boundaryY() == BoundaryType::fixed;
    bool keepZ = previousBox.boundaryZ() == BoundaryType::periodic || previousBox.boundaryZ() == BoundaryType::fixed;

    std::array<T, 6> extrema{previousBox.xmin(), previousBox.xmax(), previousBox.ymin(),
                             previousBox.ymax(), previousBox.zmin(), previousBox.zmax()};
    if (numElements)
    {
        std::tie(extrema[0], extrema[1]) =
            keepX ? std::make_tuple(previousBox.xmin(), previousBox.xmax()) : minMax(exec, x, x + numElements);
        std::tie(extrema[2], extrema[3]) =
            keepY ? std::make_tuple(previousBox.ymin(), previousBox.ymax()) : minMax(exec, y, y + numElements);
        std::tie(extrema[4], extrema[5]) =
            keepZ ? std::make_tuple(previousBox.zmin(), previousBox.zmax()) : minMax(exec, z, z + numElements);
    }

    if (!keepX || !keepY || !keepZ)
    {
        extrema[1] = -extrema[1];
        extrema[3] = -extrema[3];
        extrema[5] = -extrema[5];
        MPI_Allreduce(MPI_IN_PLACE, extrema.data(), extrema.size(), MpiType<T>{}, MPI_MIN, comm);
        extrema[1] = -extrema[1];
        extrema[3] = -extrema[3];
        extrema[5] = -extrema[5];
    }

    const T max_side_length = std::max({extrema[1] - extrema[0], extrema[3] - extrema[2], extrema[5] - extrema[4]});

    if (previousBox.boundaryX() == BoundaryType::cubic_open)
        extrema[1] = std::max(extrema[1], extrema[0] + max_side_length);
    if (previousBox.boundaryY() == BoundaryType::cubic_open)
        extrema[3] = std::max(extrema[3], extrema[2] + max_side_length);
    if (previousBox.boundaryZ() == BoundaryType::cubic_open)
        extrema[5] = std::max(extrema[5], extrema[4] + max_side_length);

    return Box<T>{extrema[0],
                  extrema[1],
                  extrema[2],
                  extrema[3],
                  extrema[4],
                  extrema[5],
                  previousBox.boundaryX(),
                  previousBox.boundaryY(),
                  previousBox.boundaryZ()};
}

} // namespace cstone
