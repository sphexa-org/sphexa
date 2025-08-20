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
#include "box.hpp"

namespace cstone
{

//! @brief compute minimum and maximum of an array range with an OpenMP reduction
template<class T>
struct MinMax
{
    std::tuple<T, T> operator()(const T* start, const T* end)
    {
        assert(end >= start);

        T minimum = INFINITY;
        T maximum = -INFINITY;

#pragma omp parallel for reduction(min : minimum) reduction(max : maximum)
        for (size_t pi = 0; pi < std::size_t(end - start); pi++)
        {
            T value = start[pi];
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }

        return std::make_tuple(minimum, maximum);
    }
};

/*! @brief compute global bounding box for local x,y,z arrays
 *
 * @tparam     T            float or double
 * @param[in]  x            x coordinate array start
 * @param[in]  y            y coordinate array start
 * @param[in]  z            z coordinate array start
 * @param[in]  numElements  length of @a x,y,z arrays
 * @param[in]  previousBox  previous coordinate bounding box, default open-boundary box with limits ignored
 * @return                  the new bounding box
 *
 * For each periodic dimension, limits are fixed and will not be modified.
 * For non-periodic dimensions, limits are determined by global min/max.
 */
template<class T, class Op = MinMax<T>>
auto makeGlobalBox(const T* x, const T* y, const T* z, size_t numElements, const Box<T>& previousBox = Box<T>(0, 1))
{
    bool keepX = previousBox.boundaryX() == BoundaryType::periodic || previousBox.boundaryX() == BoundaryType::fixed;
    bool keepY = previousBox.boundaryY() == BoundaryType::periodic || previousBox.boundaryY() == BoundaryType::fixed;
    bool keepZ = previousBox.boundaryZ() == BoundaryType::periodic || previousBox.boundaryZ() == BoundaryType::fixed;

    std::array<T, 6> extrema{previousBox.xmin(), previousBox.xmax(), previousBox.ymin(),
                             previousBox.ymax(), previousBox.zmin(), previousBox.zmax()};
    if (numElements)
    {
        std::tie(extrema[0], extrema[1]) =
            keepX ? std::make_tuple(previousBox.xmin(), previousBox.xmax()) : Op{}(x, x + numElements);
        std::tie(extrema[2], extrema[3]) =
            keepY ? std::make_tuple(previousBox.ymin(), previousBox.ymax()) : Op{}(y, y + numElements);
        std::tie(extrema[4], extrema[5]) =
            keepZ ? std::make_tuple(previousBox.zmin(), previousBox.zmax()) : Op{}(z, z + numElements);
    }

    if (!keepX || !keepY || !keepZ)
    {
        extrema[1] = -extrema[1];
        extrema[3] = -extrema[3];
        extrema[5] = -extrema[5];
        MPI_Allreduce(MPI_IN_PLACE, extrema.data(), extrema.size(), MpiType<T>{}, MPI_MIN, MPI_COMM_WORLD);
        extrema[1] = -extrema[1];
        extrema[3] = -extrema[3];
        extrema[5] = -extrema[5];
    }

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
