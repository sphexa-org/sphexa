/*
 * Ryoanji N-body solver
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Header for Ewald summation
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/traversal/groups.hpp"
#include "ryoanji/nbody/ewald.h"

namespace ryoanji
{

template<class MType, class Tc, class Ta, class Tm, class Tu>
extern void computeGravityEwaldGpu(const cstone::Vec3<Tc>& rootCenter, const MType& Mroot, cstone::GroupView grp,
                                   const Tc* x, const Tc* y, const Tc* z, const Tm* m, const cstone::Box<Tc>& box,
                                   float G, Ta* ugrav, Ta* ax, Ta* ay, Ta* az, Tu* ugravTot, EwaldSettings settings);

/*! @brief Ewald gravity on GPU for the contiguous target range [firstTarget:lastTarget)
 *
 * Group-free counterpart of the group-based overload of computeGravityEwaldGpu: the target
 * particles are specified as an index range instead of a precomputed GroupView, such that no
 * call to computeSpatialGroups is required. The group-based overload must stay available for
 * rung-sliced active sets (block-step propagator), which are not contiguous ranges.
 */
template<class MType, class Tc, class Ta, class Tm, class Tu>
extern void computeGravityEwaldGpu(const cstone::Vec3<Tc>& rootCenter, const MType& Mroot, LocalIndex firstTarget,
                                   LocalIndex lastTarget, const Tc* x, const Tc* y, const Tc* z, const Tm* m,
                                   const cstone::Box<Tc>& box, float G, Ta* ugrav, Ta* ax, Ta* ay, Ta* az,
                                   Tu* ugravTot, EwaldSettings settings);

}
