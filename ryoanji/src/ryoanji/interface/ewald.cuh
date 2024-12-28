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

}
