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

#pragma once

#include "ryoanji/nbody/types.h"
#include "kernel.hpp"

namespace ryoanji
{

template<class Tc, class Tm, class Tf, class MType>
extern void computeLeafMultipoles(const Tc* x, const Tc* y, const Tc* z, const Tm* m,
                                  const TreeNodeIndex* leafToInternal, TreeNodeIndex numLeaves,
                                  const LocalIndex* layout, const Vec4<Tf>* centers, MType* multipoles);

/*! @brief perform multipole upward sweep for one tree level
 *
 * launch config: one thread per cell of the current level
 *
 * @param[in]  firstCell        first cell to process
 * @param[in]  lastCell         last cell to process
 * @param[in]  childOffsets     cell index of first child of each node
 * @param[in]  centers          source expansion (mass) centers
 * @param[out] multipoles       output multipole of each cell
 */
template<class T, class MType>
extern void upsweepMultipoles(TreeNodeIndex firstCell, TreeNodeIndex lastCell, const TreeNodeIndex* childOffsets,
                              const Vec4<T>* centers, MType* multipoles);

} // namespace ryoanji
