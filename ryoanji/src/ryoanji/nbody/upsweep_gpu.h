/*
 * MIT License
 *
 * Copyright (c) 2024 CSCS, ETH Zurich, University of Basel, University of Zurich
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
