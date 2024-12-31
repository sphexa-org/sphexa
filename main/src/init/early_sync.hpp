/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich
 *               2022 University of Basel
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
 * @brief Early domain synchronization for x,y,z coordinates only
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/domain/assignment.hpp"
#include "cstone/domain/layout.hpp"

namespace sphexa
{

/*! @brief Re-distribute globally distributed coordinates into compact SFC domains
 *
 * @tparam       KeyType             unsigned 32- or 64-bit integer
 * @tparam       T                   float or double
 * @tparam       Vector              STL-like vector type
 * @param[in]    rank                the executing rank ID
 * @param[in]    numRanks            total number of ranks
 * @param[in]    numParticlesGlobal  sum of x,y,z sizes across all ranks
 * @param[inout] x                   x coordinates
 * @param[inout] y                   y coordinates
 * @param[inout] z                   z coordinates
 * @param[in]    globalBox           global coordinate bounding box
 *
 * This is useful to distribute only the x,y,z coordinates along the space-filling-curve in cases
 * where the initial distribution deviates by a lot from the Cornerstone SFC distribution.
 * This saves space and increases performance, because x,y,z coordinates are sent to the right places
 * @p before the other particle fields are created.
 */
template<class KeyType, class T, class Vector>
void syncCoords(size_t rank, size_t numRanks, size_t numParticlesGlobal, Vector& x, Vector& y, Vector& z,
                const cstone::Box<T>& globalBox)
{
    size_t                    bucketSize = std::max(64lu, numParticlesGlobal / (100 * numRanks));
    cstone::BufferDescription o1{0, cstone::LocalIndex(x.size()), cstone::LocalIndex(x.size())};

    cstone::GlobalAssignment<KeyType, T> distributor(rank, numRanks, bucketSize, globalBox);

    std::vector<unsigned>                    orderScratch;
    cstone::SfcSorter<std::vector<unsigned>> sorter(orderScratch);

    std::vector<T>       scratch1, scratch2;
    std::vector<KeyType> particleKeys(x.size());
    cstone::LocalIndex   newNParticlesAssigned =
        distributor.assign(o1, sorter, scratch1, scratch2, particleKeys.data(), x.data(), y.data(), z.data());
    auto exchangeSize = std::max(cstone::LocalIndex(x.size()), newNParticlesAssigned);
    reallocate(exchangeSize, 1.01, particleKeys, x, y, z);
    auto [exchangeStart, keyView] = distributor.distribute({o1.start, o1.end, exchangeSize}, sorter, scratch1, scratch2,
                                                           particleKeys.data(), x.data(), y.data(), z.data());

    scratch1.resize(x.size());
    cstone::gatherArrays({sorter.getMap() + distributor.o3start(o1), distributor.numAssigned()}, 0, std::tie(x, y, z),
                         std::tie(scratch1));
    x.resize(keyView.size());
    y.resize(keyView.size());
    z.resize(keyView.size());
    x.shrink_to_fit();
    y.shrink_to_fit();
    z.shrink_to_fit();
}

} // namespace sphexa
