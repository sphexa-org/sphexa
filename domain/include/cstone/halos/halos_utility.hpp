/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich
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
 * @brief  Implementation of halo utility functions
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 * @author ChristopherBignamini <christopher.bignamini@gmail.com>
 */

#pragma once

#include "cstone/domain/layout.hpp"
#include "cstone/util/gsl-lite.hpp"

namespace cstone
{

namespace detail
{

//! @brief check that only owned particles in [particleStart_:particleEnd_] are sent out as halos
void checkIndices(const SendList& sendList,
                  [[maybe_unused]] LocalIndex start,
                  [[maybe_unused]] LocalIndex end,
                  [[maybe_unused]] LocalIndex bufferSize);

//! @brief check halo discovery for sanity
template<class KeyType>
int checkHalos(int myRank,
               gsl::span<const TreeIndexPair> focusAssignment,
               gsl::span<const int> haloFlags,
               gsl::span<const KeyType> ftree)
{
    TreeNodeIndex firstAssignedNode = focusAssignment[myRank].start();
    TreeNodeIndex lastAssignedNode  = focusAssignment[myRank].end();

    std::array<TreeNodeIndex, 2> checkRanges[2] = {{0, firstAssignedNode},
                                                   {lastAssignedNode, TreeNodeIndex(haloFlags.size())}};

    int ret = 0;
    for (int range = 0; range < 2; ++range)
    {
#pragma omp parallel for
        for (TreeNodeIndex i = checkRanges[range][0]; i < checkRanges[range][1]; ++i)
        {
            if (haloFlags[i])
            {
                bool peerFound = false;
                for (auto peerRange : focusAssignment)
                {
                    if (peerRange.start() <= i && i < peerRange.end()) { peerFound = true; }
                }
                if (!peerFound)
                {
                    std::cout << "Assignment rank " << myRank << " " << std::oct << ftree[firstAssignedNode] << " - "
                              << ftree[lastAssignedNode] << std::dec << std::endl;
                    std::cout << "Failed node " << i << " " << std::oct << ftree[i] << " - " << ftree[i + 1] << std::dec
                              << std::endl;
                    ret = 1;
                }
            }
        }
    }
    return ret;
}

}

}