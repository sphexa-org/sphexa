/*
 * MIT License
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *               2024 University of Basel
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
 * @brief Translation unit for halo utility functions
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 * @author ChristopherBignamini <christopher.bignamini@gmail.com>
 */

#include "halos_utility.hpp"

namespace cstone
{

namespace detail
{

//! @brief check that only owned particles in [particleStart_:particleEnd_] are sent out as halos
void checkIndices(const SendList& sendList,
                  [[maybe_unused]] LocalIndex start,
                  [[maybe_unused]] LocalIndex end,
                  [[maybe_unused]] LocalIndex bufferSize)
{
    for (const auto& manifest : sendList)
    {
        for (size_t ri = 0; ri < manifest.nRanges(); ++ri)
        {
            assert(!overlapTwoRanges(LocalIndex{0}, start, manifest.rangeStart(ri), manifest.rangeEnd(ri)));
            assert(!overlapTwoRanges(end, bufferSize, manifest.rangeStart(ri), manifest.rangeEnd(ri)));
        }
    }
}

}

}