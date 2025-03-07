/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
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
 * @brief Neighbor list compression tests
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#include "gtest/gtest.h"
#include "cstone/compressneighbors.hpp"

using namespace cstone;

static_assert(std::input_iterator<NeighborListDecompIterator>);

TEST(CompressNeighbors, nibbleBuffer)
{
    char buffer[100];

    NibbleBuffer nibbles(buffer, sizeof(buffer), true);

    EXPECT_EQ(nibbles.max_size(), 2 * (sizeof(buffer) - sizeof(unsigned)));

    for (unsigned i = 0; i < nibbles.max_size(); ++i)
        nibbles.push_back(i % 16);

    EXPECT_EQ(nibbles.size(), nibbles.max_size());

    for (unsigned i = 0; i < nibbles.max_size(); ++i)
        EXPECT_EQ(nibbles[i], i % 16);

    for (int i = nibbles.max_size() - 1; i >= 0; --i)
        EXPECT_EQ(nibbles.pop_back(), i % 16);

    EXPECT_EQ(nibbles.size(), 0);
}

TEST(CompressNeighbors, roundtrip)
{
    std::array<std::uint32_t, 17> nbs = {300, 301, 302, 100, 101, 200, 400, 402, 403,
                                         404, 405, 406, 407, 408, 409, 410, 411};

    char buffer[4 * nbs.size()];
    NeighborListCompressor comp(buffer, sizeof(buffer));

    for (auto nb : nbs)
        EXPECT_TRUE(comp.push_back(nb));

    EXPECT_EQ(comp.size(), nbs.size());
    EXPECT_LT(comp.nbytes(), 4 * nbs.size());

    NeighborListDecompressor decomp(buffer, sizeof(buffer));
    EXPECT_EQ(comp.nbytes(), decomp.nbytes());

    EXPECT_EQ(comp.size(), std::distance(decomp.begin(), decomp.end()));

    std::size_t i = 0;
    for (auto nb : decomp)
        EXPECT_EQ(nbs[i++], nb);
    EXPECT_EQ(i, nbs.size());
}

TEST(CompressNeighbors, smallBuffer)
{
    std::array<std::uint32_t, 4> nbs = {300, 301, 302, 100};

    char buffer[sizeof(unsigned) + 5];
    NeighborListCompressor comp(buffer, sizeof(buffer));

    EXPECT_TRUE(comp.push_back(nbs[0]));
    EXPECT_TRUE(comp.push_back(nbs[1]));
    EXPECT_TRUE(comp.push_back(nbs[2]));
    EXPECT_FALSE(comp.push_back(nbs[3]));

    EXPECT_EQ(comp.size(), 3);
    EXPECT_LE(comp.nbytes(), sizeof(buffer));

    NeighborListDecompressor decomp(buffer, sizeof(buffer));
    EXPECT_EQ(comp.nbytes(), decomp.nbytes());

    std::size_t i = 0;
    for (auto nb : decomp)
        EXPECT_EQ(nbs[i++], nb);
    EXPECT_EQ(i, comp.size());
}

TEST(CompressNeighbors, empty)
{
    char buffer[100];
    NeighborListCompressor comp(buffer, sizeof(buffer));

    EXPECT_EQ(comp.size(), 0);
    EXPECT_LE(comp.nbytes(), sizeof(unsigned));

    NeighborListDecompressor decomp(buffer, sizeof(buffer));
    EXPECT_EQ(decomp.nbytes(), sizeof(unsigned));
    EXPECT_EQ(decomp.begin(), decomp.end());
    EXPECT_EQ(std::distance(decomp.begin(), decomp.end()), 0);
}

TEST(CompressNeighborsSimple, roundtrip)
{
    std::array<std::uint32_t, 17> nbs = {300, 301, 302, 100, 101, 200, 400, 402, 403,
                                         404, 405, 406, 407, 408, 409, 410, 411};

    char buffer[4 * nbs.size()];
    SimpleNeighborListCompressor comp(buffer, sizeof(buffer));

    for (auto nb : nbs)
        EXPECT_TRUE(comp.push_back(nb));

    EXPECT_EQ(comp.size(), nbs.size());
    EXPECT_LT(comp.nbytes(), 4 * nbs.size());

    SimpleNeighborListDecompressor decomp(buffer, sizeof(buffer));
    EXPECT_EQ(comp.nbytes(), decomp.nbytes());

    EXPECT_EQ(comp.size(), std::distance(decomp.begin(), decomp.end()));

    std::size_t i = 0;
    for (auto nb : decomp)
        EXPECT_EQ(nbs[i++], nb);
    EXPECT_EQ(i, nbs.size());
}

TEST(CompressNeighborsSimple, smallBuffer)
{
    std::array<std::uint32_t, 4> nbs = {300, 301, 302, 100};

    char buffer[sizeof(std::uint32_t) * 3];
    SimpleNeighborListCompressor comp(buffer, sizeof(buffer));

    EXPECT_TRUE(comp.push_back(nbs[0]));
    EXPECT_TRUE(comp.push_back(nbs[1]));
    EXPECT_TRUE(comp.push_back(nbs[2]));
    EXPECT_FALSE(comp.push_back(nbs[3]));

    EXPECT_EQ(comp.size(), 3);
    EXPECT_LE(comp.nbytes(), sizeof(buffer));

    SimpleNeighborListDecompressor decomp(buffer, sizeof(buffer));
    EXPECT_EQ(comp.nbytes(), decomp.nbytes());

    std::size_t i = 0;
    for (auto nb : decomp)
        EXPECT_EQ(nbs[i++], nb);
    EXPECT_EQ(i, comp.size());
}

TEST(CompressNeighborsSimple, empty)
{
    char buffer[100];
    SimpleNeighborListCompressor comp(buffer, sizeof(buffer));

    EXPECT_EQ(comp.size(), 0);
    EXPECT_LE(comp.nbytes(), 2 * sizeof(std::uint32_t));

    SimpleNeighborListDecompressor decomp(buffer, sizeof(buffer));
    EXPECT_EQ(decomp.nbytes(), 2 * sizeof(std::uint32_t));
    EXPECT_EQ(decomp.begin(), decomp.end());
    EXPECT_EQ(std::distance(decomp.begin(), decomp.end()), 0);
}
