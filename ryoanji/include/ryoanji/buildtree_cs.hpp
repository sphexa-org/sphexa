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
 * @brief  Build a tree for Ryoanji with the cornerstone framework
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <vector>

#include "cstone/tree/octree_internal_td.hpp"

#include "ryoanji/types.h"


template<class T>
auto buildFromCstone(std::vector<util::array<T, 4>>& bodies, const Box& box)
{
    int numParticles = bodies.size();
    unsigned bucketSize = 64;

    static_assert(std::is_same_v<float, T>);

    std::vector<T> x(numParticles);
    std::vector<T> y(numParticles);
    std::vector<T> z(numParticles);
    std::vector<T> m(numParticles);
    std::vector<uint64_t> keys(numParticles);

    for (int i = 0; i < numParticles; ++i)
    {
        x[i] = bodies[i][0];
        y[i] = bodies[i][1];
        z[i] = bodies[i][2];
        m[i] = bodies[i][3];
    }

    cstone::Box<T> csBox(box.X[0] - box.R, box.X[0] + box.R,
                         box.X[1] - box.R, box.X[1] + box.R,
                         box.X[2] - box.R, box.X[2] + box.R,
                         false, false, false);

    cstone::computeSfcKeys(x.data(), y.data(), z.data(), cstone::sfcKindPointer(keys.data()), numParticles, csBox);

    std::vector<int> ordering(numParticles);
    std::iota(ordering.begin(), ordering.end(), 0);
    cstone::sort_by_key(keys.begin(), keys.end(), ordering.begin());

    cstone::reorderInPlace(ordering, bodies.data());

    auto [tree, counts] = cstone::computeOctree(keys.data(), keys.data() + numParticles, bucketSize);

    cstone::TdOctree<uint64_t> octree;
    octree.update(tree.data(), cstone::nNodes(tree));

    std::vector<CellData> ryoanjiTree(octree.numTreeNodes());

    for (int i = 0; i < octree.numTreeNodes(); ++i)
    {
        int firstParticle = std::lower_bound(keys.begin(), keys.end(), octree.codeStart(i)) - keys.begin();
        int lastParticle  = std::upper_bound(keys.begin(), keys.end(), octree.codeEnd(i)) - keys.begin();
        int child = 0;
        int numChildren = 1;
        if (!octree.isLeaf(i))
        {
            child = octree.child(i, 0);
            numChildren = 8;
        }
        CellData cell(
            octree.level(i), octree.parent(i), firstParticle, lastParticle - firstParticle, child, numChildren);
        ryoanjiTree[i] = cell;
    }

    std::vector<int2> levelRange(cstone::maxTreeLevel<uint64_t>{} + 1);
    for (int level = 0; level <= cstone::maxTreeLevel<uint64_t>{}; ++level)
    {
        levelRange[level].x = octree.levelOffset(level);
        levelRange[level].y = octree.levelOffset(level + 1);
    }

    int numLevels = octree.level(octree.numTreeNodes() - 1);

    return std::make_tuple(numLevels, std::move(ryoanjiTree), std::move(levelRange));
}