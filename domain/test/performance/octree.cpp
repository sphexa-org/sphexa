/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Octree performance test
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <chrono>
#include <iostream>
#include <numeric>

#include "cstone/traversal/collisions.hpp"
#include "cstone/tree/csarray.hpp"
#include "cstone/tree/octree.hpp"

#include "coord_samples/random.hpp"
#include "coord_samples/plummer.hpp"

using namespace cstone;

template<class KeyType>
std::tuple<std::vector<KeyType>, std::vector<unsigned>> build_tree(std::span<const KeyType> keys, unsigned bucketSize)
{
    std::vector<KeyType> tree;
    std::vector<unsigned> counts;

    auto tp0               = std::chrono::high_resolution_clock::now();
    std::tie(tree, counts) = computeOctree<KeyType>(keys, bucketSize);
    auto tp1               = std::chrono::high_resolution_clock::now();

    double t0 = std::chrono::duration<double>(tp1 - tp0).count();
    std::cout << "build time from scratch " << t0 << " nNodes(tree): " << nNodes(tree)
              << " count: " << std::accumulate(begin(counts), end(counts), 0lu) << std::endl;

    tp0 = std::chrono::high_resolution_clock::now();
    updateOctree<KeyType>(keys, bucketSize, tree, counts, std::numeric_limits<unsigned>::max());
    tp1 = std::chrono::high_resolution_clock::now();

    double t1 = std::chrono::duration<double>(tp1 - tp0).count();

    int nEmptyNodes = std::count(begin(counts), end(counts), 0);
    std::cout << "build time with guess " << t1 << " nNodes(tree): " << nNodes(tree)
              << " count: " << std::accumulate(begin(counts), end(counts), 0lu) << " empty nodes: " << nEmptyNodes
              << std::endl;

    return std::make_tuple(std::move(tree), std::move(counts));
}

template<class KeyType>
void internal_tree(const std::vector<KeyType>& tree)
{
    OctreeData<KeyType, CpuTag> octree;
    octree.resize(nNodes(tree));
    auto u0 = std::chrono::high_resolution_clock::now();
    updateInternalTree<KeyType>(tree, octree.data());
    auto u1 = std::chrono::high_resolution_clock::now();
    std::cout << "first octree update: " << std::chrono::duration<double>(u1 - u0).count() << std::endl;

    auto u2 = std::chrono::high_resolution_clock::now();
    updateInternalTree<KeyType>(tree, octree.data());
    auto u3 = std::chrono::high_resolution_clock::now();
    std::cout << "second octree update: " << std::chrono::duration<double>(u3 - u2).count() << std::endl;
}

int main()
{
    using KeyType = uint64_t;
    Box<double> box{-1, 1};

    int numParticles = 20000000;
    int bucketSize   = 16;

    RandomGaussianCoordinates<double, HilbertKey<KeyType>> randomBox(numParticles, box);

    // tree build from random gaussian coordinates
    auto [tree, counts] = build_tree<KeyType>(randomBox.particleKeys(), bucketSize);
    internal_tree(tree);

    auto px = plummer<double>(numParticles);
    std::vector<KeyType> pxCodes(numParticles);
    Box<double> pBox(*std::min_element(begin(px[0]), end(px[0])), *std::max_element(begin(px[0]), end(px[0])),
                     *std::min_element(begin(px[1]), end(px[1])), *std::max_element(begin(px[1]), end(px[1])),
                     *std::min_element(begin(px[2]), end(px[2])), *std::max_element(begin(px[2]), end(px[2])));

    std::cout << "plummer box: " << pBox.xmin() << " " << pBox.xmax() << " " << pBox.ymin() << " " << pBox.ymax() << " "
              << pBox.zmin() << " " << pBox.zmax() << std::endl;

    computeSfcKeys(px[0].data(), px[1].data(), px[2].data(), sfcKindPointer(pxCodes.data()), numParticles, pBox);
    std::sort(begin(pxCodes), end(pxCodes));

    std::tie(tree, counts) = build_tree<KeyType>(pxCodes, bucketSize);
}
