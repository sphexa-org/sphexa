/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Random coordinates generation for testing
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <algorithm>
#include <random>
#include <vector>

#include "cstone/findneighbors.hpp"
#include "cstone/domain/layout.hpp"
#include "cstone/primitives/gather.hpp"
#include "cstone/sfc/sfc.hpp"
#include "cstone/tree/definitions.h"

namespace cstone
{

template<class Integer>
std::vector<Integer> makeRandomUniformKeys(size_t numKeys, int seed = 42)
{
    Integer maxCoord = nodeRange<Integer>(0) - 1;
    std::mt19937 gen(seed);
    std::uniform_int_distribution<Integer> distribution(0, maxCoord);

    auto randInt = [&distribution, &gen]() { return distribution(gen); };
    std::vector<Integer> ret(numKeys);
    std::generate(ret.begin(), ret.end(), randInt);
    std::sort(ret.begin(), ret.end());

    return ret;
}

template<class Integer>
std::vector<Integer> makeRandomGaussianKeys(size_t numKeys, int seed = 42)
{
    Integer maxCoord = nodeRange<Integer>(0) - 1;
    std::mt19937 gen(seed);
    std::normal_distribution<double> distribution(double(maxCoord) / 2, double(maxCoord) / 5);

    auto randInt = [&distribution, &gen, maxCoord]()
    {
        double x = distribution(gen);
        // we can't cut down x to maxCoord in case it's too big, otherwise there will be too many keys in the last cell
        while (x < 0.0 || x > maxCoord)
        {
            x = distribution(gen);
        }
        return Integer(x);
    };

    std::vector<Integer> ret(numKeys);
    std::generate(ret.begin(), ret.end(), randInt);
    std::sort(ret.begin(), ret.end());

    return ret;
}

//! @brief can be used to calculate reasonable smoothing lengths for each particle
template<class KeyType, class Tc, class Th>
void adjustSmoothingLength(LocalIndex numParticles,
                           unsigned ng0,
                           unsigned ngmax,
                           const std::vector<KeyType>& sfcKeys,
                           const std::vector<Tc>& x,
                           const std::vector<Tc>& y,
                           const std::vector<Tc>& z,
                           std::vector<Th>& h,
                           const Box<Tc>& box)
{
    std::vector<cstone::LocalIndex> neighbors(numParticles * ngmax);
    std::vector<unsigned> neighborCounts(numParticles);

    unsigned bucketSize   = 16;
    auto [csTree, counts] = computeOctree<KeyType>(std::span(sfcKeys), bucketSize);
    OctreeData<KeyType, CpuTag> octree;
    octree.resize(nNodes(csTree));
    updateInternalTree<KeyType>(csTree, octree.data());

    std::vector<LocalIndex> layout(nNodes(csTree) + 1, 0);
    std::inclusive_scan(counts.begin(), counts.end(), layout.begin() + 1);

    std::span<const KeyType> nodeKeys(octree.prefixes);
    std::vector<Vec3<Tc>> centers(octree.numNodes), sizes(octree.numNodes);
    nodeFpCenters<KeyType>(nodeKeys, centers.data(), sizes.data(), box);

    OctreeNsView<Tc, KeyType> nsView{octree.numLeafNodes,
                                     octree.numNodes,
                                     octree.prefixes.data(),
                                     octree.childOffsets.data(),
                                     octree.parents.data(),
                                     octree.internalToLeaf.data(),
                                     octree.leafToInternal.data(),
                                     octree.levelRange.data(),
                                     nullptr,
                                     layout.data(),
                                     centers.data(),
                                     sizes.data()};

    // adjust h[i] such that each particle has between ng0/2 and ngmax neighbors
#pragma omp parallel for
    for (LocalIndex i = 0; i < numParticles; ++i)
    {
        int iteration = 0;
        do
        {
            neighborCounts[i] = findNeighbors(i, x.data(), y.data(), z.data(), h.data(), nsView, box, ngmax,
                                              neighbors.data() + i * ngmax);

            const Tc c0 = 1023;
            unsigned nn = std::max(neighborCounts[i], 1u);
            h[i]        = h[i] * 0.5 * pow(1.0 + (c0 * ng0) / nn, 1.0 / 10.0);
        } while ((neighborCounts[i] < ng0 / 4u || neighborCounts[i] >= ngmax) && iteration++ < 10);
    }
}

template<class T, class KeyType_>
class RandomCoordinatesBase
{
public:
    using KeyType = KeyType_;
    using Integer = typename KeyType::ValueType;

    RandomCoordinatesBase(size_t n, Box<T> box)
        : box_(std::move(box))
        , x_(n)
        , y_(n)
        , z_(n)
        , h_(n)
        , codes_(n)
    {
    }

    void adjustH(unsigned ng0, unsigned ngmax)
    {
        if (not isSfcOrdered_) throw std::runtime_error("adjustH can only be called with SFC ordered particles\n");
        adjustSmoothingLength(x_.size(), ng0, ngmax, codes_, x_, y_, z_, h_, box_);
    }

    void shuffle()
    {
        std::vector<LocalIndex> permutation(x_.size());
        std::iota(begin(permutation), end(permutation), LocalIndex(0));
        std::mt19937 gen(0);
        std::ranges::shuffle(permutation, gen);

        std::vector<T> s1(x_.size());
        std::vector<Integer> s2(x_.size());
        gatherArrays(permutation, 0, std::tie(x_, y_, z_, h_, codes_), std::tie(s1, s2));
        isSfcOrdered_ = false;
    }

    const std::vector<T>& x() const { return x_; }
    const std::vector<T>& y() const { return y_; }
    const std::vector<T>& z() const { return z_; }
    const std::vector<T>& h() const { return h_; }
    const std::vector<Integer>& particleKeys() const { return codes_; }
    const Box<T>& box() const { return box_; }

protected:
    void sfcSort()
    {
        std::size_t n = x_.size();
        auto keyData  = (KeyType*)(codes_.data());
        computeSfcKeys(x_.data(), y_.data(), z_.data(), keyData, n, box_);

        std::vector<LocalIndex> sfcOrder(n);
        std::iota(begin(sfcOrder), end(sfcOrder), LocalIndex(0));
        sort_by_key(begin(codes_), end(codes_), begin(sfcOrder));

        std::vector<T> temp(x_.size());
        gatherArrays(sfcOrder, 0, std::tie(x_, y_, z_, h_), std::tie(temp));
        isSfcOrdered_ = true;
    }

    void estimateH(std::size_t hOffset)
    {
        if (not isSfcOrdered_) throw std::runtime_error("estimateH can only be called with SFC ordered particles\n");
        std::size_t n = x_.size();
        for (std::size_t i = 0; i < n; ++i)
        {
            std::size_t j = i > hOffset ? i - hOffset : std::min(n, i + hOffset);

            Vec3<T> pi{x_[i], y_[i], z_[i]};
            Vec3<T> pj{x_[j], y_[j], z_[j]};

            // avoid exact 0.5, because this will lead to lots of borderline particle pairs with dist == searchRadius
            h_[i] = 0.5001 * std::sqrt(norm2(pi - pj));
        }
    }

    bool isSfcOrdered_{false};
    Box<T> box_;
    std::vector<T> x_, y_, z_, h_;
    std::vector<Integer> codes_;
};

template<class T, class KeyType_>
class RandomCoordinates : public RandomCoordinatesBase<T, KeyType_>
{
public:
    using Base = RandomCoordinatesBase<T, KeyType_>;

    RandomCoordinates(std::size_t n, Box<T> box, std::size_t hOffset = 5, int seed = 42)
        : Base(n, box)
    {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<T> disX(box.xmin(), box.xmax());
        std::uniform_real_distribution<T> disY(box.ymin(), box.ymax());
        std::uniform_real_distribution<T> disZ(box.zmin(), box.zmax());

        auto randX = [&disX, &gen]() { return disX(gen); };
        auto randY = [&disY, &gen]() { return disY(gen); };
        auto randZ = [&disZ, &gen]() { return disZ(gen); };

        std::ranges::generate(Base::x_, randX);
        std::ranges::generate(Base::y_, randY);
        std::ranges::generate(Base::z_, randZ);

        Base::sfcSort();
        Base::estimateH(hOffset);
    }
};

template<class T, class KeyType_>
class RandomGaussianCoordinates : public RandomCoordinatesBase<T, KeyType_>
{
public:
    using Base = RandomCoordinatesBase<T, KeyType_>;

    RandomGaussianCoordinates(std::size_t n, Box<T> box, std::size_t hOffset = 5, int seed = 42)
        : Base(n, box)
    {
        std::mt19937 gen(seed);
        std::normal_distribution<T> disX((box.xmax() + box.xmin()) / 2, box.lx() / 5);
        std::normal_distribution<T> disY((box.ymax() + box.ymin()) / 2, box.ly() / 5);
        std::normal_distribution<T> disZ((box.zmax() + box.zmin()) / 2, box.lz() / 5);

        auto makeDist = [&gen](auto& dist, T a, T b)
        {
            return [a, b, &dist, &gen]()
            {
                T x = dist(gen);
                while (x < a || x >= b)
                {
                    x = dist(gen);
                }
                return x;
            };
        };

        std::ranges::generate(Base::x_, makeDist(disX, box.xmin(), box.xmax()));
        std::ranges::generate(Base::y_, makeDist(disY, box.ymin(), box.ymax()));
        std::ranges::generate(Base::z_, makeDist(disZ, box.zmin(), box.zmax()));

        Base::sfcSort();
        Base::estimateH(hOffset);
    }
};

} // namespace cstone
