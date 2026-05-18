/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Computing cluster overhead for various interaction radii
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#include <algorithm>
#include <cassert>
#include <format>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "../coord_samples/random.hpp"
#include "performance/csv.hpp"

constexpr std::size_t ngmax = 4096;

std::tuple<std::vector<unsigned>, std::vector<unsigned>> findNeighbors(const cstone::Box<double>& box,
                                                                       const double* __restrict__ x,
                                                                       const double* __restrict__ y,
                                                                       const double* __restrict__ z,
                                                                       const unsigned n,
                                                                       const double r)
{
    std::vector<unsigned> neighborsCount(n), neighbors(n * ngmax);
#pragma omp parallel for
    for (unsigned i = 0; i < n; ++i)
    {
        double xi = x[i];
        double yi = y[i];
        double zi = z[i];
        for (unsigned j = 0; j < n; ++j)
        {
            double xj  = x[j];
            double yj  = y[j];
            double zj  = z[j];
            double xij = xi - xj;
            double yij = yi - yj;
            double zij = zi - zj;
            applyPBC(box, r, xij, yij, zij);
            if (xij * xij + yij * yij + zij * zij < r * r)
            {
                assert(neighborsCount[i] < ngmax);
                neighbors[i * ngmax + neighborsCount[i]++] = j;
            }
        }
    }
    return {std::move(neighborsCount), std::move(neighbors)};
}

void filterNeighbors(const cstone::Box<double>& box,
                     const double* __restrict__ x,
                     const double* __restrict__ y,
                     const double* __restrict__ z,
                     std::tuple<std::vector<unsigned>, std::vector<unsigned>>& neighborhood,
                     const double r)
{
    auto& neighborsCount = std::get<0>(neighborhood);
    auto& neighbors      = std::get<1>(neighborhood);
    const unsigned n     = neighborsCount.size();
#pragma omp parallel for
    for (unsigned i = 0; i < n; ++i)
    {
        unsigned nbs = 0;
        double xi    = x[i];
        double yi    = y[i];
        double zi    = z[i];
        for (unsigned nb = 0; nb < neighborsCount[i]; ++nb)
        {
            const unsigned j = neighbors[i * ngmax + nb];
            double xj        = x[j];
            double yj        = y[j];
            double zj        = z[j];
            double xij       = xi - xj;
            double yij       = yi - yj;
            double zij       = zi - zj;
            applyPBC(box, r, xij, yij, zij);
            if (xij * xij + yij * yij + zij * zij < r * r)
            {
                assert(nbs < ngmax);
                neighbors[i * ngmax + nbs++] = j;
            }
        }
        neighborsCount[i] = nbs;
    }
}

double computeClusterOverhead(const std::tuple<std::vector<unsigned>, std::vector<unsigned>>& neighborhood,
                              const std::tuple<unsigned, unsigned>& clusterSize)
{
    auto const& neighborsCount = std::get<0>(neighborhood);
    auto const& neighbors      = std::get<1>(neighborhood);

    const unsigned iSize = std::get<0>(clusterSize);
    const unsigned jSize = std::get<1>(clusterSize);

    const unsigned n = neighborsCount.size();

    unsigned long actualPairInteractions = 0;
#pragma omp parallel for reduction(+ : actualPairInteractions)
    for (unsigned i = 0; i < n; ++i)
        actualPairInteractions += neighborsCount[i];

    unsigned long clusterPairInteractions = 0;
#pragma omp parallel for reduction(+ : clusterPairInteractions)
    for (unsigned ic = 0; ic < (n + iSize - 1) / iSize; ++ic)
    {
        std::set<unsigned> clusterNeighbors;

        for (unsigned i = ic * iSize; i < std::min(ic * iSize + iSize, n); ++i)
        {
            const unsigned nbs = neighborsCount[i];
            for (unsigned nb = 0; nb < nbs; ++nb)
            {
                const unsigned j = neighbors[i * ngmax + nb];
                clusterNeighbors.insert(j / jSize);
            }
        }

        clusterPairInteractions += clusterNeighbors.size() * iSize * jSize;
    }
    return double(clusterPairInteractions) / double(actualPairInteractions);
}

std::vector<double> radii()
{
    std::vector<double> rs;
    for (unsigned i = 20; i >= 5; --i)
        rs.push_back(i / 10.0);
    return rs;
}

std::vector<std::tuple<unsigned, unsigned>> clusterSizes() { return {{1, 1}, {2, 2}, {4, 2}, {4, 4}, {4, 8}}; }

double expectedNumberOfNeighbors(const cstone::Box<double>& box, const unsigned n, const double r)
{
    return 4.0 / 3.0 * M_PI * r * r * r * n / (box.lx() * box.ly() * box.lz());
}

int main()
{
    using namespace cstone;
    using StrongKeyType = cstone::HilbertKey<std::uint64_t>;

    constexpr unsigned n = 100000;
    const double rfac    = 1.0 / 10.0;
    const Box<double> box{0.0, 1.0, BoundaryType::periodic};
    RandomCoordinates<double, StrongKeyType> coords(n, box);

    auto rs = radii();
    auto cs = clusterSizes();

    auto fmtClusterSize = [](std::tuple<unsigned, unsigned> const& c)
    { return std::format("{}x{}", std::get<0>(c), std::get<1>(c)); };

    std::map<std::string, std::vector<double>> clusterOverheads;
    clusterOverheads["r"]   = std::vector<double>();
    clusterOverheads["nbs"] = std::vector<double>();
    for (auto const& c : cs)
        clusterOverheads[fmtClusterSize(c)] = std::vector<double>();

    auto neighborhood = findNeighbors(box, coords.x().data(), coords.y().data(), coords.z().data(), n, rs[0] * rfac);

    for (unsigned i = 0; i < rs.size(); ++i)
    {
        const double r = rs[i];
        for (auto const& c : cs)
            clusterOverheads[fmtClusterSize(c)].push_back(computeClusterOverhead(neighborhood, c));
        clusterOverheads["r"].push_back(r);
        clusterOverheads["nbs"].push_back(expectedNumberOfNeighbors(box, n, r * rfac));

        if (i + 1 < rs.size())
            filterNeighbors(box, coords.x().data(), coords.y().data(), coords.z().data(), neighborhood,
                            rs[i + 1] * rfac);
    }

    for (auto& ce : clusterOverheads)
    {
        auto& v = std::get<1>(ce);
        std::reverse(v.begin(), v.end());
    }

    saveCsv(std::cout, clusterOverheads);
    saveCsv("cluster-overhead.csv", clusterOverheads);

    return 0;
}
