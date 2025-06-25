/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief General data structures and functions for the supercluster neighborhood
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <cassert>

namespace cstone::ijloop::gpu_supercluster_nb_list_neighborhood_detail
{

struct SuperclusterInfo
{
    //! @brief index of the supercluster, defining which particles belong to it
    unsigned index;
    //! @brief number of neighbor clusters
    unsigned neighborsCount;
    //! @brief start index in the neighbor data arra.
    std::size_t dataIndex;

    //! @brief less-than operator for sorting superclusters by descending neighbor count (for load balancing)
    constexpr bool operator<(const SuperclusterInfo& other) const { return neighborsCount > other.neighborsCount; }
};

/*! amount of storage required by the bitmasks stored per supercluster
 *
 * @param[in] numJClusters number of neighboring j clusters of the supercluster
 *
 * @return number of 32bit integers required to store the bitmasks
 */
template<class Config>
constexpr __forceinline__ unsigned masksSize(unsigned numJClusters)
{
    return (numJClusters * Config::iClustersPerSupercluster * Config::numWarpsPerInteraction + 31) / 32;
}

//! supercluster index of a particle
template<class Config>
constexpr __forceinline__ unsigned superclusterIndex(unsigned i)
{
    return i / Config::superclusterSize;
}

//! j-cluster index of a particle
template<class Config>
constexpr __forceinline__ unsigned jClusterIndex(unsigned j)
{
    return j / Config::jSize;
}

/*! start particle index offset of the first supercluster, required to align the supercluster boundaries to the first
 * traversed particle (i.e. first domain particle instead of first halo particle)
 *
 * @param[in] firstBody index of the first domain particle
 *
 * @return required particle index shift
 */
template<class Config>
constexpr __forceinline__ unsigned clusterOffset(unsigned firstBody)
{
    const unsigned offset =
        (firstBody + Config::superclusterSize - 1) / Config::superclusterSize * Config::superclusterSize - firstBody;
    assert(offset < Config::superclusterSize);
    return offset;
}

} // namespace cstone::ijloop::gpu_supercluster_nb_list_neighborhood_detail
