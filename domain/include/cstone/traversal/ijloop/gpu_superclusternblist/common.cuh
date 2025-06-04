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
 * @brief General data structures and functions for the supercluster neighborhood
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

namespace cstone::ijloop::gpu_supercluster_nb_list_neighborhood_detail
{

struct SuperclusterInfo
{
    //! @brief index of the supercluster, defining which particles belong to it
    unsigned index;
    //! @brief number of neighbor clusters
    unsigned neighborsCount;
    //! @brief start index in the neighbor data arra.
    unsigned dataIndex;

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
