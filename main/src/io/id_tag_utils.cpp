/*
 * MIT License
 *
 * SPH-EXA
 * Copyright (c) 2024 CSCS, ETH Zurich, University of Basel, University of Zurich
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
 * @brief  CPU/GPU Particle ID tag utilities, CPU implementations
 *
 * @author Christopher Bignamini <christopher.bignamini@gmail.com>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "id_tag_utils.hpp"

namespace sphexa
{

/*! @brief Id tagging (in first:last range) from list, CPU version
 *
 * @param[out] ids          ordered id list
 * @param[in]  first        first id index // TODO number of elements and pass iterator?
 * @param[in]  last         last (excluded) id index
 * @param[in]  selectedIds  indexes to be tagged
 */
void tagIdsInList(std::vector<uint64_t>& ids, size_t first, size_t last, const std::vector<uint64_t>& selectedIds)
{
    const auto idListBeginIt = ids.begin()+first;
    const auto idListEndIt = ids.begin()+last;
    auto lastFound = 0;
    std::for_each(selectedIds.begin(), selectedIds.end(), [idListBeginIt, idListEndIt, &lastFound](auto selectedIds){
        auto lower = std::lower_bound(idListBeginIt+lastFound, idListEndIt, selectedIds);
        if(lower != idListEndIt && *lower == selectedIds) {
            lastFound = lower - idListBeginIt + 1;
            *lower = *lower | msbMask;
        }
    });
}

/*! @brief Id tagging (in first:last range) in spherical volume, CPU version
 *
 * @param[out] ids           ordered id list
 * @param[in]  x             x coordinates
 * @param[in]  y             y coordinates
 * @param[in]  z             z coordinates
 * @param[in]  first         first id index // TODO number of elements and pass iterator?
 * @param[in]  last          last (excluded) id index
 * @param[in]  selSphereData spherical volume definition
 */
void tagIdsInSphere(std::vector<uint64_t>& ids, const std::vector<CoordinateType>& x, const std::vector<CoordinateType>& y,
    const std::vector<CoordinateType>& z, size_t firstIndex, size_t lastIndex, const IdSelectionSphere& selSphereData)
{
    // TODO: can we use C++23 zip iterators? Is there anything already implemented in SPH-EXA?
    const auto squareRadius = selSphereData.radius*selSphereData.radius;
//#pragma omp parallel for
    for(auto particleIndex = firstIndex; particleIndex < lastIndex; particleIndex++){
        auto current_x = x[particleIndex];
        auto current_y = y[particleIndex];
        auto current_z = z[particleIndex];
        if((current_x - selSphereData.center[0])*(current_x - selSphereData.center[0]) +
            (current_y - selSphereData.center[1])*(current_y - selSphereData.center[1]) +
            (current_z - selSphereData.center[2])*(current_z - selSphereData.center[2]) <= squareRadius) {
            ids[particleIndex] = ids[particleIndex] | msbMask;
        }
    }
}


/*! @brief Tagged id identification function  
 *
 * @param[in]  ids          ordered id list
 * @param[in]  first        first id index // TODO number of elements and pass iterator?
 * @param[in]  last         last (excluded) id index
 * @param[out] taggedIdsIndexes  vector of indexes (positions wrt of selected particles
 */
void findTaggedIds(const std::vector<uint64_t>& ids, size_t first, size_t last, std::vector<uint64_t>& taggedIdsIndexes)
{
    // Find the selected particles in local id list and save their indexes
    // TODO: switch to GPU-like implementation?
    uint64_t idIndex = first;
    std::for_each(ids.begin()+first, ids.begin()+last, [&taggedIdsIndexes, &idIndex](auto& id){
        if((id & sphexa::msbMask) != 0) {
            taggedIdsIndexes.push_back(idIndex); // TODO: inefficient due to resizing, avoid push_back usage
        }
        idIndex++;
    });
}

}