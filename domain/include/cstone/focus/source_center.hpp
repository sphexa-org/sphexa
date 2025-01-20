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
 * @brief Compute leaf cell source centers based on local information
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/util/array.hpp"
#include "cstone/tree/octree.hpp"
#include "cstone/traversal/macs.hpp"

namespace cstone
{

template<class T>
using SourceCenterType = util::array<T, 4>;

//! @brief add a single body contribution to a mass center
template<class T>
HOST_DEVICE_FUN void addBody(SourceCenterType<T>& center, const SourceCenterType<T>& source)
{
    T weight = std::abs(source[3]);

    center[0] += weight * source[0];
    center[1] += weight * source[1];
    center[2] += weight * source[2];
    center[3] += weight;
}

//! @brief finish mass center computation by dividing coordinates by total absolute mass
template<class T>
HOST_DEVICE_FUN SourceCenterType<T> normalizeMass(SourceCenterType<T> center)
{
    T invM = (center[3] != T(0.0)) ? T(1.0) / center[3] : T(1.0);
    center[0] *= invM;
    center[1] *= invM;
    center[2] *= invM;

    return center;
}

//! @brief compute a mass center from particles
template<class Ts, class Tc, class Tm>
HOST_DEVICE_FUN SourceCenterType<Ts>
massCenter(const Tc* x, const Tc* y, const Tc* z, const Tm* m, LocalIndex first, LocalIndex last)
{
    SourceCenterType<Ts> center{0, 0, 0, 0};
    for (LocalIndex i = first; i < last; ++i)
    {
        addBody(center, SourceCenterType<Ts>{Ts(x[i]), Ts(y[i]), Ts(z[i]), Ts(m[i])});
    }

    return normalizeMass(center);
}

//! @brief compute a mass center from other mass centers for use in tree upsweep
template<class T>
struct CombineSourceCenter
{
    HOST_DEVICE_FUN
    SourceCenterType<T> operator()(TreeNodeIndex /*nodeIdx*/, TreeNodeIndex child, const SourceCenterType<T>* centers)
    {
        SourceCenterType<T> center{0, 0, 0, 0};

        for (TreeNodeIndex i = child; i < child + 8; ++i)
        {
            addBody(center, centers[i]);
        }
        return normalizeMass(center);
    }
};

/*! @brief compute mass center coordinates for leaf nodes
 *
 * @param x                 source body x coordinates
 * @param y                 source body y coordinates
 * @param z                 source body z coordinates
 * @param m                 source body masses
 * @param leafToInternal    translation map from cornerstone leaf cell array indices to node indices of the full
 *                          octree
 * @param layout            array of length numLeafNodes + 1, the i-th element contains the index to of the first
 *                          particle in x,y,z,m contained in the i-th leaf node of the octree
 * @param sourceCenter      array of length numNodes of the full octree
 */
template<class T1, class T2, class T3>
void computeLeafMassCenter(gsl::span<const T1> x,
                           gsl::span<const T1> y,
                           gsl::span<const T1> z,
                           gsl::span<const T2> m,
                           gsl::span<const TreeNodeIndex> leafToInternal,
                           const LocalIndex* layout,
                           SourceCenterType<T3>* sourceCenter)
{
#pragma omp parallel for
    for (size_t leafIdx = 0; leafIdx < leafToInternal.size(); ++leafIdx)
    {
        TreeNodeIndex i = leafToInternal[leafIdx];
        sourceCenter[i] = massCenter<T3>(x.data(), y.data(), z.data(), m.data(), layout[leafIdx], layout[leafIdx + 1]);
    }
}

//! @brief replace the last center element (mass) with the squared mac radius
template<class T, class KeyType>
void setMac(gsl::span<const KeyType> nodeKeys,
            gsl::span<SourceCenterType<T>> centers,
            float invTheta,
            const Box<T>& box)
{
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nodeKeys.size(); ++i)
    {
        Vec4<T> center = centers[i];
        T mac          = computeVecMacR2(nodeKeys[i], util::makeVec3(center), invTheta, box);
        centers[i][3]  = (center[3] != T(0)) ? mac : T(0);
    }
}

//! @brief compute geometric node centers based on node SFC keys and the global bounding box
template<class KeyType, class T>
void nodeFpCenters(gsl::span<const KeyType> prefixes, Vec3<T>* centers, Vec3<T>* sizes, const Box<T>& box)
{
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < prefixes.size(); ++i)
    {
        KeyType prefix                  = prefixes[i];
        KeyType startKey                = decodePlaceholderBit(prefix);
        unsigned level                  = decodePrefixLength(prefix) / 3;
        auto nodeBox                    = sfcIBox(sfcKey(startKey), level);
        util::tie(centers[i], sizes[i]) = centerAndSize<KeyType>(nodeBox, box);
    }
}

//! @brief set @p centers to geometric node centers with Mac radius l * invTheta
template<class KeyType, class T>
void geoMacSpheres(gsl::span<const KeyType> prefixes, SourceCenterType<T>* centers, float invTheta, const Box<T>& box)
{
#pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < prefixes.ssize(); ++i)
    {
        centers[i] = computeMinMacR2(prefixes[i], invTheta, box);
    }
}

} // namespace cstone
