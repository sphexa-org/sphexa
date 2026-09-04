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
 * @brief Volume definition and gradient of h architecture portable kernel
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 */

#pragma once

#include "cstone/traversal/ijloop/ijloop.hpp"

#include "sph/sph_kernels.hpp"

namespace sph
{

template<class T, class Kernel>
struct VeInteraction
{
    Kernel wh;

    template<class ParticleData, class Tc>
    constexpr auto operator()(const ParticleData& iData, const ParticleData& jData, cstone::Vec3<Tc> const& /* r_ij */,
                              T r2) const
    {
        const auto [i, iPos, hi, xmassi] = iData;
        const auto [j, jPos, hj, xmassj] = jData;

        auto hInv = T(1) / hi;

        T dist = std::sqrt(r2);
        T vloc = dist * hInv;
        T w    = wh(vloc);

        T kxi = w * xmassj;

        return std::make_tuple(kxi);
    }
};

template<class T, class Tc>
struct VePostamble
{
    Tc K;

    template<class ParticleData, class Result>
    constexpr auto operator()(const ParticleData& iData, const Result& result) const
    {
        const auto [i, iPos, hi, xmassi] = iData;
        auto [kxi]                       = result;

        auto hInv  = T(1) / hi;
        auto h3Inv = hInv * hInv * hInv;

        kxi *= K * h3Inv;

        return std::make_tuple(kxi);
    }
};

template<class Neighbordhood, class Tc, class T>
void veIjLoop(const Neighbordhood& neighborhood, Tc K, const T* xm, KernelVariant<T> const& wh, T* kx)
{
    std::visit(
        [&]<class Kernel>(Kernel wh)
        {
            neighborhood.ijLoop(cstone::ijloop::makeIjLoopData<Tc, T*>(
                std::make_tuple(xm), std::make_tuple(kx), VeInteraction<T, Kernel>{wh}, VePostamble<T, Tc>{K}));
        },
        wh);
}

} // namespace sph
