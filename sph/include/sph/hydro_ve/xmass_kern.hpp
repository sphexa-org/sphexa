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
 * @brief Generalized volume elements
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 */

#pragma once

#include "cstone/cuda/annotation.hpp"
#include "cstone/sfc/box.hpp"
#include "cstone/traversal/ijloop/ijloop.hpp"

#include "sph/kernels.hpp"
#include "sph/table_lookup.hpp"

namespace sph
{

template<class T>
struct XmassInteraction
{
    const T* wh;

    template<class ParticleData, class Tc>
    constexpr auto operator()(const ParticleData& iData, const ParticleData& jData, cstone::Vec3<Tc> const& /* r_ij */,
                              T r2) const
    {
        const auto [i, iPos, hi, mi] = iData;
        const auto [j, jPos, hj, mj] = jData;

        T hInv = 1.0 / hi;

        T dist = std::sqrt(r2);

        T vloc = dist * hInv;
        T w    = lt::lookup(wh, vloc);

        return std::make_tuple(w * mj);
    }
};

//! @brief a particular choice of defining generalized volume elements
template<class T, class Tm>
HOST_DEVICE_FUN inline T veDefinition(Tm mass, T rhoZero)
{
    return mass / rhoZero;
}

template<class T, class Tc>
struct XmassPostamble
{
    Tc K;

    template<class ParticleData, class Result>
    constexpr auto operator()(const ParticleData& iData, const Result& result) const
    {
        const auto [i, iPos, hi, mi] = iData;
        auto [rho0i]                 = result;

        T hInv  = 1.0 / hi;
        T h3Inv = hInv * hInv * hInv;

        return std::make_tuple(veDefinition(mi, rho0i * K * h3Inv));
    }
};

template<size_t stride = 1, class Tc, class Tm, class T>
HOST_DEVICE_FUN inline T xmassJLoop(cstone::LocalIndex i, Tc K, const cstone::Box<Tc>& box,
                                    const cstone::LocalIndex* neighbors, unsigned neighborsCount, const Tc* x,
                                    const Tc* y, const Tc* z, const T* h, const Tm* m, const T* wh, const T* /*whd*/)
{
    XmassInteraction      interaction{wh};
    XmassPostamble<T, Tc> postamble{K};

    const auto input  = std::make_tuple(m);
    T          xmassi = 0;
    const auto output = std::make_tuple((&xmassi) - i);

    const auto iData  = cstone::ijloop::loadParticleData(x, y, z, h, input, i);
    const bool usePbc = cstone::ijloop::requiresPbcHandling(box, iData);

    auto result = interaction(iData, iData, cstone::Vec3<Tc>{0, 0, 0}, T(0));
    for (unsigned pj = 0; pj < neighborsCount; ++pj)
    {
        cstone::LocalIndex j = neighbors[stride * pj];

        const auto jData = cstone::ijloop::loadParticleData(x, y, z, h, input, j);

        const auto [r_ij, r2] = cstone::ijloop::posDiffAndDistSq(usePbc, box, iData, jData);

        cstone::ijloop::updateResult(result, interaction(iData, jData, r_ij, r2));
    }

    result = postamble(iData, result);

    cstone::ijloop::storeParticleData(output, i, result);

    return xmassi;
}

template<class Neighborhood, class Tc, class T, class Tm>
void xmassIjLoop(Neighborhood const& neighborhood, Tc K, const Tm* m, const T* wh, T* xmass)
{
    neighborhood.ijLoop(std::make_tuple(m), std::make_tuple(xmass), XmassInteraction<T>{wh}, XmassPostamble<T, Tc>{K},
                        cstone::ijloop::symmetry::even);
}

} // namespace sph
