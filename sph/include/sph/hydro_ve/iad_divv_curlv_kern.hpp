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
 * @brief Fused IAD and divVCurlV kernels
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <limits>

#include "sph/hydro_ve/iad_kern.hpp"
#include "sph/hydro_ve/divv_curlv_kern.hpp"

namespace sph
{

template<class T>
struct IADDivVCurlVInteraction
{
    const T* wh;

    template<class ParticleData, class Tc>
    constexpr auto operator()(const ParticleData& iData, const ParticleData& jData, const cstone::Vec3<Tc>& r_ij,
                              const T r2) const
    {
        auto const [i, iPos, hi, vxi, vyi, vzi, xmi, kxi] = iData;
        auto const [j, jPos, hj, vxj, vyj, vzj, xmj, kxj] = jData;

        auto iadIData  = std::make_tuple(i, iPos, hi, xmi, kxi);
        auto iadJData  = std::make_tuple(j, jPos, hj, xmj, kxj);
        auto iadResult = IADInteraction<T>{wh}(iadIData, iadJData, r_ij, r2);

        // c11i ... c33i are only read in postamble, so we can pass dummy values here
        T    dummy = std::numeric_limits<T>::signaling_NaN();
        auto divVCurlVIData =
            std::make_tuple(i, iPos, hi, vxi, vyi, vzi, xmi, kxi, dummy, dummy, dummy, dummy, dummy, dummy);
        auto divVCurlVJData =
            std::make_tuple(j, jPos, hj, vxj, vyj, vzj, xmj, kxj, dummy, dummy, dummy, dummy, dummy, dummy);
        auto divVCurlVResult = DivVCurlVInteraction<T>{wh}(divVCurlVIData, divVCurlVJData, r_ij, r2);

        return std::tuple_cat(iadResult, divVCurlVResult);
    }
};

template<bool DoCurlV, bool DoGradV, class T, class Tc>
struct IADDivVCurlVPostamble
{
    Tc K;

    template<class ParticleData, class Result>
    constexpr auto operator()(const ParticleData& iData, const Result& result) const
    {
        auto const [i, iPos, hi, vxi, vyi, vzi, xmi, kxi]         = iData;
        auto [tau11, tau12, tau13, tau22, tau23, tau33, dVxiXFactor, dVxiYFactor, dVxiZFactor, dVyiXFactor, dVyiYFactor,
              dVyiZFactor, dVziXFactor, dVziYFactor, dVziZFactor] = result;

        auto const [c11i, c12i, c13i, c22i, c23i, c33i] = IADPostamble<T, Tc>{K}(
            std::make_tuple(i, iPos, hi, xmi, kxi), std::make_tuple(tau11, tau12, tau13, tau22, tau23, tau33));

        auto const divVCurlVResult = DivVCurlVPostamble<DoCurlV, DoGradV, T, Tc>{K}(
            std::make_tuple(i, iPos, hi, vxi, vyi, vzi, xmi, kxi, c11i, c12i, c13i, c22i, c23i, c33i),
            std::make_tuple(dVxiXFactor, dVxiYFactor, dVxiZFactor, dVyiXFactor, dVyiYFactor, dVyiZFactor, dVziXFactor,
                            dVziYFactor, dVziZFactor));

        return std::tuple_cat(std::make_tuple(c11i, c12i, c13i, c22i, c23i, c33i), divVCurlVResult);
    }
};

template<class Neighborhood, class Tc, class T>
void iadDivVCurlVIjLoop(const Neighborhood& neighborhood, Tc K, const T* vx, const T* vy, const T* vz, const T* xm,
                        const T* kx, T* c11, T* c12, T* c13, T* c22, T* c23, T* c33, const T* wh, T* divv, T* curlv,
                        T* dV11, T* dV12, T* dV13, T* dV22, T* dV23, T* dV33, bool doGradV)
{
    const auto input = std::make_tuple(vx, vy, vz, xm, kx);
    // TODO: check symmetry
    if (curlv && doGradV)
    {
        const auto output =
            std::make_tuple(c11, c12, c13, c22, c23, c33, divv, curlv, dV11, dV12, dV13, dV22, dV23, dV33);
        neighborhood.ijLoop(input, output, IADDivVCurlVInteraction<T>{wh}, IADDivVCurlVPostamble<true, true, T, Tc>{K});
    }
    else if (curlv)
    {
        const auto output = std::make_tuple(c11, c12, c13, c22, c23, c33, divv, curlv);
        neighborhood.ijLoop(input, output, IADDivVCurlVInteraction<T>{wh},
                            IADDivVCurlVPostamble<true, false, T, Tc>{K});
    }
    else if (doGradV)
    {
        const auto output = std::make_tuple(c11, c12, c13, c22, c23, c33, divv, dV11, dV12, dV13, dV22, dV23, dV33);
        neighborhood.ijLoop(input, output, IADDivVCurlVInteraction<T>{wh},
                            IADDivVCurlVPostamble<false, true, T, Tc>{K});
    }
    else
    {
        const auto output = std::make_tuple(c11, c12, c13, c22, c23, c33, divv);
        neighborhood.ijLoop(input, output, IADDivVCurlVInteraction<T>{wh},
                            IADDivVCurlVPostamble<false, false, T, Tc>{K});
    }
}

} // namespace sph
