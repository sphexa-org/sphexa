/*
 * MIT License
 *
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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUTh WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUTh NOTh LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENTh SHALL THE
 * AUTHORS OR COPYRIGHTh HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORTh OR OTHERWISE, ARISING FROM,
 * OUTh OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file integrate quantities related to magneto-hydrodynamics
 *
 * uses Adams-Bashforth 2nd order integration
 * @author Lukas Schmidt
 */

#pragma once
#include "cstone/primitives/accel_switch.hpp"

#include "sph/sph_gpu.hpp"

namespace sph::magneto
{
//! @brief update a quantity according to Adams-Bashforth (2nd order)
template<class TU, class TD, class TDm1>
HOST_DEVICE_FUN TU updateQuantity(TU q_old, double dt, double dt_m1, TD dq, TDm1 dq_m1)
{
    TU u_new = q_old + dq * dt + 0.5 * (dq - dq_m1) / dt_m1 * std::abs(dt) * dt;
    return u_new;
}

template<class Tc, class Tm1>
void integrateMagneticField(size_t firstIndex, size_t lastIndex, double dt, double dt_m1, Tc* Bx, Tc* By, Tc* Bz,
                            Tc* dBx, Tc* dBy, Tc* dBz, Tm1* dBx_m1, Tm1* dBy_m1, Tm1* dBz_m1)
{

#pragma omp parallel for schedule(static)
    for (size_t i = firstIndex; i < lastIndex; ++i)
    {
        Bx[i] = updateQuantity(Bx[i], dt, dt_m1, dBx[i], dBx_m1[i]);
        By[i] = updateQuantity(By[i], dt, dt_m1, dBy[i], dBy_m1[i]);
        Bz[i] = updateQuantity(Bz[i], dt, dt_m1, dBz[i], dBz_m1[i]);

        dBx_m1[i] = dBx[i];
        dBy_m1[i] = dBy[i];
        dBz_m1[i] = dBz[i];
    }
}

template<class Th, class Tm1>
void integrateAuxiliaryQuantities(size_t firstIndex, size_t lastIndex, double dt, double dt_m1, Th* psi_ch,
                                  Th* d_psi_ch, Tm1* d_psi_ch_m1)
{

#pragma omp parallel for schedule(static)
    for (size_t i = firstIndex; i < lastIndex; ++i)
    {
        psi_ch[i]      = updateQuantity(psi_ch[i], dt, dt_m1, d_psi_ch[i], d_psi_ch_m1[i]);
        d_psi_ch_m1[i] = d_psi_ch[i];
    }
}

template<class MagnetoData>
void integrateMagneticQuantities(const GroupView grp, MagnetoData& md, float dt, float dt_m1)
{
    if constexpr (cstone::HaveGpu<typename MagnetoData::AcceleratorType>{})
    {
        cuda::integrateMagneticQuantitiesGpu(grp, md, dt, dt_m1);
    }
    integrateMagneticField(grp.firstBody, grp.lastBody, dt, dt_m1, md.Bx.data(), md.By.data(), md.Bz.data(),
                           md.dBx.data(), md.dBy.data(), md.dBz.data(), md.dBx_m1.data(), md.dBy_m1.data(),
                           md.dBz_m1.data());
    integrateAuxiliaryQuantities(grp.firstBody, grp.lastBody, dt, dt_m1, md.psi_ch.data(), md.d_psi_ch.data(),
                                 md.d_psi_ch_m1.data());
}

} // namespace sph::magneto
