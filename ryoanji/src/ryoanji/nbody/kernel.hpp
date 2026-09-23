/*
 * Ryoanji N-body solver
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

#pragma once

/*! @file
 * @brief  Particle-2-particle (P2P) kernel
 *
 * @author Rio Yokota <rioyokota@gsic.titech.ac.jp>
 */

#include "types.h"

namespace ryoanji
{

HOST_DEVICE_FUN HOST_DEVICE_INLINE float inverseSquareRoot(float x)
{
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return rsqrtf(x);
#else
    return 1.0f / std::sqrt(x);
#endif
}

HOST_DEVICE_FUN HOST_DEVICE_INLINE double inverseSquareRoot(double x)
{
#if defined(__HIP_DEVICE_COMPILE__)
    return rsqrt(x);
#elif defined(__CUDA_ARCH__)
    /*! @brief inline reciprocal square root
     *
     * Seeds with the hardware single-precision reciprocal square root (~2^-23 relative error)
     * and refines with one Newton-Raphson iteration, doubling the correct bits to ~2^-46.
     * This is far below the precision of the float accelerations the result is accumulated
     * into, but much cheaper than the ~2 ulp double-precision libdevice rsqrt, whose
     * special-case handling and extra refinements are not needed for squared distances.
     *
     * Note: unlike libdevice rsqrt, x == 0 returns NaN instead of inf. Downstream use in P2P/M2P
     * produces NaN either way for coincident particles with zero softening.
     */
    double y = (double)rsqrtf((float)x);
    y *= (1.5 - 0.5 * x * y * y);
    return y;
#else
    return 1.0 / std::sqrt(x);
#endif
}

/*! @brief interaction between two particles
 *
 * @param acc     acceleration to add to
 * @param pos_i
 * @param pos_j
 * @param m_j
 * @param h_i
 * @param h_j
 * @return        input acceleration plus contribution from this call
 */
template<class Ta, class Tc, class Th, class Tm>
HOST_DEVICE_FUN DEVICE_INLINE Vec4<Ta> P2P(Vec4<Ta> acc, const Vec3<Tc>& pos_i, const Vec3<Tc>& pos_j, Tm m_j, Th h_i,
                                           Th h_j)
{
    Vec3<Tc> dX = pos_j - pos_i;
    Tc       R2 = norm2(dX);

    Th h_ij  = h_i + h_j;
    Th h_ij2 = h_ij * h_ij;
    Tc R2eff = (R2 < h_ij2) ? h_ij2 : R2;

    Tc invR   = inverseSquareRoot(R2eff);
    Tc invR2  = invR * invR;
    Tc invR3m = m_j * invR * invR2;

    acc[0] -= invR3m * R2;
    acc[1] += dX[0] * invR3m;
    acc[2] += dX[1] * invR3m;
    acc[3] += dX[2] * invR3m;

    return acc;
}

} // namespace ryoanji
