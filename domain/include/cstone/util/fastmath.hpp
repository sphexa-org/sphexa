/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Math functions that resolve to fast-math equivalents even without fast-math compiler flag
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <cmath>

#define CSTONE_FAST_MATH [[gnu::optimize("-ffast-math")]]

namespace util::fastmath
{

CSTONE_FAST_MATH constexpr float sin(float x)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __sinf(x);
#else
    return std::sin(x);
#endif
}

CSTONE_FAST_MATH constexpr double sin(double x) { return std::sin(x); }

CSTONE_FAST_MATH constexpr float cos(float x)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __cosf(x);
#else
    return std::cos(x);
#endif
}

CSTONE_FAST_MATH constexpr double cos(double x) { return std::cos(x); }

CSTONE_FAST_MATH constexpr float rcp(float x)
{
#ifdef __CUDA_ARCH__
    // __frcp_rn might not flush to zero and thus can be significantly slower
    asm("rcp.approx.ftz.f32 %0,%0;" : "+f"(x) :);
    return x;
#elif defined(__HIP_DEVICE_COMPILE__)
    return __frcp_rn(x);
#else
    return 1.0f / x;
#endif
}

CSTONE_FAST_MATH constexpr double rcp(double x)
{
#ifdef __CUDA_ARCH__
    // __drcp_rn might not flush to zero and thus can be significantly slower
    asm("rcp.approx.ftz.f64 %0,%0;" : "+d"(x) :);
    return x;
#elif defined(__HIP_DEVICE_COMPILE__)
    return __drcp_rn(x);
#else
    return 1.0 / x;
#endif
}

CSTONE_FAST_MATH constexpr float sqrt(float x)
{
#if defined(__CUDA_ARCH__)
    // __fsqrt_rn might not flush to zero and thus can be significantly slower
    asm("sqrt.approx.ftz.f32 %0,%0;" : "+f"(x) :);
    return x;
#elif defined(__HIP_DEVICE_COMPILE__)
    return __fsqrt_rn(x);
#else
    return std::sqrt(x);
#endif
}

CSTONE_FAST_MATH constexpr double sqrt(double x)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __dsqrt_rn(x);
#else
    return std::sqrt(x);
#endif
}

CSTONE_FAST_MATH constexpr float pow(float x, float y)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __powf(x, y);
#else
    return std::pow(x, y);
#endif
}

CSTONE_FAST_MATH constexpr double pow(double x, double y) { return std::pow(x, y); }

} // namespace util::fastmath
