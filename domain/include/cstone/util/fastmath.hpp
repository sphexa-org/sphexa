/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Fast device math functions with possibly lower precision.
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#include <cmath>

namespace util::fastmath
{

constexpr float sin(float x)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __sinf(x);
#else
    return std::sin(x);
#endif
}

constexpr double sin(double x) { return std::sin(x); }

constexpr float sqrt(float x)
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

constexpr double sqrt(double x)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __dsqrt_rn(x);
#else
    return std::sqrt(x);
#endif
}

constexpr float rsqrt(float x)
{
#ifdef __CUDA_ARCH__
    // __frsqrt_rn might not flush to zero and thus can be significantly slower
    asm("rsqrt.approx.ftz.f32 %0,%0;" : "+f"(x) :);
    return x;
#elif defined(__HIP_DEVICE_COMPILE__)
    return __frsqrt_rn(x);
#else
    return 1.0f / std::sqrt(x);
#endif
}

constexpr double rsqrt(double x)
{
#ifdef __CUDA_ARCH__
    // ::rsqrt might not flush to zero and thus can be significantly slower
    asm("rsqrt.approx.ftz.f64 %0,%0;" : "+d"(x) :);
    return x;
#elif defined(__HIP_DEVICE_COMPILE__)
    return ::rsqrt(x);
#else
    return 1.0 / std::sqrt(x);
#endif
}

constexpr float cbrt(float x)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::cbrtf(x);
#else
    return std::cbrt(x);
#endif
}

constexpr float cbrt(double x)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::cbrt(x);
#else
    return std::cbrt(x);
#endif
}

constexpr float rcp(float x)
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

constexpr double rcp(double x)
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

constexpr float div(float x, float y)
{
#ifdef __CUDA_ARCH__
    return __fdividef(x, y);
#elif defined(__HIP_DEVICE_COMPILE__)
    return __fdiv_rn(x, y);
#else
    return x / y;
#endif
}

constexpr double div(double x, double y)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __ddiv_rn(x, y);
#else
    return x / y;
#endif
}

constexpr float pow(float x, float y)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __powf(x, y);
#else
    return std::pow(x, y);
#endif
}

constexpr double pow(double x, double y) { return std::pow(x, y); }

constexpr float exp(float x)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __expf(x);
#else
    return std::exp(x);
#endif
}

constexpr double exp(double x)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::exp(x);
#else
    return std::exp(x);
#endif
}

} // namespace util::fastmath
