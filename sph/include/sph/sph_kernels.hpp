/*! @file
 * @brief Definition, selection and tabulation of smoothing kernels
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <variant>
#include <vector>

#include "cstone/primitives/fastmath.hpp"
#include "cstone/util/reallocate.hpp"
#include "kernels.hpp"

namespace util
{
/*! @brief integrate a function according to simpson's rule
 *
 * @tparam F
 * @param a      start of the integration interval
 * @param b      end of the integration interval
 * @param n      number of intervals
 * @param func   the integrand to be integrated
 * @return       the integral of func over [a, b]
 */
template<class F>
constexpr double simpson(double a, double b, uint64_t n, F&& func)
{
    uint64_t numOdd  = n / 2;
    uint64_t numEven = (numOdd >= 1) ? numOdd - 1 : 0;
    double   h       = (b - a) / double(n);

    std::vector<double> samplesOdd(numOdd);
    std::vector<double> samplesEven(numEven);

    for (uint64_t i = 0; i < numOdd; ++i)
    {
        uint64_t idx  = 2 * (i + 1) - 1;
        double   x    = a + double(idx) * h;
        samplesOdd[i] = func(x);
    }
    for (uint64_t i = 0; i < numEven; ++i)
    {
        uint64_t idx   = 2 * (i + 1);
        double   x     = a + double(idx) * h;
        samplesEven[i] = func(x);
    }
    // optional sorting for better accuracy
    std::sort(samplesOdd.begin(), samplesOdd.end());
    std::sort(samplesEven.begin(), samplesEven.end());

    return h / 3.0 *
           (func(a) + func(b) + 4.0 * std::accumulate(samplesOdd.begin(), samplesOdd.end(), 0.0) +
            2.0 * std::accumulate(samplesEven.begin(), samplesEven.end(), 0.0));
}
} // namespace util

namespace sph
{

//! @brief reference normalization constant from interpolation constants, now legacy functionality
template<typename T>
T sphynx_3D_k(T n)
{
    // b0, b1, b2 and b3 are defined in "SPHYNX: an accurate density-based SPH method for astrophysical applications",
    // DOI: 10.1051/0004-6361/201630208
    T b0 = 2.7012593e-2;
    T b1 = 2.0410827e-2;
    T b2 = 3.7451957e-3;
    T b3 = 4.7013839e-2;

    return b0 + b1 * std::sqrt(n) + b2 * n + b3 * std::sqrt(n * n * n);
}

template<class T>
constexpr inline T kernelSupport = 2.0;

//! @brief compute the 3D normalization constant for an arbitrary kernel
template<class F>
constexpr double kernel_3D_k(F sphKernel)
{
    auto kernelVol3D = [sphKernel = std::move(sphKernel)](double x) { return 4.0 * M_PI * x * x * sphKernel(x); };

    uint64_t numIntervals = 2000;
    return 1.0 / util::simpson(0, kernelSupport<double>, numIntervals, kernelVol3D);
}

template<class T>
struct SincN
{
    T n;

    constexpr T operator()(const T x) const
    {
        if (x >= kernelSupport<T>) return 0;
        const T w = wharmonic_std(x);
        return n == T(6) ? (w * w) * (w * w) * (w * w) : cstone::fastmath::pow(w, n);
    }

    constexpr T derivative(const T x) const
    {
        if (x >= kernelSupport<T>) return 0;
        const T w   = wharmonic_std(x);
        const T dw  = wharmonic_derivative_std(x);
        const T wn1 = n == T(6) ? (w * w) * (w * w) * w : cstone::fastmath::pow(w, n - 1);
        return n * wn1 * dw;
    }
};

//! @brief smoothing kernel as a linear combination of two sinc^n terms
template<class T>
struct SincN1SincN2
{
    constexpr T operator()(const T x) const
    {
        // Local sinc copies to work around NVCC bug (tested with 13.1)
        constexpr SincN<T> sincN1_ = sincN1;
        constexpr SincN<T> sincN2_ = sincN2;
        return a * K1 * sincN1_(x) + (1 - a) * K2 * sincN2_(x);
    }
    constexpr T derivative(const T x) const
    {
        // Local sinc copies to work around NVCC bug (tested with 13.1)
        constexpr SincN<T> sincN1_ = sincN1;
        constexpr SincN<T> sincN2_ = sincN2;
        return a * K1 * sincN1_.derivative(x) + (1 - a) * K2 * sincN2_.derivative(x);
    }

    static constexpr T        a      = 0.9;
    static constexpr T        n1     = 4.0;
    static constexpr T        n2     = 9.0;
    static constexpr SincN<T> sincN1 = SincN<T>{n1};
    static constexpr SincN<T> sincN2 = SincN<T>{n2};
    T                         K1     = kernel_3D_k(sincN1);
    T                         K2     = kernel_3D_k(sincN2);
};

template<class T>
struct WendlandC2
{
    constexpr T operator()(const T x) const
    {
        if (x >= kernelSupport<T>) return T(0);
        const T r    = T(0.5) * x;
        const T omr  = T(1) - r;
        const T omr2 = omr * omr;
        const T omr4 = omr2 * omr2;
        return omr4 * (T(1) + T(4) * r);
    }

    constexpr T derivative(const T x) const
    {
        if (x >= kernelSupport<T>) return T(0);
        const T r    = T(0.5) * x;
        const T omr  = T(1) - r;
        const T omr3 = omr * omr * omr;
        return T(-10) * r * omr3;
    };
};

template<class T>
struct WendlandC4
{
    constexpr T operator()(const T x) const
    {
        if (x >= kernelSupport<T>) return T(0);
        const T r    = T(0.5) * x;
        const T omr  = T(1) - r;
        const T omr2 = omr * omr;
        const T omr6 = omr2 * omr2 * omr2;
        return omr6 * (T(1) + T(6) * r + T(35.0 / 3.0) * r * r);
    }

    constexpr T derivative(const T x) const
    {
        if (x >= kernelSupport<T>) return T(0);
        const T r    = T(0.5) * x;
        const T omr  = T(1) - r;
        const T omr2 = omr * omr;
        const T omr5 = omr2 * omr2 * omr;
        return T(-56.0 / 6.0) * r * (T(5) * r + T(1)) * omr5;
    };
};

template<class T>
struct WendlandC6
{
    constexpr T operator()(const T x) const
    {
        if (x >= kernelSupport<T>) return T(0);
        const T r    = T(0.5) * x;
        const T r2   = r * r;
        const T omr  = T(1) - r;
        const T omr2 = omr * omr;
        const T omr4 = omr2 * omr2;
        const T omr8 = omr4 * omr4;
        return omr8 * (T(1) + T(8) * r + T(25) * r2 + T(32) * r2 * r);
    }

    constexpr T derivative(const T x) const
    {
        if (x >= kernelSupport<T>) return T(0);
        const T r    = T(0.5) * x;
        const T omr  = T(1) - r;
        const T omr2 = omr * omr;
        const T omr3 = omr2 * omr;
        const T omr7 = omr3 * omr3 * omr;
        return T(-11) * r * (T(16) * r * r + T(7) * r + T(1)) * omr7;
    };
};

template<class T>
using KernelVariant = std::variant<SincN<T>, SincN1SincN2<T>, WendlandC2<T>, WendlandC4<T>, WendlandC6<T>>;

enum SphKernelType : int
{
    sinc_n          = 0,
    sinc_n1_sinc_n2 = 1,
    wendland_c2     = 2,
    wendland_c4     = 3,
    wendland_c6     = 4,
};

/*! @brief return the SPH kernel as a function object
 *
 * If sinc_n is chosen, n will be set to @p sincIndex.
 * For sinc_n1_plus_sinc_n2, the linear combination and exponents are fixed here
 */
template<class T>
KernelVariant<T> getSphKernel(SphKernelType choice, T sincIndex)
{
    switch (choice)
    {
        case SphKernelType::sinc_n: return SincN<T>{sincIndex};
        case SphKernelType::sinc_n1_sinc_n2: return SincN1SincN2<T>{};
        case SphKernelType::wendland_c2: return WendlandC2<T>{};
        case SphKernelType::wendland_c4: return WendlandC4<T>{};
        case SphKernelType::wendland_c6: return WendlandC6<T>{};
        default: throw std::runtime_error("Invalid SPH kernel type");
    }
}

} // namespace sph
