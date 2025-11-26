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
#include <vector>

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
double simpson(double a, double b, uint64_t n, F&& func)
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

//! @brief compute the 3D normalization constant for an arbitrary kernel
template<class F>
double kernel_3D_k(F&& sphKernel, double support)
{
    auto kernelVol3D = [sphKernel](double x) { return 4.0 * M_PI * x * x * sphKernel(x); };

    uint64_t numIntervals = 2000;
    return 1.0 / util::simpson(0, support, numIntervals, kernelVol3D);
}

//! @brief tabulate and arbitrary function at N points between lower support and upperSupport
template<typename T, std::size_t N, class F>
std::array<T, N> tabulateFunction(F&& func, double lowerSupport, double upperSupport)
{
    constexpr int    numIntervals = N - 1;
    std::array<T, N> table;

    const T dx = (upperSupport - lowerSupport) / numIntervals;
    for (size_t i = 0; i < N; ++i)
    {
        T normalizedVal = lowerSupport + i * dx;
        table[i]        = func(normalizedVal);
    }

    return table;
}

//! @brief derivative of sinc(Pi/2 * x)^sincIndex w.r.t. x
template<class T>
T powSincDerivative(T x, T sincIndex)
{
    return sincIndex * std::pow(wharmonic_std(x), sincIndex - 1) * wharmonic_derivative_std(x);
}

//! @brief smoothing kernel as a linear combination of two sinc^n terms
template<class T>
struct SincN1SincN2
{
    SincN1SincN2()
    {
        T support = 2.0;
        K1        = kernel_3D_k([this](auto x) { return std::pow(wharmonic_std(x), n1); }, support);
        K2        = kernel_3D_k([this](auto x) { return std::pow(wharmonic_std(x), n2); }, support);
    }

    T kernel(T x) const
    {
        return a * K1 * std::pow(wharmonic_std(x), n1) + (1 - a) * K2 * std::pow(wharmonic_std(x), n2);
    }
    T derivative(T x) const { return a * K1 * powSincDerivative(x, n1) + (1 - a) * K2 * powSincDerivative(x, n2); }

    T a  = 0.9;
    T n1 = 4.0;
    T n2 = 9.0;
    T K1, K2;
};

enum SphKernelType : int
{
    sinc_n          = 0,
    sinc_n1_sinc_n2 = 1,
};

/*! @brief return the SPH kernel as a function object
 *
 * If sinc_n is chosen, n will be set to @p sincIndex.
 * For sinc_n1_plus_sinc_n2, the linear combination and exponents are fixed here
 */
template<class T>
std::function<T(T)> getSphKernel(SphKernelType choice, T sincIndex)
{
    if (choice == SphKernelType::sinc_n)
    {
        return [sincIndex](T x) { return std::pow(wharmonic_std(x), sincIndex); };
    }
    else if (choice == SphKernelType::sinc_n1_sinc_n2)
    {
        auto kfunc = SincN1SincN2<T>{};
        return [f = kfunc](T x) { return f.kernel(x); };
    }
    return [](T x) { return x; };
}

template<class T>
std::function<T(T)> getSphKernelDerivative(SphKernelType choice, T sincIndex)
{
    if (choice == SphKernelType::sinc_n)
    {
        return [sincIndex](T x) { return powSincDerivative(x, sincIndex); };
    }
    else if (choice == SphKernelType::sinc_n1_sinc_n2)
    {
        auto kfunc = SincN1SincN2<T>{};
        return [f = kfunc](T x) { return f.derivative(x); };
    }
    return [](T x) { return x; };
}

} // namespace sph
