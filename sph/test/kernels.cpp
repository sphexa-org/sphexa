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
 * @brief SPH basic kernel tests
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#ifdef HAVE_QUADMATH

#include <cmath>

#include <quadmath.h>

#include "gtest/gtest.h"

#include "sph/kernels.hpp"

using namespace cstone;
using namespace sph;

using reffloat_t = _Float128;

template<class T, class FRef, class F>
void checkErrors(FRef fRef, F f, double absTol, double relTol)
{
    constexpr reffloat_t startExp  = -40;
    constexpr reffloat_t endExp    = 2;
    constexpr int        n         = 1000;
    double               maxAbsErr = 0;
    double               maxRelErr = 0;
    for (int i = 0; i < n; ++i)
    {
        const reffloat_t x = powq(reffloat_t(2), startExp + (endExp - startExp) * i / n);

        const reffloat_t yRef = fRef(x);
        const T          y    = f(T(x));

        const reffloat_t absErr = fabsq(reffloat_t(y) - yRef);
        const reffloat_t relErr = absErr / fabsq(yRef);
        maxAbsErr               = std::max(maxAbsErr, double(absErr));
        maxRelErr               = std::max(maxRelErr, double(relErr));
    }
    EXPECT_LE(maxAbsErr, absTol);
    EXPECT_LE(maxRelErr, relTol);
}

TEST(Kernels, WharmonicStd)
{
    const auto fRef = [](reffloat_t x) { return sinq(x * M_PI_2q) / (x * M_PI_2q); };
    const auto f    = [](auto x) { return wharmonic_std(x); };
    checkErrors<double>(fRef, f, 1e-15, 1e-14);
    checkErrors<float>(fRef, f, 1e-6, 1e-4);
}

TEST(Kernels, WharmonicDerivativeStd)
{
    const auto fRef = [](reffloat_t x) { return cosq(x * M_PI_2q) / x - 2 * sinq(x * M_PI_2q) / (M_PIq * x * x); };
    const auto f    = [](double x) { return wharmonic_derivative_std(x); };
    checkErrors<double>(fRef, f, 1e-14, 1e-9);
    checkErrors<float>(fRef, f, 1e-7, 1e-5);
}

#endif
