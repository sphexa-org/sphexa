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

#include <cmath>

#include "gtest/gtest.h"

#include "sph/kernels.hpp"

using namespace cstone;
using namespace sph;

using float128 = _Float128;

constexpr float128 pi128 = 3.1415926535897932384626433832795028841971693993751f128;

inline float128 fabs128(float128 x) { return fabsf128(x); }
inline float128 sin128(float128 x) { return sinf128(x); }
inline float128 cos128(float128 x) { return cosf128(x); }
inline float128 pow128(float128 x, float128 y) { return powf128(x, y); }

template<class T, class FRef, class F>
void checkErrors(FRef fRef, F f, double absTol, double relTol)
{
    constexpr float128 startExp  = -40;
    constexpr float128 endExp    = 1;
    constexpr int      n         = 10000;
    double             maxAbsErr = 0;
    double             maxRelErr = 0;
    for (int i = 0; i < n; ++i)
    {
        const float128 x = pow128(float128(2), startExp + (endExp - startExp) * i / n);

        const float128 yRef = fRef(x);
        const T        y    = f(T(x));

        const float128 absErr = fabs128(float128(y) - yRef);
        const float128 relErr = absErr / fabs128(yRef);
        maxAbsErr             = std::max(maxAbsErr, double(absErr));
        maxRelErr             = std::max(maxRelErr, double(relErr));
    }
    EXPECT_LE(maxAbsErr, absTol);
    EXPECT_LE(maxRelErr, relTol);
}

TEST(Kernels, WharmonicStd)
{
    const auto fRef = [](float128 x) { return sin128(x * pi128 / 2) / (x * pi128 / 2); };
    const auto f    = [](auto x) { return wharmonic_std(x); };
    checkErrors<double>(fRef, f, 1e-15, 1e-13);
    checkErrors<float>(fRef, f, 1e-6, 1e-4);
}

TEST(Kernels, WharmonicDerivativeStd)
{
    const auto fRef = [](float128 x)
    { return cos128(x * pi128 / 2) / x - 2 * sin128(x * pi128 / 2) / (pi128 * x * x); };
    const auto f = [](auto x) { return wharmonic_derivative_std(x); };
    checkErrors<double>(fRef, f, 1e-14, 1e-9);
    checkErrors<float>(fRef, f, 1e-6, 1e-5);
}
