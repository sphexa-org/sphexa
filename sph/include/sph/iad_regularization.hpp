#pragma once

#include <limits>

#include "cstone/cuda/annotation.hpp"

namespace sph
{

template<class T>
HOST_DEVICE_FUN constexpr T iadMomentDet(T tau11, T tau12, T tau13, T tau22, T tau23, T tau33)
{
    return tau11 * tau22 * tau33 + T(2) * tau12 * tau23 * tau13 - tau11 * tau23 * tau23 -
           tau22 * tau13 * tau13 - tau33 * tau12 * tau12;
}

template<class T>
HOST_DEVICE_FUN constexpr T iadMomentSecondInvariant(T tau11, T tau12, T tau13, T tau22, T tau23, T tau33)
{
    return tau11 * tau22 + tau11 * tau33 + tau22 * tau33 - tau12 * tau12 - tau13 * tau13 - tau23 * tau23;
}

template<class T>
HOST_DEVICE_FUN constexpr T iadMomentQuality(T det, T trAvg)
{
    return (det > T(0) && trAvg > T(0)) ? det / (trAvg * trAvg * trAvg) : T(0);
}

template<class T>
HOST_DEVICE_FUN constexpr T iadRidgeQuality(T det, T secondInvariant, T trace, T trAvg, T lambda)
{
    T delta     = lambda * trAvg;
    T detRidge  = det + secondInvariant * delta + trace * delta * delta + delta * delta * delta;
    T trAvgRidge = trAvg + delta;
    return iadMomentQuality(detRidge, trAvgRidge);
}

template<class T>
HOST_DEVICE_FUN constexpr T regularizeIadMomentMatrix(T& tau11, T& tau12, T& tau13, T& tau22, T& tau23, T& tau33,
                                                      T conditionQualityTarget, bool* wasRegularized = nullptr)
{
    T trace = tau11 + tau22 + tau33;
    T trAvg = trace / T(3);
    T det = iadMomentDet(tau11, tau12, tau13, tau22, tau23, tau33);

    if (wasRegularized) { *wasRegularized = false; }

    if (!(conditionQualityTarget > T(0)) || !(trAvg > T(0))) { return det; }

    T quality = iadMomentQuality(det, trAvg);
    if (quality >= conditionQualityTarget) { return det; }

    T target = conditionQualityTarget;
    T maxTarget = T(1) - T(64) * std::numeric_limits<T>::epsilon();
    if (target > maxTarget) { target = maxTarget; }
    if (quality >= target) { return det; }

    T secondInvariant = iadMomentSecondInvariant(tau11, tau12, tau13, tau22, tau23, tau33);

    T a = det / (trAvg * trAvg * trAvg);
    T b = secondInvariant / (trAvg * trAvg);
    T c3 = T(1) - target;
    T c2 = T(3) * c3;
    T c1 = b - T(3) * target;
    T c0 = a - target;

    auto f = [&](T lam) { return c3 * lam * lam * lam + c2 * lam * lam + c1 * lam + c0; };

    T hi = T(1);
    for (int iter = 0; iter < 10; ++iter)
    {
        if (f(hi) >= T(0)) break;
        hi *= T(2);
    }

    for (int iter = 0; iter < 8; ++iter)
    {
        T fp = T(3) * c3 * hi * hi + T(2) * c2 * hi + c1;
        hi -= f(hi) / fp;
    }

    T delta = hi * trAvg;
    tau11 += delta;
    tau22 += delta;
    tau33 += delta;

    if (wasRegularized) { *wasRegularized = true; }
    return iadMomentDet(tau11, tau12, tau13, tau22, tau23, tau33);
}

} // namespace sph
