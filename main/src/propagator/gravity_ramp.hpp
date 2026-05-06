/*
 * MIT License
 *
 * Copyright (c) 2026 CSCS, ETH Zurich, University of Basel, University of Zurich
 */

#pragma once

#include <cmath>

namespace sphexa
{

template<class T>
T gravityRampFactor(T time, T rampStartTime, T rampTime)
{
    if (rampTime <= T(0) || rampStartTime < T(0)) { return T(1); }

    T tau = time - rampStartTime;
    if (tau <= T(0)) { return T(0); }
    if (tau >= rampTime) { return T(1); }

    constexpr T pi = T(3.141592653589793238462643383279502884);
    return T(0.5) * (T(1) - std::cos(pi * tau / rampTime));
}

template<class DataType>
class ScopedGravityRamp
{
    using T = typename DataType::RealType;

    typename DataType::HydroData& d_;
    T                            targetG_;

public:
    explicit ScopedGravityRamp(DataType& simData)
        : d_(simData.hydro)
        , targetG_(d_.g)
    {
        d_.g = targetG_ * gravityRampFactor(d_.ttot, d_.gravRampStartTime, d_.gravRampTime);
    }

    ~ScopedGravityRamp() { d_.g = targetG_; }

    ScopedGravityRamp(const ScopedGravityRamp&)            = delete;
    ScopedGravityRamp& operator=(const ScopedGravityRamp&) = delete;
};

} // namespace sphexa
