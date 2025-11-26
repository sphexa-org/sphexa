#pragma once

#include "cstone/cuda/annotation.hpp"
#include "cstone/util/array.hpp"

namespace sph
{

//! @brief compute time-step based on the signal velocity
template<class T1, class T2, class T3>
HOST_DEVICE_FUN inline auto tsKCourant(T1 maxvsignal, T2 h, T3 c, float Kcour)
{
    using T = std::common_type_t<T1, T2, T3>;
    T v     = maxvsignal > T(0) ? maxvsignal : c;
    return T(Kcour * h / v);
}

/*! @brief estimate updated smoothing length to bring the neighbor count closer to ng0
 *
 * @tparam T    float or double
 * @param ng0   target neighbor count
 * @param nc    current neighbor count
 * @param h     current smoothing length
 * @return      updated smoothing length
 */
template<class T>
HOST_DEVICE_FUN T updateH(unsigned ng0, unsigned nc, T h)
{
    constexpr T c0  = 1023.0;
    constexpr T exp = 1.0 / 10.0;
    return h * T(0.5) * std::pow(T(1) + c0 * ng0 / T(nc), exp);
}

//! @brief sinc(PI/2 * v)
template<typename T>
HOST_DEVICE_FUN inline T wharmonic_std(T v)
{
    if (v == 0.0) { return 1.0; }

    const T Pv = M_PI_2 * v;
    return std::sin(Pv) / Pv;
}

/*! @brief Derivative of sinc(PI/2 * v) w.r to v
 *
 * Unoptimized for clarity as this is only used to construct look-up tables
 */
template<typename T>
HOST_DEVICE_FUN inline T wharmonic_derivative_std(T v)
{
    if (v == 0.0) return 0.0;

    constexpr T piHalf = M_PI_2;
    const T     Pv     = piHalf * v;
    const T     sincv  = std::sin(Pv) / (Pv);

    return sincv * piHalf * ((std::cos(Pv) / std::sin(Pv)) - T(1) / Pv);
}

/*! @brief calculate the artificial viscosity between a pair of two particles
 *
 * @tparam T      float or double
 * @param alpha_i viscosity switch of particle i
 * @param alpha_j viscosity switch of particle j
 * @param c_i     speed of sound particle i
 * @param c_j     speed of sound particle j
 * @param w_ij    relative velocity (v_i - v_j), projected onto the connecting axis (r_i - r_j)
 * @return        the viscosity
 */
template<typename T>
HOST_DEVICE_FUN inline T artificial_viscosity(T alpha_i, T alpha_j, T c_i, T c_j, T w_ij)
{
    // alpha is const for now, but will be different for each particle when using viscosity switching
    constexpr T beta = T(2.0);

    T viscosity_ij = T(0.0);
    if (w_ij < T(0.0))
    {
        T vij_signal = (alpha_i + alpha_j) * T(0.25) * (c_i + c_j) - beta * w_ij;
        viscosity_ij = -vij_signal * w_ij;
    }

    return viscosity_ij;
}

//! @brief symmetric 3x3 matrix-vector product
template<class Tv, class Tm>
HOST_DEVICE_FUN HOST_DEVICE_INLINE util::array<Tv, 3> symv(const util::array<Tm, 6>& mat, const util::array<Tv, 3>& vec)
{
    util::array<Tv, 3> ret;
    ret[0] = mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2];
    ret[1] = mat[3] * vec[1] + mat[4] * vec[2];
    ret[2] = mat[5] * vec[2];
    return ret;
}

} // namespace sph
