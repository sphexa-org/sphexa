#pragma once

#include "cstone/util/tuple.hpp"

#include "kernels.hpp"

namespace sph
{

enum EosType : int
{
    idealGas   = 0,
    isothermal = 1,
    polytropic = 2
};

//! @brief returns the heat capacity for given mean molecular weight
template<class T1, class T2>
HOST_DEVICE_FUN constexpr T1 idealGasCv(T1 mui, T2 gamma)
{
    constexpr T1 R = 8.317e7;
    return R / mui / (gamma - T1(1));
}

/*! @brief Reduced version of Ideal gas EOS for internal energy
 *
 * @param u     internal energy
 * @param rho   baryonic density
 * @param gamma adiabatic index
 *
 * This EOS is used for simple cases where we don't need the temperature.
 * Returns pressure, speed of sound
 */
template<class T1, class T2, class T3>
HOST_DEVICE_FUN auto idealGasEOS_u(T1 u, T2 rho, T3 gamma)
{
    using Tc = std::common_type_t<T1, T2, T3>;

    Tc tmp = u * (gamma - Tc(1));
    Tc p   = rho * tmp;
    Tc c   = std::sqrt(gamma * tmp);

    return util::tuple<Tc, Tc>{p, c};
}

template<class T1, class T2, class T3>
HOST_DEVICE_FUN auto idealGasEOS(T1 temp, T2 rho, T3 mui, T1 gamma)
{
    return idealGasEOS_u(idealGasCv(mui, gamma) * temp, rho, gamma);
}

/*! @brief Isothermal equation of state
 *
 * @param c     speed of sound
 * @param rho   baryonic density
 *
 */
template<typename T1, typename T2>
HOST_DEVICE_FUN auto isothermalEOS(T1 c, T2 rho)
{
    using Tc = std::common_type_t<T1, T2>;
    Tc p     = rho * c * c;
    return p;
}

/*! @brief General polytropic equation of state.
 * @param K_poly       polytropic constant
 * @param gamma_poly   polytropic exponent
 * @param rho          SPH density
 *
 * Returns pressure and sound speed
 *
 * For a 1.4 M_sun and 12.8 km neutron star the values are
 * K_poly = 2.246341237993810232e-10
 * gammapol = 3.e0
 */
template<typename T1, typename T2, typename T3>
HOST_DEVICE_FUN auto polytropicEOS(T1 K_poly, T2 gamma_poly, T3 rho)
{
    using Tc = std::common_type_t<T1, T2, T3>;

    Tc p = K_poly * std::pow(rho, gamma_poly);
    Tc c = std::sqrt(gamma_poly * p / rho);

    return util::tuple<Tc, Tc>{p, c};
}

} // namespace sph
