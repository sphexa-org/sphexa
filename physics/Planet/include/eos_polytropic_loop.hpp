#pragma once

#include <vector>

#include "cstone/util/tuple.hpp"

namespace disk
{
/*! @brief general polytropic equation of state. P = Kpoly * rho^exp_poly
 *
 * @param Kpoly polytropic constant
 * @param exp_poly polytropic exponent
 * @param gamma adiabatic exponent (used to calculate the sound speed)
 * @param rho SPH density
 */
template<typename T1, typename T2, typename T3, typename T4>
HOST_DEVICE_FUN auto polytropicEOS(T1 Kpoly, T2 exp_poly, T3 gamma, T4 rho)
{
    using Tc = std::common_type_t<T1, T2, T3, T4>;

    Tc p = Kpoly * std::pow(rho, exp_poly);
    Tc c = std::sqrt(gamma * p / rho);

    return util::tuple<Tc, Tc>{p, c};
}

} // namespace planet
