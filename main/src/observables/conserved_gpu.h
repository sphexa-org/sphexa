/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich
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
 * @brief  Energy and momentum reductions on the GPU
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <tuple>

#include "cstone/tree/definitions.h"

namespace sphexa
{

/*! @brief compute conserved energies and momenta on the GPU
 *
 * @param[in] cv         heat capacity
 * @param[in] x          x-coordinates
 * @param[in] y          y-coordinates
 * @param[in] z          z-coordinates
 * @param[in] vx         x-velocities
 * @param[in] vy         v-velocities
 * @param[in] vz         z-velocities
 * @param[in] temp       temperatures, can be nullptr
 * @param[in] u          internal energies, can be nullptr
 * @param[in] m          masses
 * @param[in] first      first particle index to include in the sum
 * @param[in] last       last particle index to include in the sum
 * @return               A tuple with the total kinetic energy, internal energy, linear momentum and angular momentum
 */
template<class Tc, class Tv, class Tt, class Tm>
extern std::tuple<double, double, cstone::Vec3<double>, cstone::Vec3<double>>
conservedQuantitiesGpu(double cv, const Tc* x, const Tc* y, const Tc* z, const Tv* vx, const Tv* vy, const Tv* vz,
                       const Tt* temp, const Tt* u, const Tm* m, size_t first, size_t last);

/*! @brief compute the magnetic energy on the GPU
 *
 * @param mu_0  magnetic permeability
 * @param divB  divergence of B-field
 * @param h     smoothing length
 * @param Bx    Magnetic field x-component
 * @param By    Magnetic field y-component
 * @param Bz    Magnetic field z-component
 * @param first first particle index to include in the sum
 * @param last  last particle index to include in the sum
 * @return      total magnetic energy of the range [first, last]
 */
template<class Tc, class Th>
extern std::tuple<double, double, double> magneticEnergyGpu(Tc mu_0, const Th* xm, const Th* kx, const Th* divB,
                                                            const Th* h, const Tc* Bx, const Tc* By, const Tc* Bz,
                                                            size_t first, size_t last);

} // namespace sphexa
