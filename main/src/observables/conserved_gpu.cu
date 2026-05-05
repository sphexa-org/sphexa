/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich
 *               2022 University of Basel
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

#include <thrust/execution_policy.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#include "conserved_gpu.h"

namespace sphexa
{

using cstone::Vec3;
using thrust::get;

struct TuplePlus
{
    using type = thrust::tuple<double, Vec3<double>, Vec3<double>>;
    HOST_DEVICE_FUN type operator()(const type& a, const type& b) const
    {
        return type(get<0>(a) + get<0>(b), get<1>(a) + get<1>(b), get<2>(a) + get<2>(b));
    }
};

/*! @brief Functor to compute kinetic and internal energy and linear and angular momentum
 *
 * @tparam Tc   type of x,y,z coordinates
 * @tparam Tm   type of mass
 * @tparam Tv   type of velocities
 */
template<class Tc, class Tm, class Tv>
struct EMom
{
    /*! @brief compute energies and momenta for a single particle
     *
     * @param p   Tuple<x,y,z,m,vx,vy,vz,temp> with data for one particle
     * @return    Tuple<kinetic energy, internal energy, linear momentum, angular momentum>
     */
    HOST_DEVICE_FUN TuplePlus::type operator()(const thrust::tuple<Tc, Tc, Tc, Tm, Tv, Tv, Tv>& p)
    {
        Vec3<double> X{get<0>(p), get<1>(p), get<2>(p)};
        Vec3<double> V{get<4>(p), get<5>(p), get<6>(p)};
        Tm           m = get<3>(p);
        return {m * norm2(V), double(m) * V, double(m) * cross(X, V)};
    }
};

/*! @brief Functor to compute internal energy contribution for one particle
 *
 * @tparam Tm   type of mass
 * @tparam Tt   type of temperature/cv
 */
template<class Tm, class Tt>
struct ComputeInternalEnergy
{
    const Tm* m;
    const Tt* cv;
    const Tt* temp;
    size_t    first;

    __device__ double operator()(size_t i) const { return m[i + first] * cv[i + first] * temp[i + first]; }
};

template<class Tc, class Tv, class Tt, class Tm>
std::tuple<double, double, Vec3<double>, Vec3<double>>
conservedQuantitiesGpu(double cvIdealGas, const Tc* x, const Tc* y, const Tc* z, const Tv* vx, const Tv* vy,
                       const Tv* vz, const Tt* temp, const Tt* u, const Tt* cv, const Tm* m, size_t first, size_t last)
{
    auto it1 = thrust::make_zip_iterator(
        thrust::make_tuple(x + first, y + first, z + first, m + first, vx + first, vy + first, vz + first));
    auto it2 = thrust::make_zip_iterator(
        thrust::make_tuple(x + last, y + last, z + last, m + last, vx + last, vy + last, vz + last));

    //! apply EMom to each particle and reduce results into a single sum
    auto ret = thrust::transform_reduce(thrust::device, it1, it2, EMom<Tc, Tm, Tv>{}, TuplePlus::type{}, TuplePlus{});
    auto [eKin, linMom, angMom] = std::make_tuple(get<0>(ret), get<1>(ret), get<2>(ret));

    double eInt = 0.0;
    if (temp != nullptr)
    {
        if (cv != nullptr)
        {
            // elementwise product: eInt = sum_i( m[i] * cv[i] * temp[i] )
            auto n = last - first;
            eInt   = thrust::transform_reduce(
                thrust::device, thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(n),
                ComputeInternalEnergy<Tm, Tt>{m, cv, temp, first}, 0.0, thrust::plus<double>());
        }
        else { eInt = cvIdealGas * thrust::inner_product(thrust::device, m + first, m + last, temp + first, Tt(0.0)); }
    }
    else if (u != nullptr) { eInt = thrust::inner_product(thrust::device, m + first, m + last, u + first, Tt(0.0)); }

    return {0.5 * eKin, eInt, linMom, angMom};
}

#define CONSERVED_Q_GPU(Tc, Tv, Tt, Tm)                                                                                \
    template std::tuple<double, double, Vec3<double>, Vec3<double>> conservedQuantitiesGpu(                            \
        double cvIdealGas, const Tc* x, const Tc* y, const Tc* z, const Tv* vx, const Tv* vy, const Tv* vz,            \
        const Tt* temp, const Tt* u, const Tt* cv, const Tm* m, size_t, size_t)

CONSERVED_Q_GPU(double, double, double, double);
CONSERVED_Q_GPU(double, double, double, float);
CONSERVED_Q_GPU(double, float, double, float);
CONSERVED_Q_GPU(float, float, float, float);

} // namespace sphexa

