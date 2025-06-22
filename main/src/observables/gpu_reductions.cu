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
 * @brief reductions for different observables on the GPU
 *
 * @author Lukas Schmidt
 */

#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/remove.h>

#include "cstone/util/tuple.hpp"
#include "gpu_reductions.h"
#include "sph/positions.hpp"

namespace sphexa
{

using cstone::Box;
using cstone::Vec3;
using thrust::get;

//!@brief functor for Kelvin-Helmholtz growth rate
template<class Tc, class Tv, class T>
struct GrowthRate
{
    HOST_DEVICE_FUN
    thrust::tuple<double, double, double> operator()(const thrust::tuple<Tc, Tc, Tv, T, T>& p)
    {
        auto [xi, yi, vyi, xmi, kxi] = p;
        auto voli                    = xmi / kxi;
        Tc   aux;

        if (yi < ybox * Tc(0.5)) { aux = std::exp(-4.0 * M_PI * std::abs(yi - 0.25)); }
        else { aux = std::exp(-4.0 * M_PI * std::abs(ybox - yi - 0.25)); }
        Tc si = vyi * voli * std::sin(4.0 * M_PI * xi) * aux;
        Tc ci = vyi * voli * std::cos(4.0 * M_PI * xi) * aux;
        Tc di = voli * aux;

        return thrust::make_tuple(si, ci, di);
    }

    Tc ybox;
};

template<class Tc, class Tv, class T>
std::tuple<double, double, double> gpuGrowthRate(const Tc* x, const Tc* y, const Tv* vy, const T* xm, const T* kx,
                                                 const cstone::Box<Tc>& box, size_t startIndex, size_t endIndex)
{
    auto it1 = thrust::make_zip_iterator(
        thrust::make_tuple(x + startIndex, y + startIndex, vy + startIndex, xm + startIndex, kx + startIndex));

    auto it2 = thrust::make_zip_iterator(
        thrust::make_tuple(x + endIndex, y + endIndex, vy + endIndex, xm + endIndex, kx + endIndex));

    auto plus = util::TuplePlus<thrust::tuple<double, double, double>>{};
    auto init = thrust::make_tuple(0.0, 0.0, 0.0);

    auto result = thrust::transform_reduce(thrust::device, it1, it2, GrowthRate<Tc, T, Tv>{box.ly()}, init, plus);

    return {get<0>(result), get<1>(result), get<2>(result)};
}

#define GROWTH_GPU(Tc, Tv, T)                                                                                          \
    template std::tuple<double, double, double> gpuGrowthRate(const Tc* x, const Tc* y, const Tv* vy, const T* xm,     \
                                                              const T* kx, const cstone::Box<Tc>& box,                 \
                                                              size_t startIndex, size_t endIndex)

GROWTH_GPU(double, double, double);
GROWTH_GPU(double, double, float);
GROWTH_GPU(double, float, float);
GROWTH_GPU(float, float, float);

//!@brief functor for the machSquare summing
template<class Tv, class T>
struct MachSquareSum
{
    HOST_DEVICE_FUN double operator()(const thrust::tuple<Tv, Tv, Tv, T>& p)
    {
        Vec3<Tv> V{get<0>(p), get<1>(p), get<2>(p)};
        auto     c = get<3>(p);
        return norm2(V) / (c * c);
    }
};

template<class Tv, class T>
double machSquareSumGpu(const Tv* vx, const Tv* vy, const Tv* vz, const T* c, size_t first, size_t last)
{
    auto it1 = thrust::make_zip_iterator(thrust::make_tuple(vx + first, vy + first, vz + first, c + first));
    auto it2 = thrust::make_zip_iterator(thrust::make_tuple(vx + last, vy + last, vz + last, c + last));

    auto   plus               = thrust::plus<double>{};
    double localMachSquareSum = 0.0;

    localMachSquareSum =
        thrust::transform_reduce(thrust::device, it1, it2, MachSquareSum<Tv, T>{}, localMachSquareSum, plus);

    return localMachSquareSum;
}

#define MACH_GPU(Tv, T)                                                                                                \
    template double machSquareSumGpu(const Tv* vx, const Tv* vy, const Tv* vz, const T* c, size_t, size_t)

MACH_GPU(double, double);
MACH_GPU(double, float);
MACH_GPU(float, float);

//!@brief functor to count particles that belong to the cloud
template<class T, class Tt, class Tm>
struct Survivors
{
    HOST_DEVICE_FUN
    double operator()(const thrust::tuple<Tt, T, T, Tm>& p)
    {
        auto temp = get<0>(p);
        auto kx   = get<1>(p);
        auto xm   = get<2>(p);
        auto m    = get<3>(p);
        auto rhoi = kx / xm * m;
        if (rhoi >= 0.64 * rhoBubble && temp <= 0.9 * tempWind) { return m; }
        else { return 0; }
    }

    double rhoBubble;
    double tempWind;
};

template<class T, class Tt, class Tm>
double survivingMassGpu(const Tt* temp, const T* kx, const T* xmass, const Tm* m, double rhoBubble, double tempWind,
                        size_t first, size_t last)
{
    auto it1 = thrust::make_zip_iterator(thrust::make_tuple(temp + first, kx + first, xmass + first, m + first));
    auto it2 = thrust::make_zip_iterator(thrust::make_tuple(temp + last, kx + last, xmass + last, m + last));

    return thrust::transform_reduce(thrust::device, it1, it2, Survivors<T, Tt, Tm>{rhoBubble, tempWind}, 0.0,
                                    thrust::plus{});
}

#define SURVIVORS(T, Tt, Tm)                                                                                           \
    template double survivingMassGpu(const Tt* temp, const T* kx, const T* xmass, const Tm* m, double, double, size_t, \
                                     size_t)

SURVIVORS(double, double, double);
SURVIVORS(float, float, float);
SURVIVORS(float, double, float);

//! @brief functor to gather valid position/velocity pairs for Rayleigh-Taylor observable
template<class T, class Tc, class Tm>
struct MarkRampCond
{
    HOST_DEVICE_FUN
    void operator()(thrust::tuple<T, T, T, Tm, AuxT<T>&, AuxT<T>&> p)
    {
        T    h               = get<0>(p);
        T    y               = get<1>(p);
        T    vy              = get<2>(p);
        Tm   markRamp        = get<3>(p);
        bool closeToBoundary = thrust::min(std::abs(y - ymin), std::abs(y - ymax)) / h < 8.;
        if (markRamp > 0.05 && !closeToBoundary)
        {
            thrust::get<4>(p) = {y, vy};
            thrust::get<5>(p) = {y, vy};
        }
    }
    Tc ymin;
    Tc ymax;
};

//! @brief calculates the Rayleigh-Taylor observable on GPUs
template<class T, class Tc, class Th>
std::tuple<std::vector<AuxT<T>>, std::vector<AuxT<T>>> localGrowthRateRTGpu(size_t first, size_t last, Tc ymin, Tc ymax,
                                                                            const Th* h, const T* y, const Th* vy,
                                                                            const Th* markRamp)
{
    thrust::device_vector<AuxT<T>> targetUp(last - first);
    thrust::device_vector<AuxT<T>> targetDown(last - first);

    auto it1 = thrust::make_zip_iterator(
        thrust::make_tuple(h + first, y + first, vy + first, markRamp + first, targetUp.begin(), targetDown.begin()));
    auto it2 = thrust::make_zip_iterator(
        thrust::make_tuple(h + last, y + last, vy + last, markRamp + last, targetUp.end(), targetDown.end()));

    thrust::for_each(thrust::device, it1, it2, MarkRampCond<T, Tc, Th>{ymin, ymax});

    auto endUp   = thrust::remove_if(thrust::device, targetUp.begin(), targetUp.end(), invalidAuxTEntry<T>());
    auto endDown = thrust::remove_if(thrust::device, targetDown.begin(), targetDown.end(), invalidAuxTEntry<T>());

    thrust::sort(thrust::device, targetUp.begin(), endUp, greaterRT());
    thrust::sort(thrust::device, targetDown.begin(), endDown, lowerRT());

    targetUp.resize(RT_N_AVG);
    targetDown.resize(RT_N_AVG);

    std::vector<AuxT<T>> retUp(RT_N_AVG);
    std::vector<AuxT<T>> retDown(RT_N_AVG);
    thrust::copy(targetUp.begin(), targetUp.end(), retUp.begin());
    thrust::copy(targetDown.begin(), targetDown.end(), retDown.begin());

    return std::make_tuple(retUp, retDown);
}

#define RTGROWTH(T, Tc, Th)                                                                                            \
    template std::tuple<std::vector<AuxT<T>>, std::vector<AuxT<T>>> localGrowthRateRTGpu(                              \
        size_t first, size_t last, Tc ymin, Tc ymax, const Th* h, const T* y, const Th* vy, const Th* markRamp);

RTGROWTH(double, double, double);
RTGROWTH(double, float, double);
RTGROWTH(double, double, float);
RTGROWTH(double, float, float);
RTGROWTH(float, float, float);

} // namespace sphexa
