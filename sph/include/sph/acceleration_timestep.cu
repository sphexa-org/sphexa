//
// Created by Noah Kubli on 19.03.2025.
//
#include "acceleration_timestep_gpu.hpp"
#include "cstone/tree/definitions.h"
#include "cstone/util/array.hpp"

#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

namespace sph
{

template<typename T>
struct AccelerationTimestep
{
    __device__ T operator()(const thrust::tuple<T, T, T, T>& a_h)
    {
        const T               h_i = thrust::get<3>(a_h);
        const cstone::Vec3<T> A{thrust::get<0>(a_h), thrust::get<1>(a_h), thrust::get<2>(a_h)};
        return h_i * h_i / norm2(A);
    }
};

template<typename T>
T accelerationTimestepGPU(size_t first, size_t last, const T* ax, const T* ay, const T* az, const T* h)
{
    if (last <= first) { return INFINITY; }
    auto begin = thrust::make_zip_iterator(ax + first, ay + first, az + first, h + first);
    auto end   = thrust::make_zip_iterator(ax + last, ay + last, az + last, h + last);
    return thrust::transform_reduce(thrust::device, begin, end, AccelerationTimestep<T>{}, INFINITY,
                                    thrust::minimum<T>{});
}

#define ACCELERATION_TIMESTEP_GPU(T)                                                                                   \
    template T accelerationTimestepGPU(size_t first, size_t last, const T* ax, const T* ay, const T* az, const T* h);

ACCELERATION_TIMESTEP_GPU(double);
ACCELERATION_TIMESTEP_GPU(float);

} // namespace sph
