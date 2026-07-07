/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  SPH density kernel with various neighbor search strategies
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <format>
#include <limits>
#include <map>
#include <numbers>
#include <string>
#include <tuple>
#include <typeinfo>
#include <vector>

#include <thrust/universal_vector.h>

#include "cstone/cuda/cuda_runtime.hpp"
#include "cstone/cuda/thrust_util.cuh"
#include "cstone/traversal/ijloop/gpu_alwaystraverse.cuh"
#include "cstone/traversal/ijloop/gpu_superclusternblist.cuh"

#include "../coord_samples/random.hpp"
#include "./nbbenchmark.cuh"

/* some fast math functions to reach good performance at -O3 */

constexpr float fast_sin(float x)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __sinf(x);
#else
    return std::sin(x);
#endif
}

constexpr double fast_sin(double x) { return std::sin(x); }

constexpr float fast_rcp(float x)
{
#ifdef __CUDA_ARCH__
    // __frcp_rn might not flush to zero and thus can be significantly slower
    asm("rcp.approx.ftz.f32 %0,%0;" : "+f"(x) :);
    return x;
#elif defined(__HIP_DEVICE_COMPILE__)
    return __frcp_rn(x);
#else
    return 1.0f / x;
#endif
}

constexpr double fast_rcp(double x)
{
#ifdef __CUDA_ARCH__
    // __drcp_rn might not flush to zero and thus can be significantly slower
    asm("rcp.approx.ftz.f64 %0,%0;" : "+d"(x) :);
    return x;
#elif defined(__HIP_DEVICE_COMPILE__)
    return __drcp_rn(x);
#else
    return 1.0 / x;
#endif
}

constexpr float fast_sqrt(float x)
{
#if defined(__CUDA_ARCH__)
    // __fsqrt_rn might not flush to zero and thus can be significantly slower
    asm("sqrt.approx.ftz.f32 %0,%0;" : "+f"(x) :);
    return x;
#elif defined(__HIP_DEVICE_COMPILE__)
    return __fsqrt_rn(x);
#else
    return std::sqrt(x);
#endif
}

constexpr double fast_sqrt(double x)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __dsqrt_rn(x);
#else
    return std::sqrt(x);
#endif
}

/* smoothing kernel evaluation functionality borrowed from SPH-EXA */

constexpr int kTableSize = 20000;

template<typename T>
constexpr inline T wharmonicStd(T v)
{
    if (v == 0) { return 1; }

    constexpr T halfPi = std::numbers::pi_v<T> / T(2);
    const T Pv         = halfPi * v;
    return fast_sin(Pv) * fast_rcp(Pv);
}

template<class T, class F>
thrust::universal_vector<T>
tabulateFunction(F&& func, const double lowerSupport, const double upperSupport, const std::size_t n)
{
    thrust::universal_vector<T> table(n);

    const T dx = (upperSupport - lowerSupport) / (n - 1);
    for (size_t i = 0; i < n; ++i)
    {
        T normalizedVal = lowerSupport + i * dx;
        table[i]        = func(normalizedVal);
    }

    // required on AMD for decent performance
    int device;
    checkGpuErrors(cudaGetDevice(&device));
#if defined(__HIPCC__) && (HIP_VERSION_MAJOR < 7 || (HIP_VERSION_MAJOR == 7 && HIP_VERSION_MINOR == 0))
    checkGpuErrors(hipMemPrefetchAsync(cstone::rawPtr(table), sizeof(T) * n, device));
#else
    checkGpuErrors(cudaMemPrefetchAsync(cstone::rawPtr(table), sizeof(T) * n, {cudaMemLocationTypeDevice, device}, 0));
#endif

    return table;
}

template<class T>
auto kernelTable()
{
    return tabulateFunction<T>([](T x) { return std::pow(wharmonicStd(x), 6.0); }, 0.0, 2.0, kTableSize);
}

template<bool UseKernelTable, class T>
constexpr inline T table_lookup(const T* table, T v)
{
    if constexpr (UseKernelTable)
    {
        constexpr int numIntervals = kTableSize - 1;
        constexpr T support        = 2.0;
        constexpr T dx             = support / numIntervals;
        constexpr T invDx          = T(1) / dx;

        int idx = v * invDx;

        T derivative = (idx >= numIntervals) ? 0.0 : (table[idx + 1] - table[idx]) * invDx;
        return (idx >= numIntervals) ? 0.0 : table[idx] + derivative * (v - T(idx) * dx);
    }
    else
    {
        T w  = wharmonicStd(v);
        T w2 = w * w;
        return w2 * w2 * w2;
    }
}

template<bool UseKernelTable, class T>
struct DensityKernelFun
{
    const T* wh;

    template<class ParticleData, class Tc>
    constexpr auto operator()(ParticleData const& iData, ParticleData const& jData, cstone::Vec3<Tc>, T distSq) const
    {
        const auto [i, iPos, hi, mi] = iData;
        const auto [j, jPos, hj, mj] = jData;
        const T dist                 = fast_sqrt(distSq);
        const T vloc                 = dist * fast_rcp(hi);
        const T w                    = i == j ? T(1) : table_lookup<UseKernelTable>(wh, vloc);
        return std::make_tuple(cstone::ijloop::symmetric::even(w * mj));
    }
};

template<class Tc, class T, class StrongKeyType, bool UseKernelTable>
void benchmarkMain()
{
    using namespace cstone;

    constexpr unsigned ngmax = 256;

    constexpr unsigned scale = 10;
    constexpr unsigned n     = 100000 * scale;
    const T h                = 0.75 / 20 / std::cbrt(scale);

    RandomCoordinates<Tc, StrongKeyType> coords(n, {0, 1, BoundaryType::open});

    const auto wh = kernelTable<T>();
    const DensityKernelFun<UseKernelTable, T> kernelFun{rawPtr(wh)};
    const auto inputValues         = std::tuple(T(1));
    const auto initialOutputValues = std::tuple(std::numeric_limits<T>::quiet_NaN());

    const auto runBenchmark = [&](const char* name, auto const& neighborhood)
    {
        printf("--- %s ---\n", name);
        const auto result = benchmarkNeighborhood<Tc, T, StrongKeyType>(coords, neighborhood, h, 1, ngmax, kernelFun,
                                                                        inputValues, initialOutputValues);
        printf("\n");
    };

    runBenchmark("DIRECT TREE TRAVERSAL", ijloop::GpuAlwaysTraverseNeighborhoodBuilder{ngmax});

    using SuperclusterNb = ijloop::GpuSuperclusterNbListNeighborhoodBuilder<>::withClusterSize<
        8, GpuConfig::warpSize / 8>::withSuperclusterSize<64>::withoutSymmetry;
    constexpr unsigned ncmax = 360;
    runBenchmark("SUPERCLUSTERED", SuperclusterNb::withoutCompression{ncmax});
    runBenchmark("COMPRESSED SUPERCLUSTERED (Band et al. Compression)",
                 SuperclusterNb::withCompression<BandEtAlWarpCompression<false>>{ncmax});
    runBenchmark("COMPRESSED SUPERCLUSTERED (Nibble-based Compression)",
                 SuperclusterNb::withCompression<NibbleWarpCompression<false>>{ncmax});

    using SymmetricSuperclusterNb     = SuperclusterNb::withSymmetry;
    constexpr unsigned ncmaxSymmetric = 360;
    runBenchmark("SUPERCLUSTERED SYMMETRIC", SymmetricSuperclusterNb::withoutCompression{ncmaxSymmetric});
    runBenchmark("COMPRESSED SUPERCLUSTERED SYMMETRIC (Band et al. Compression)",
                 SuperclusterNb::withCompression<BandEtAlWarpCompression<false>>{ncmaxSymmetric});
    runBenchmark("COMPRESSED SUPERCLUSTERED SYMMETRIC (Nibble-based Compression)",
                 SuperclusterNb::withCompression<NibbleWarpCompression<false>>{ncmaxSymmetric});
}

int main()
{
    using StrongKeyType = cstone::HilbertKey<std::uint64_t>;

    printf("=== DOUBLE COORDINATES, DOUBLE VALUES, KERNEL TABLE ===\n\n");
    benchmarkMain<double, double, StrongKeyType, true>();

    printf("=== DOUBLE COORDINATES, DOUBLE VALUES, DIRECT KERNEL EVALUATION ===\n\n");
    benchmarkMain<double, double, StrongKeyType, false>();

    printf("=== DOUBLE COORDINATES, FLOAT VALUES, KERNEL TABLE ===\n\n");
    benchmarkMain<double, float, StrongKeyType, true>();

    printf("=== DOUBLE COORDINATES, FLOAT VALUES, DIRECT KERNEL EVALUATION ===\n\n");
    benchmarkMain<double, float, StrongKeyType, false>();

    return 0;
}
