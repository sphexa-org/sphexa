/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
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
 * @brief  SPH density kernel with various neighbor search strategies
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <format>
#include <limits>
#include <map>
#include <tuple>
#include <vector>

#include <thrust/universal_vector.h>

#include "cstone/cuda/thrust_util.cuh"
#include "cstone/cuda/cuda_runtime.hpp"
#include "cstone/traversal/ijloop/gpu_alwaystraverse.cuh"
#include "cstone/traversal/ijloop/gpu_clusternblist.cuh"
#include "cstone/traversal/ijloop/gpu_fullnblist.cuh"
#include "cstone/traversal/ijloop/gpu_superclusternblist.cuh"

#include "../coord_samples/random.hpp"
#include "./csv.hpp"
#include "./gromacs_ijloop.cuh"
#include "./nbbenchmark.cuh"

/* smoothing kernel evaluation functionality borrowed from SPH-EXA */

constexpr int kTableSize = 20000;

template<class T>
constexpr __forceinline__ T fastSin(T x)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if constexpr (std::is_same_v<T, float>)
        x = __sinf(x);
    else
#endif
        x = std::sin(x);
    return x;
}

template<class T>
constexpr __forceinline__ T fastInv(T x)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if constexpr (std::is_same_v<T, float>)
#if defined(__CUDA_ARCH__)
        asm("rcp.approx.ftz.f32 %0,%0;" : "+f"(x) :);
#else
        x = __frcp_rn(x);
#endif
    else if constexpr (std::is_same_v<T, double>)
#if defined(__CUDA_ARCH__)
        asm("rcp.approx.ftz.f64 %0,%0;" : "+d"(x) :);
#else
        x = __drcp_rn(x);
#endif
    else
#endif
        x = T(1) / x;
    return x;
}

template<class T>
constexpr __forceinline__ T fastSqrt(T x)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if constexpr (std::is_same_v<T, float>)
        x = __fsqrt_rn(x);
    else if constexpr (std::is_same_v<T, double>)
        x = __dsqrt_rn(x);
    else
#endif
        x = std::sqrt(x);
    return x;
}

template<typename T>
constexpr inline T wharmonic_std(T v)
{
    if (v == 0) { return 1; }

    const T Pv = T(M_PI_2) * v;
    return fastSin(Pv) * fastInv(Pv);
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
    checkGpuErrors(cudaMemPrefetchAsync(rawPtr(table), sizeof(T) * n, device, 0));

    return table;
}

template<class T>
auto kernelTable()
{
    return tabulateFunction<T>([](T x) { return std::pow(wharmonic_std(x), 6.0); }, 0.0, 2.0, kTableSize);
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
        T w  = wharmonic_std(v);
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
        const T dist                 = fastSqrt(distSq);
        const T vloc                 = dist * fastInv(hi);
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

    std::map<std::string, std::vector<double>> times;

    const auto runBenchmark = [&](const char* name, auto const& neighborhood)
    {
        printf("--- %s ---\n", name);
        times[name] = benchmarkNeighborhood<Tc, T, StrongKeyType>(coords, neighborhood, h, 1, ngmax, kernelFun,
                                                                  inputValues, initialOutputValues);
        printf("\n");
    };

    runBenchmark("DIRECT TREE TRAVERSAL", ijloop::GpuAlwaysTraverseNeighborhood{ngmax});
    runBenchmark("FULL NB LIST", ijloop::GpuFullNbListNeighborhood{ngmax});
    runBenchmark("GROMACS SUPERCLUSTERED", ijloop::GromacsLikeNeighborhood{ngmax});

    using BaseClusterNb = ijloop::GpuClusterNbListNeighborhood<>::withNcMax<192>::withClusterSize<4, 4>;
    runBenchmark("CLUSTERED", BaseClusterNb::withoutSymmetry::withoutCompression{});
    runBenchmark("COMPRESSED CLUSTERED", BaseClusterNb::withoutSymmetry::withCompression<9>{});

    using SymmetricClusterNb = BaseClusterNb::withNcMax<128>::withSymmetry;
    runBenchmark("CLUSTERED SYMMETRIC", SymmetricClusterNb::withoutCompression{});
    runBenchmark("COMPRESSED CLUSTERED", SymmetricClusterNb::withCompression<7>{});

    using BaseSuperclusterNb =
        ijloop::GpuSuperclusterNbListNeighborhood<>::withClusterSize<8, 8>::withSuperclusterSize<64>::withNcMax<1024>;
    runBenchmark("SUPERCLUSTERED", BaseSuperclusterNb::withoutSymmetry::withoutCompression{});
    runBenchmark("COMPRESSED SUPERCLUSTERED", BaseSuperclusterNb::withoutSymmetry::withCompression{});

    using SymmetricSuperclusterNb = BaseSuperclusterNb::withNcMax<512>::withSymmetry;
    runBenchmark("SUPERCLUSTERED SYMMETRIC", SymmetricSuperclusterNb::withoutCompression{});
    runBenchmark("COMPRESSED SUPERCLUSTERED SYMMETRIC", SymmetricSuperclusterNb::withCompression{});

    saveCsv(std::format("sph_density_results_{}_{}.csv", typeid(Tc).name(), typeid(T).name()), times);
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
