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
#include <limits>
#include <tuple>

#include <thrust/universal_vector.h>

#include "cstone/cuda/thrust_util.cuh"
#include "cstone/traversal/ijloop/gpu_alwaystraverse.cuh"
#include "cstone/traversal/ijloop/gpu_clusternblist.cuh"
#include "cstone/traversal/ijloop/gpu_fullnblist.cuh"
#include "cstone/traversal/ijloop/gpu_superclusternblist.cuh"

#include "../coord_samples/random.hpp"
#include "./gromacs_ijloop.cuh"
#include "./nbbenchmark.cuh"

/* smoothing kernel evaluation functionality borrowed from SPH-EXA */

constexpr int kTableSize = 20000;

template<typename T>
constexpr inline T wharmonic_std(T v)
{
    if (v == 0.0) { return 1.0; }

    const T Pv = T(M_PI_2) * v;
    return std::sin(Pv) / Pv;
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
        const T dist                 = std::sqrt(distSq);
        const T vloc                 = dist * (T(1) / hi);
        const T w                    = i == j ? T(1) : table_lookup<UseKernelTable>(wh, vloc);
        return std::make_tuple(w * mj);
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

    RandomCoordinates<Tc, StrongKeyType> coords(n, {0, 1, BoundaryType::periodic});

    const auto wh = kernelTable<T>();
    const DensityKernelFun<UseKernelTable, T> kernelFun{rawPtr(wh)};
    const auto inputValues         = std::tuple(T(1));
    const auto initialOutputValues = std::tuple(std::numeric_limits<T>::quiet_NaN());

    const auto runBenchmark = [&](const char* name, auto const& neighborhood)
    {
        printf("--- %s ---\n", name);
        benchmarkNeighborhood<Tc, T, StrongKeyType>(coords, neighborhood, h, ngmax, kernelFun, ijloop::symmetry::even,
                                                    inputValues, initialOutputValues);
        printf("\n");
    };

    runBenchmark("BATCHED DIRECT", ijloop::GpuAlwaysTraverseNeighborhood{ngmax});
    runBenchmark("NAIVE TWO-STAGE", ijloop::GpuFullNbListNeighborhood{ngmax});
    runBenchmark("GROMACS CLUSTERED TWO-STAGE", ijloop::GromacsLikeNeighborhood{ngmax});

    using BaseClusterNb = ijloop::GpuClusterNbListNeighborhood<>::withNcMax<160>::withClusterSize<4, 4>;
    runBenchmark("CLUSTERED TWO-STAGE", BaseClusterNb::withoutSymmetry::withoutCompression{});
    runBenchmark("COMPRESSED CLUSTERED TWO-STAGE", BaseClusterNb::withoutSymmetry::withCompression<9>{});

    using SymmetricClusterNb = BaseClusterNb::withNcMax<96>::withSymmetry;
    runBenchmark("CLUSTERED TWO-STAGE SYMMETRIC", SymmetricClusterNb::withoutCompression{});
    runBenchmark("COMPRESSED CLUSTERED TWO-STAGE ", SymmetricClusterNb::withCompression<7>{});

    using BaseSuperclusterNb =
        ijloop::GpuSuperclusterNbListNeighborhood<>::withClusterSize<8, 8>::withSuperclusterSize<64>::withNcMax<1024>;
    runBenchmark("SUPERCLUSTERED TWO-STAGE", BaseSuperclusterNb::withoutSymmetry::withoutCompression{});
    runBenchmark("COMPRESSED SUPERCLUSTERED TWO-STAGE", BaseSuperclusterNb::withoutSymmetry::withCompression{});

    using SymmetricSuperclusterNb = BaseSuperclusterNb::withNcMax<512>::withSymmetry;
    runBenchmark("SUPERCLUSTERED TWO-STAGE SYMMETRIC", SymmetricSuperclusterNb::withoutCompression{});
    runBenchmark("COMPRESSED SUPERCLUSTERED TWO-STAGE SYMMETRIC", SymmetricSuperclusterNb::withCompression{});
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
