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
 * @brief  Lennard-Jones kernel with various neighbor search strategies
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#include <cstdint>
#include <cstdio>
#include <limits>
#include <tuple>

#include "cstone/traversal/ijloop/gpu_alwaystraverse.cuh"
#include "cstone/traversal/ijloop/gpu_clusternblist.cuh"
#include "cstone/traversal/ijloop/gpu_fullnblist.cuh"
#include "cstone/traversal/ijloop/gpu_superclusternblist.cuh"

#include "../coord_samples/face_centered_cubic.hpp"
#include "./gromacs_ijloop.cuh"
#include "./nbbenchmark.cuh"

template<class T>
struct LjKernelFun
{
    T lj1, lj2;

    template<class ParticleData, class Tc>
    constexpr __host__ __device__ auto
    operator()(ParticleData const& iData, ParticleData const& jData, cstone::Vec3<Tc> ijPosDiff, T distSq) const
    {
        const auto [i, iPos, hi] = iData;
        const auto [j, jPos, hj] = jData;
        const T r2inv            = T(1) / distSq;
        const T r6inv            = r2inv * r2inv * r2inv;
        const T forcelj          = r6inv * (lj1 * r6inv - lj2);
        const T fpair            = i == j ? 0 : forcelj * r2inv;
        return std::make_tuple(T(ijPosDiff[0]) * fpair, T(ijPosDiff[1]) * fpair, T(ijPosDiff[2]) * fpair);
    }
};

template<class Tc, class T, class StrongKeyType>
void benchmarkMain()
{
    using namespace cstone;

    constexpr unsigned ngmax = 224;

    constexpr unsigned nx = 200;
    constexpr T h         = 1.9;

    FaceCenteredCubicCoordinates<Tc, StrongKeyType> coords(nx, nx, nx, {0, 1.6795962 * nx, BoundaryType::periodic});

    constexpr LjKernelFun<T> kernelFun{T(48), T(24)};
    constexpr auto inputValues         = std::tuple();
    constexpr auto initialOutputValues = std::tuple(
        std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN());

    const auto runBenchmark = [&](const char* name, auto const& neighborhood)
    {
        printf("--- %s ---\n", name);
        benchmarkNeighborhood<Tc, T, StrongKeyType>(coords, neighborhood, h, ngmax, kernelFun, ijloop::symmetry::odd,
                                                    inputValues, initialOutputValues);
        printf("\n");
    };

    runBenchmark("BATCHED DIRECT", ijloop::GpuAlwaysTraverseNeighborhood{ngmax});
    runBenchmark("NAIVE TWO-STAGE", ijloop::GpuFullNbListNeighborhood{ngmax});
    runBenchmark("GROMACS CLUSTERED TWO-STAGE", ijloop::GromacsLikeNeighborhood{ngmax});

    using BaseClusterNb = ijloop::GpuClusterNbListNeighborhood<>::withNcMax<160>::withClusterSize<4, 4>;
    runBenchmark("CLUSTERED TWO-STAGE", BaseClusterNb::withoutSymmetry::withoutCompression{});
    runBenchmark("COMPRESSED CLUSTERED TWO-STAGE", BaseClusterNb::withoutSymmetry::withCompression<8>{});

    using SymmetricClusterNb = BaseClusterNb::withNcMax<128>::withSymmetry;
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

    printf("=== DOUBLE COORDINATES, DOUBLE VALUES ===\n\n");
    benchmarkMain<double, double, StrongKeyType>();

    printf("=== DOUBLE COORDINATES, FLOAT VALUES ===\n\n");
    benchmarkMain<double, float, StrongKeyType>();

    printf("=== FLOAT COORDINATES, FLOAT VALUES ===\n\n");
    benchmarkMain<float, float, StrongKeyType>();

    return 0;
}
