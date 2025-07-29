/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Lennard-Jones kernel with various neighbor search strategies
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <format>
#include <limits>
#include <map>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include "cstone/traversal/ijloop/gpu_alwaystraverse.cuh"
#include "cstone/traversal/ijloop/gpu_fullnblist.cuh"
#include "cstone/traversal/ijloop/gpu_superclusternblist.cuh"
#include "cstone/util/fastmath.hpp"

#include "../coord_samples/face_centered_cubic.hpp"
#include "./csv.hpp"
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
        using namespace cstone::ijloop;
        const auto [i, iPos, hi, qi] = iData;
        const auto [j, jPos, hj, qj] = jData;
        const T r2                   = std::max(distSq, T(1e-1));
        const T rinv                 = util::fastmath::rsqrt(r2);
        const T r2inv                = rinv * rinv;
        const T r6inv                = r2inv * r2inv * r2inv;
        const T forcelj              = r6inv * (lj1 * r6inv - lj2) * r2inv;
        const T forcecoul            = qi * qj * r2inv * rinv;
        const T fpair                = i == j ? 0 : forcelj + forcecoul;
        return std::make_tuple(symmetric::odd(T(ijPosDiff[0]) * fpair), symmetric::odd(T(ijPosDiff[1]) * fpair),
                               symmetric::odd(T(ijPosDiff[2]) * fpair));
    }
};

template<class Tc, class T, class StrongKeyType>
void benchmarkMain()
{
    using namespace cstone;

    constexpr unsigned ngmax = 320;

    constexpr unsigned nx           = 100;
    constexpr T h                   = 1.75;
    constexpr float searchExtFactor = 1.9 / h;

    FaceCenteredCubicCoordinates<Tc, StrongKeyType> coords(nx, nx, nx, {0, 1.6795962 * nx, BoundaryType::open});

    constexpr LjKernelFun<T> kernelFun{T(48), T(24)};
    constexpr auto inputValues         = std::tuple(T(12));
    constexpr auto initialOutputValues = std::tuple(
        std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN());

    std::map<std::string, std::vector<double>> times;

    const auto runBenchmark = [&](const char* name, auto const& neighborhood)
    {
        printf("--- %s ---\n", name);
        times[name] =
            benchmarkNeighborhood<Tc, T, StrongKeyType>(coords, neighborhood, h, searchExtFactor, ngmax, kernelFun,
                                                        inputValues, initialOutputValues, std::is_same_v<T, double>);
        printf("\n");
    };

    runBenchmark("DIRECT TREE TRAVERSAL", ijloop::GpuAlwaysTraverseNeighborhood{ngmax});
    runBenchmark("FULL NB LIST", ijloop::GpuFullNbListNeighborhood{ngmax});
    runBenchmark("GROMACS SUPERCLUSTERED", ijloop::GromacsLikeNeighborhood{ngmax});

    using BaseSuperclusterNb =
        ijloop::GpuSuperclusterNbListNeighborhood<>::withClusterSize<8, 8>::withSuperclusterSize<64>;
    runBenchmark("SUPERCLUSTERED", BaseSuperclusterNb::withoutSymmetry::withoutCompression{512});
    runBenchmark("COMPRESSED SUPERCLUSTERED", BaseSuperclusterNb::withoutSymmetry::withCompression{512});

    using SymmetricSuperclusterNb = BaseSuperclusterNb::withSymmetry;
    runBenchmark("SUPERCLUSTERED SYMMETRIC", SymmetricSuperclusterNb::withoutCompression{256});
    runBenchmark("COMPRESSED SUPERCLUSTERED SYMMETRIC", SymmetricSuperclusterNb::withCompression{256});

    saveCsv(std::format("lennard_jones_results_{}_{}.csv", typeid(Tc).name(), typeid(T).name()), times);
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
