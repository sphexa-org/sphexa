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
#include "cstone/traversal/ijloop/gpu_compressednblist.cuh"
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

    T scale3 = 1.0;
    if (const char* scaleStr = std::getenv("CSTONE_NEIGHBOR_TEST_SCALE")) scale3 = std::atof(scaleStr);
    const T scale = std::cbrt(scale3);

    const unsigned ngmax = 320 * scale3;

    constexpr unsigned nx       = 100;
    const T h                   = 1.75 * scale;
    const float searchExtFactor = 1.9 * scale / h;

    FaceCenteredCubicCoordinates<Tc, StrongKeyType> coords(nx, nx, nx, {0, 1.6795962 * nx, BoundaryType::open});

    constexpr LjKernelFun<T> kernelFun{T(48), T(24)};
    constexpr auto inputValues         = std::tuple(T(12));
    constexpr auto initialOutputValues = std::tuple(
        std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN());

    std::map<std::string, std::vector<float>> times, buildTimes, bytesPerParticle;

    const auto runBenchmark = [&](const char* name, auto const& neighborhood)
    {
        printf("--- %s ---\n", name);
        const auto result =
            benchmarkNeighborhood<Tc, T, StrongKeyType>(coords, neighborhood, h, searchExtFactor, ngmax, kernelFun,
                                                        inputValues, initialOutputValues, std::is_same_v<T, double>);
        times[name]            = std::move(result.runTimes);
        buildTimes[name]       = {result.buildTime};
        bytesPerParticle[name] = {result.numBytesPerParticle};
        printf("\n");
    };

    runBenchmark("DIRECT TREE TRAVERSAL", ijloop::GpuAlwaysTraverseNeighborhoodBuilder{ngmax});
    runBenchmark("FULL NB LIST", ijloop::GpuFullNbListNeighborhoodBuilder{ngmax});
    runBenchmark("COMPRESSED FULL NB LIST", ijloop::GpuCompressedNbListNeighborhoodBuilder<>::withoutSymmetry{ngmax});
    runBenchmark("COMPRESSED HALF NB LIST", ijloop::GpuCompressedNbListNeighborhoodBuilder<>::withSymmetry{ngmax});
    runBenchmark("GROMACS SUPERCLUSTERED", ijloop::GromacsLikeNeighborhoodBuilder{ngmax});

    using SuperclusterNb = ijloop::GpuSuperclusterNbListNeighborhoodBuilder<>::withClusterSize<
        8, GpuConfig::warpSize / 8>::withSuperclusterSize<64>::withoutSymmetry;
    const unsigned ncmax = 300 + scale3 * 150;
    runBenchmark("SUPERCLUSTERED", SuperclusterNb::withoutCompression{ncmax});
    runBenchmark("COMPRESSED SUPERCLUSTERED (Band et al. Compression)",
                 SuperclusterNb::withCompression<BandEtAlWarpCompression<false>>{ncmax});
    runBenchmark("COMPRESSED SUPERCLUSTERED (Nibble-based Compression)",
                 SuperclusterNb::withCompression<NibbleWarpCompression<false>>{ncmax});

    using SymmetricSuperclusterNb = SuperclusterNb::withSymmetry;
    const unsigned ncmaxSymmetric = 300 + scale3 * 130;
    runBenchmark("SUPERCLUSTERED SYMMETRIC", SymmetricSuperclusterNb::withoutCompression{ncmaxSymmetric});
    runBenchmark("COMPRESSED SUPERCLUSTERED SYMMETRIC (Band et al. Compression)",
                 SymmetricSuperclusterNb::withCompression<BandEtAlWarpCompression<false>>{ncmaxSymmetric});
    runBenchmark("COMPRESSED SUPERCLUSTERED SYMMETRIC (Nibble-based Compression)",
                 SymmetricSuperclusterNb::withCompression<NibbleWarpCompression<false>>{ncmaxSymmetric});

    saveCsv(std::format("lennard_jones_results_{}_{}_{}.csv", typeid(Tc).name(), typeid(T).name(), scale3), times);
    saveCsv(std::format("lennard_jones_buildtime_{}_{}_{}.csv", typeid(Tc).name(), typeid(T).name(), scale3),
            buildTimes);
    saveCsv(std::format("lennard_jones_bytespp_{}_{}_{}.csv", typeid(Tc).name(), typeid(T).name(), scale3),
            bytesPerParticle);
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
