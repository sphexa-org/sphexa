/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Neighbor list compression tests
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#include <cstdint>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "gtest/gtest.h"

#include "cstone/cuda/errorcheck.cuh"
#include "cstone/cuda/gpu_config.cuh"
#include "cstone/cuda/thrust_util.cuh"
#include "cstone/traversal/ijloop/compressneighbors.cuh"

using namespace cstone;

template<class Compression>
__global__ void roundtrip(std::uint32_t const* __restrict__ input,
                          std::uint32_t* __restrict__ output,
                          const unsigned numNeighbors,
                          const unsigned sharedMemSize)
{
    extern __shared__ char compressed[];

    const unsigned nBytes = warpCompressNeighbors<Compression>(input, compressed, numNeighbors);
    __syncthreads();
    if (threadIdx.x == 0)
    {
        for (unsigned i = nBytes; i < sharedMemSize; ++i)
            compressed[i] = 0xff;
    }
    __syncthreads();
    warpDecompressNeighbors<typename Compression::Decompression>(compressed, output, numNeighbors);
}

template<class T>
class CompressNeighborsGpu : public ::testing::Test
{
};

using Compressions =
    ::testing::Types<NibbleWarpCompression<false>, BandEtAlWarpCompression<false>, DummyWarpCompression<false>>;

TYPED_TEST_SUITE(CompressNeighborsGpu, Compressions);

TYPED_TEST(CompressNeighborsGpu, roundtrip)
{
    using Compression = TypeParam;

    thrust::device_vector<std::uint32_t> nbs = {300, 301, 302, 100, 101, 200, 400, 402, 403,
                                                404, 405, 406, 407, 408, 409, 410, 411};
    thrust::device_vector<std::uint32_t> roundtripped(nbs.size());

    const unsigned sharedMemSize = sizeof(std::uint32_t) * nbs.size();
    roundtrip<Compression>
        <<<1, GpuConfig::warpSize, sharedMemSize>>>(rawPtr(nbs), rawPtr(roundtripped), nbs.size(), sharedMemSize);
    kernelSuccess("roundtrip");

    EXPECT_EQ(roundtripped, nbs);
}

TYPED_TEST(CompressNeighborsGpu, empty)
{
    using Compression = TypeParam;

    thrust::device_vector<std::uint32_t> nbs(0);
    thrust::device_vector<std::uint32_t> roundtripped(nbs.size());

    const unsigned sharedMemSize = sizeof(std::uint32_t);
    roundtrip<Compression>
        <<<1, GpuConfig::warpSize, sharedMemSize>>>(rawPtr(nbs), rawPtr(roundtripped), nbs.size(), sharedMemSize);
    kernelSuccess("roundtrip");

    EXPECT_EQ(nbs, roundtripped);
}

TYPED_TEST(CompressNeighborsGpu, manyConsecutive)
{
    using Compression = TypeParam;

    thrust::device_vector<std::uint32_t> nbs = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 16, 17,
                                                18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                                                34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49};
    thrust::device_vector<std::uint32_t> roundtripped(nbs.size());

    const unsigned sharedMemSize = sizeof(std::uint32_t) * nbs.size();
    roundtrip<Compression>
        <<<1, GpuConfig::warpSize, sharedMemSize>>>(rawPtr(nbs), rawPtr(roundtripped), nbs.size(), sharedMemSize);
    kernelSuccess("roundtrip");

    EXPECT_EQ(roundtripped, nbs);
}

TYPED_TEST(CompressNeighborsGpu, large)
{
    using Compression = TypeParam;

    thrust::device_vector<std::uint32_t> nbs = {
        777363, 777364, 777374, 777375, 777376, 777377, 777387, 777389, 777390, 777391, 777398, 777399, 777400,
        777401, 777402, 777403, 777404, 777405, 782347, 782360, 782363, 782365, 782366, 782368, 782369, 782380,
        782381, 782382, 782384, 782397, 783245, 783249, 783250, 783251, 783253, 783254, 783255, 783256, 783277,
        783278, 783280, 783281, 784926, 784929, 784938, 784939, 784941, 784952, 784953, 784956, 784957, 785032,
        785035, 785036, 785037, 785038, 785039, 785054, 785057, 785059, 785060, 785063, 785064, 785070, 785071,
        785072, 785073, 785074, 785075, 785076, 785077, 785078, 785079, 785080, 785081, 785082, 785083, 785084,
        785085, 785086, 785087, 785092, 785093, 785094, 785095, 785096, 785097, 785098, 785099, 785100, 785101,
        785102, 785103, 785104, 785105, 785106, 785107, 785108, 785109, 785110, 785111, 785112, 785113, 785114,
        785115, 785116, 785117, 785118, 785119, 785120, 785121, 785122, 785123, 785124, 785125, 785126, 785127,
        785128, 785129, 785130, 785131, 785132, 785133, 785134, 785135, 785137, 785141, 785145, 785146, 785151};
    thrust::device_vector<std::uint32_t> roundtripped(nbs.size());

    const unsigned sharedMemSize = sizeof(std::uint32_t) * nbs.size();
    roundtrip<Compression>
        <<<1, GpuConfig::warpSize, sharedMemSize>>>(rawPtr(nbs), rawPtr(roundtripped), nbs.size(), sharedMemSize);
    kernelSuccess("roundtrip");

    EXPECT_EQ(roundtripped, nbs);
}

template<class Compression>
__global__ void perThreadRoundtrip(std::uint32_t const* __restrict__ input,
                                   std::uint32_t const* __restrict__ offsets,
                                   std::uint32_t* __restrict__ output,
                                   const unsigned sharedMemSize)
{
    extern __shared__ char compressed[];

    const unsigned numNeighbors = offsets[threadIdx.x + 1] - offsets[threadIdx.x];

    const unsigned nBytes = warpCompressNeighbors<Compression>(input + offsets[threadIdx.x], compressed, numNeighbors);
    __syncthreads();
    if (threadIdx.x == 0)
    {
        for (unsigned i = nBytes; i < sharedMemSize; ++i)
            compressed[i] = 0xff;
    }
    __syncthreads();
    warpDecompressNeighbors<typename Compression::Decompression>(compressed, output + offsets[threadIdx.x],
                                                                 numNeighbors);
}

template<class T>
class PerThreadCompressNeighborsGpu : public ::testing::Test
{
};

using PerThreadCompressions =
    ::testing::Types<NibbleWarpCompression<true>, BandEtAlWarpCompression<true>, DummyWarpCompression<true>>;

TYPED_TEST_SUITE(PerThreadCompressNeighborsGpu, PerThreadCompressions);

TYPED_TEST(PerThreadCompressNeighborsGpu, perThreadRoundtrip)
{
    using Compression = TypeParam;

    std::vector<std::uint32_t> nbsHost(2000), offsetsHost(GpuConfig::warpSize + 1);

    std::default_random_engine eng(42);
    offsetsHost[0]                   = 0;
    offsetsHost[GpuConfig::warpSize] = nbsHost.size();
    std::generate(offsetsHost.begin() + 1, offsetsHost.end() - 1,
                  std::bind(std::uniform_int_distribution<std::uint32_t>(1, nbsHost.size() - 1), std::ref(eng)));
    std::sort(offsetsHost.begin() + 1, offsetsHost.end() - 1);

    std::bernoulli_distribution bd(0.25);
    unsigned i = 0, j = 0, counter = 100000;
    while (i < nbsHost.size())
    {
        if (bd(eng))
        {
            if (i == offsetsHost[j])
            {
                counter -= 100;
                ++j;
            }
            nbsHost[i++] = counter;
        }
        ++counter;
    }

    thrust::device_vector<std::uint32_t> nbs = nbsHost;
    thrust::device_vector<std::uint32_t> roundtripped(nbs.size());
    thrust::device_vector<std::uint32_t> offsets = offsetsHost;

    const unsigned sharedMemSize = sizeof(std::uint32_t) * nbs.size();
    perThreadRoundtrip<Compression>
        <<<1, GpuConfig::warpSize, sharedMemSize>>>(rawPtr(nbs), rawPtr(offsets), rawPtr(roundtripped), sharedMemSize);
    kernelSuccess("roundtrip");

    EXPECT_EQ(roundtripped, nbs);
}

TYPED_TEST(PerThreadCompressNeighborsGpu, perThreadEmpty)
{
    using Compression = TypeParam;

    thrust::device_vector<std::uint32_t> offsets(cstone::GpuConfig::warpSize + 1);

    const unsigned sharedMemSize = sizeof(std::uint32_t);
    perThreadRoundtrip<Compression>
        <<<1, GpuConfig::warpSize, sharedMemSize>>>(nullptr, rawPtr(offsets), nullptr, sharedMemSize);
    kernelSuccess("roundtrip");
}

TYPED_TEST(PerThreadCompressNeighborsGpu, perThreadManyConsecutive)
{
    using Compression = TypeParam;

    std::vector<std::uint32_t> nbsHost(2000), offsetsHost(GpuConfig::warpSize + 1);

    std::default_random_engine eng(42);
    offsetsHost[0]                   = 0;
    offsetsHost[GpuConfig::warpSize] = nbsHost.size();
    std::generate(offsetsHost.begin() + 1, offsetsHost.end() - 1,
                  std::bind(std::uniform_int_distribution<std::uint32_t>(1, nbsHost.size() - 1), std::ref(eng)));
    std::sort(offsetsHost.begin() + 1, offsetsHost.end() - 1);

    std::iota(nbsHost.begin(), nbsHost.end(), 1000);

    thrust::device_vector<std::uint32_t> nbs = nbsHost;
    thrust::device_vector<std::uint32_t> roundtripped(nbs.size());
    thrust::device_vector<std::uint32_t> offsets = offsetsHost;

    const unsigned sharedMemSize = sizeof(std::uint32_t) * nbs.size();
    perThreadRoundtrip<Compression>
        <<<1, GpuConfig::warpSize, sharedMemSize>>>(rawPtr(nbs), rawPtr(offsets), rawPtr(roundtripped), sharedMemSize);
    kernelSuccess("roundtrip");

    EXPECT_EQ(roundtripped, nbs);
}
