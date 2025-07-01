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

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "gtest/gtest.h"

#include "cstone/cuda/errorcheck.cuh"
#include "cstone/cuda/gpu_config.cuh"
#include "cstone/cuda/thrust_util.cuh"
#include "cstone/traversal/ijloop/compressneighbors.cuh"

using namespace cstone;

__global__ void roundtrip(std::uint32_t const* __restrict__ input,
                          std::uint32_t* __restrict__ output,
                          unsigned n_input,
                          unsigned* n_output,
                          unsigned sharedMemSize)
{
    extern __shared__ char compressed[];

    warpCompressNeighbors(input, compressed, n_input);
    __syncthreads();
    if (threadIdx.x == 0)
    {
        const unsigned nBytes = compressedNeighborsSize(compressed);
        for (unsigned i = nBytes; i < sharedMemSize; ++i)
            compressed[i] = 0xff;
    }
    __syncthreads();
    warpDecompressNeighbors(compressed, output, *n_output);
}

TEST(CompressNeighborsGpu, roundtrip)
{
    thrust::device_vector<std::uint32_t> nbs = {300, 301, 302, 100, 101, 200, 400, 402, 403,
                                                404, 405, 406, 407, 408, 409, 410, 411};
    thrust::device_vector<std::uint32_t> roundtripped(nbs.size());
    thrust::device_vector<unsigned> output_nb_count(1);

    const unsigned sharedMemSize = sizeof(std::uint32_t) * nbs.size();
    roundtrip<<<1, GpuConfig::warpSize, sharedMemSize>>>(rawPtr(nbs), rawPtr(roundtripped), nbs.size(),
                                                         rawPtr(output_nb_count), sharedMemSize);
    kernelSuccess("roundtrip");

    ASSERT_EQ(output_nb_count[0], nbs.size());
    EXPECT_EQ(roundtripped, nbs);
}

TEST(CompressNeighborsGpu, empty)
{
    thrust::device_vector<std::uint32_t> nbs(0);
    thrust::device_vector<std::uint32_t> roundtripped(nbs.size());
    thrust::device_vector<unsigned> output_nb_count(1);

    const unsigned sharedMemSize = sizeof(std::uint32_t);
    roundtrip<<<1, GpuConfig::warpSize, sharedMemSize>>>(rawPtr(nbs), rawPtr(roundtripped), nbs.size(),
                                                         rawPtr(output_nb_count), sharedMemSize);
    kernelSuccess("roundtrip");

    ASSERT_EQ(output_nb_count[0], nbs.size());
    EXPECT_EQ(nbs, roundtripped);
}

TEST(CompressNeighborsGpu, manyConsecutive)
{
    thrust::device_vector<std::uint32_t> nbs = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 16, 17,
                                                18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                                                34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49};
    thrust::device_vector<std::uint32_t> roundtripped(nbs.size());
    thrust::device_vector<unsigned> output_nb_count(1);

    const unsigned sharedMemSize = sizeof(std::uint32_t) * nbs.size();
    roundtrip<<<1, GpuConfig::warpSize, sharedMemSize>>>(rawPtr(nbs), rawPtr(roundtripped), nbs.size(),
                                                         rawPtr(output_nb_count), sharedMemSize);
    kernelSuccess("roundtrip");

    EXPECT_EQ(output_nb_count[0], nbs.size());
    EXPECT_EQ(roundtripped, nbs);
}

TEST(CompressNeighborsGpu, large)
{
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
    thrust::device_vector<unsigned> output_nb_count(1);

    const unsigned sharedMemSize = sizeof(std::uint32_t) * nbs.size();
    roundtrip<<<1, GpuConfig::warpSize, sharedMemSize>>>(rawPtr(nbs), rawPtr(roundtripped), nbs.size(),
                                                         rawPtr(output_nb_count), sharedMemSize);
    kernelSuccess("roundtrip");

    ASSERT_EQ(output_nb_count[0], nbs.size());
    EXPECT_EQ(roundtripped, nbs);
}
