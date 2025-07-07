/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Tests for warp-level primitives
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include <algorithm>
#include <random>
#include <ranges>
#include <span>
#include <type_traits>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cstone/cuda/thrust_util.cuh"
#include "cstone/primitives/warpscan.cuh"

using namespace cstone;

__device__ unsigned globalIndex()
{
    const auto blockIndex = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    return blockIndex * blockDim.x * blockDim.y * blockDim.z + threadIdx.x + threadIdx.y * blockDim.x +
           threadIdx.z * blockDim.x * blockDim.y;
}

template<class InputT, class OutputT = InputT, class F>
__global__ void applyWarpFunction(InputT* const input, OutputT* output, F f)
{
    const unsigned index = globalIndex();
    output[index]        = f(input[index]);
}

template<class T>
std::tuple<dim3, dim3, thrust::host_vector<T>> testData()
{
    const dim3 numBlocks = {5, 2, 3};
    const dim3 blockSize = {GpuConfig::warpSize / 4, 2, 6};

    thrust::host_vector<T> data(blockSize.x * blockSize.y * blockSize.z * numBlocks.x * numBlocks.y * numBlocks.z,
                                T(0));
    std::fill(data.begin() + GpuConfig::warpSize, data.begin() + GpuConfig::warpSize * 2, T(1));

    using Dist = std::conditional_t<
        std::is_floating_point_v<T>, std::uniform_real_distribution<T>,
        std::conditional_t<std::is_same_v<T, bool>, std::bernoulli_distribution, std::uniform_int_distribution<T>>>;

    Dist dist;
    std::default_random_engine eng;
    std::generate(data.begin() + GpuConfig::warpSize * 2, data.end(), std::bind(dist, std::ref(eng)));

    return {std::move(numBlocks), std::move(blockSize), std::move(data)};
}

template<class T>
using WarpSpan = std::span<T, GpuConfig::warpSize>;

template<class InputT, class OutputT, class WarpF>
thrust::host_vector<OutputT> computeReference(thrust::host_vector<InputT> const& input, WarpF warpF)
{
    thrust::host_vector<OutputT> output(input.size());
    for (std::size_t warp = 0; warp < input.size() / GpuConfig::warpSize; ++warp)
    {
        WarpSpan<const InputT> warpInput(&input[warp * GpuConfig::warpSize], &input[(warp + 1) * GpuConfig::warpSize]);
        WarpSpan<OutputT> warpOutput(&output[warp * GpuConfig::warpSize], &output[(warp + 1) * GpuConfig::warpSize]);
        warpF(warpInput, warpOutput);
    }
    return output;
}

template<class InputT, class OutputT = InputT, class F>
void testOnDevice(F f)
{
    const auto [numBlocks, blockSize, input] = testData<InputT>();

    thrust::device_vector<InputT> deviceInput = input;
    thrust::device_vector<OutputT> deviceOutput(input.size());
    applyWarpFunction<<<numBlocks, blockSize>>>(rawPtr(deviceInput), rawPtr(deviceOutput), f);
    checkGpuErrors(cudaDeviceSynchronize());

    thrust::host_vector<OutputT> reference = computeReference<InputT, OutputT>(input, F::reference);
    thrust::host_vector<OutputT> output    = deviceOutput;

    EXPECT_EQ(output, reference);
}

struct WarpMin
{
    template<class T>
    __device__ T operator()(T x) const
    {
        return warpMin(x);
    }

    static constexpr auto reference = [](auto input, auto output)
    { std::ranges::fill(output, *std::ranges::min_element(input)); };
};

TEST(WarpScan, warpMin)
{
    testOnDevice<int>(WarpMin{});
    testOnDevice<float>(WarpMin{});
    testOnDevice<double>(WarpMin{});
}

struct WarpMax
{
    template<class T>
    __device__ T operator()(T x) const
    {
        return warpMax(x);
    }

    static constexpr auto reference = [](auto input, auto output)
    { std::ranges::fill(output, *std::ranges::max_element(input)); };
};

TEST(WarpScan, warpMax)
{
    testOnDevice<int>(WarpMax{});
    testOnDevice<float>(WarpMax{});
    testOnDevice<double>(WarpMax{});
}

struct WarpInclusiveScanInt
{
    __device__ int operator()(int x) const { return inclusiveScanInt(x); }

    static constexpr auto reference = [](auto input, auto output)
    { std::inclusive_scan(input.begin(), input.end(), output.begin()); };
};

TEST(WarpScan, inclusiveScanInt) { testOnDevice<int>(WarpInclusiveScanInt{}); }

struct WarpExclusiveScanBool
{
    __device__ int operator()(bool x) const { return exclusiveScanBool(x); }

    static constexpr auto reference = [](auto input, auto output)
    { std::exclusive_scan(input.begin(), input.end(), output.begin(), 0, std::plus<int>()); };
};

TEST(WarpScan, exclusiveScanBool) { testOnDevice<bool, int>(WarpExclusiveScanBool{}); }

template<int Carry>
struct WarpInclusiveSegscanInt
{
    __device__ int operator()(int x) const { return inclusiveSegscanInt(x, Carry); }

    static constexpr auto reference = [](auto input, auto output)
    {
        int result = Carry;
        for (std::size_t i = 0; i < input.size(); ++i)
        {
            result    = input[i] < 0 ? -input[i] - 1 : result + input[i];
            output[i] = result;
        }
    };
};

TEST(WarpScan, inclusiveSegscanInt)
{
    testOnDevice<int>(WarpInclusiveSegscanInt<1>{});
    testOnDevice<int>(WarpInclusiveSegscanInt<42>{});
    testOnDevice<int>(WarpInclusiveSegscanInt<-42>{});
}

struct WarpStreamCompact
{
    template<class T>
    __device__ T operator()(T x) const
    {
        __shared__ T buffer[GpuConfig::warpSize * 3];
        T* tmp            = buffer + GpuConfig::warpSize * (threadIdx.z / 2);
        const int numKeep = streamCompact(&x, x <= 0, tmp);
        return laneIndex() < numKeep ? x : T(42);
    }

    static constexpr auto reference = [](auto input, auto output)
    {
        auto [_, out] = std::ranges::copy_if(input, output.begin(), [](auto x) { return x <= 0; });
        std::fill(out, output.end(), 42);
    };
};

TEST(WarpScan, streamCompact)
{
    testOnDevice<int>(WarpStreamCompact{});
    testOnDevice<float>(WarpStreamCompact{});
    testOnDevice<double>(WarpStreamCompact{});
}

struct WarpSpreadSeg8
{
    __device__ int operator()(int x) const { return spreadSeg8(x); }

    static constexpr auto reference = [](auto input, auto output)
    {
        for (std::size_t i = 0; i < output.size(); ++i)
            output[i] = i % 8 == 0 ? input[i / 8] : output[i - 1] + 1;
    };
};

TEST(WarpScan, warpSpreadSeg8) { testOnDevice<int>(WarpSpreadSeg8{}); }
