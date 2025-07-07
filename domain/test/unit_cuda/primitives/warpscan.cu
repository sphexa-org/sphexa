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
#include <functional>
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

struct SomeStruct
{
    int a;
    float b;
    double c;
    bool d;

    bool operator==(SomeStruct const& other) const
    {
        return a == other.a && b == other.b && c == other.c && d == other.d;
    }
};

template<class T>
std::tuple<dim3, dim3, thrust::host_vector<T>> testData()
{
    const dim3 numBlocks = {5, 2, 3};
    const dim3 blockSize = {GpuConfig::warpSize / 4, 2, 6};

    thrust::host_vector<T> data(blockSize.x * blockSize.y * blockSize.z * numBlocks.x * numBlocks.y * numBlocks.z, T{});

    std::default_random_engine eng;
    if constexpr (std::is_same_v<T, SomeStruct>)
    {
        using IntDist    = std::uniform_int_distribution<int>;
        using FloatDist  = std::uniform_real_distribution<float>;
        using DoubleDist = std::uniform_real_distribution<double>;
        using BoolDist   = std::bernoulli_distribution;

        auto randomInt    = std::bind(IntDist{}, std::ref(eng));
        auto randomFloat  = std::bind(FloatDist{}, std::ref(eng));
        auto randomDouble = std::bind(DoubleDist{}, std::ref(eng));
        auto randomBool   = std::bind(BoolDist{}, std::ref(eng));

        std::generate(data.begin(), data.end() - GpuConfig::warpSize,
                      [&] { return SomeStruct{randomInt(), randomFloat(), randomDouble(), randomBool()}; });
    }
    else
    {
        using Dist = std::conditional_t<
            std::is_floating_point_v<T>, std::uniform_real_distribution<T>,
            std::conditional_t<std::is_same_v<T, bool>, std::bernoulli_distribution, std::uniform_int_distribution<T>>>;

        std::generate(data.begin(), data.end() - GpuConfig::warpSize, std::bind(Dist{}, std::ref(eng)));
    }

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

struct WarpLaneIndex
{
    __device__ unsigned operator()(unsigned) const { return laneIndex(); }

    static constexpr auto reference = [](auto input, auto output) { std::iota(output.begin(), output.end(), 0u); };
};

TEST(WarpScan, laneIndex) { testOnDevice<unsigned>(WarpLaneIndex{}); }

template<int Src>
struct WarpShflSync
{
    template<class T>
    __device__ T operator()(T x) const
    {
        return shflSync(x, Src);
    };

    static constexpr auto reference = [](auto input, auto output) { std::ranges::fill(output, input[Src]); };
};

TEST(WarpScan, shflSync)
{
    testOnDevice<int>(WarpShflSync<GpuConfig::warpSize / 10>{});
    testOnDevice<int>(WarpShflSync<GpuConfig::warpSize - 1>{});
    testOnDevice<float>(WarpShflSync<GpuConfig::warpSize / 3>{});
    testOnDevice<float>(WarpShflSync<GpuConfig::warpSize - 1>{});
    testOnDevice<double>(WarpShflSync<GpuConfig::warpSize / 7>{});
    testOnDevice<double>(WarpShflSync<GpuConfig::warpSize - 1>{});
    testOnDevice<SomeStruct>(WarpShflSync<GpuConfig::warpSize / 2>{});
    testOnDevice<SomeStruct>(WarpShflSync<GpuConfig::warpSize - 1>{});
}

template<GpuConfig::ThreadMask LaneMask>
struct WarpShflXorSync
{
    template<class T>
    __device__ T operator()(T x) const
    {
        return shflXorSync(x, LaneMask);
    };

    static constexpr auto reference = [](auto input, auto output)
    {
        for (std::size_t i = 0; i < output.size(); ++i)
            output[i] = input[i ^ LaneMask];
    };
};

TEST(WarpScan, shflXorSync)
{
    testOnDevice<int>(WarpShflXorSync<2>{});
    testOnDevice<int>(WarpShflXorSync<4>{});
    testOnDevice<float>(WarpShflXorSync<8>{});
    testOnDevice<float>(WarpShflXorSync<16>{});
    testOnDevice<double>(WarpShflXorSync<2>{});
    testOnDevice<double>(WarpShflXorSync<4>{});
    testOnDevice<SomeStruct>(WarpShflXorSync<8>{});
    testOnDevice<SomeStruct>(WarpShflXorSync<16>{});
}

template<unsigned Delta>
struct WarpShflUpSync
{
    template<class T>
    __device__ T operator()(T x) const
    {
        return shflUpSync(x, Delta);
    };

    static constexpr auto reference = [](auto input, auto output)
    {
        std::copy_n(input.begin(), Delta, output.begin());
        std::copy_n(input.begin(), GpuConfig::warpSize - Delta, output.begin() + Delta);
    };
};

TEST(WarpScan, shflUpSync)
{
    testOnDevice<int>(WarpShflUpSync<1>{});
    testOnDevice<int>(WarpShflUpSync<2>{});
    testOnDevice<float>(WarpShflUpSync<3>{});
    testOnDevice<float>(WarpShflUpSync<4>{});
    testOnDevice<double>(WarpShflUpSync<5>{});
    testOnDevice<double>(WarpShflUpSync<6>{});
    testOnDevice<SomeStruct>(WarpShflUpSync<7>{});
    testOnDevice<SomeStruct>(WarpShflUpSync<8>{});
}

template<unsigned Delta>
struct WarpShflDownSync
{
    template<class T>
    __device__ T operator()(T x) const
    {
        return shflDownSync(x, Delta);
    };

    static constexpr auto reference = [](auto input, auto output)
    {
        std::copy_n(input.begin() + Delta, GpuConfig::warpSize - Delta, output.begin());
        std::copy_n(input.end() - Delta, Delta, output.end() - Delta);
    };
};

TEST(WarpScan, shflDownSync)
{
    testOnDevice<int>(WarpShflDownSync<1>{});
    testOnDevice<int>(WarpShflDownSync<2>{});
    testOnDevice<float>(WarpShflDownSync<3>{});
    testOnDevice<float>(WarpShflDownSync<4>{});
    testOnDevice<double>(WarpShflDownSync<5>{});
    testOnDevice<double>(WarpShflDownSync<6>{});
    testOnDevice<SomeStruct>(WarpShflDownSync<7>{});
    testOnDevice<SomeStruct>(WarpShflDownSync<8>{});
}

struct WarpBallotSync
{
    __device__ GpuConfig::ThreadMask operator()(bool x) const { return ballotSync(x); };

    static constexpr auto reference = [](auto input, auto output)
    {
        GpuConfig::ThreadMask result = 0;
        for (std::size_t i = 0; i < output.size(); ++i)
            result |= input[i] << i;
        std::ranges::fill(output, result);
    };
};

TEST(WarpScan, ballotSync) { testOnDevice<bool, GpuConfig::ThreadMask>(WarpBallotSync{}); }

struct WarpAnySync
{
    __device__ bool operator()(bool x) const { return anySync(x); };

    static constexpr auto reference = [](auto input, auto output)
    { std::ranges::fill(output, std::accumulate(input.begin(), input.end(), false, std::logical_or<bool>{})); };
};

TEST(WarpScan, anySync) { testOnDevice<bool>(WarpAnySync{}); }

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

struct WarpBitwiseOr
{
    template<class T>
    __device__ T operator()(T x) const
    {
        return warpBitwiseOr(x);
    }

    static constexpr auto reference = [](auto input, auto output)
    {
        using T = decltype(output)::value_type;
        std::ranges::fill(output, std::accumulate(input.begin(), input.end(), T(0), std::bit_or<T>{}));
    };
};

TEST(WarpScan, warpBitwiseOr)
{
    testOnDevice<int>(WarpBitwiseOr{});
    testOnDevice<unsigned>(WarpBitwiseOr{});
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

struct WarpReduceBool
{
    __device__ int operator()(bool x) const { return reduceBool(x); }

    static constexpr auto reference = [](auto input, auto output)
    { std::ranges::fill(output, std::accumulate(input.begin(), input.end(), 0)); };
};

TEST(WarpScan, reduceBool) { testOnDevice<bool, int>(WarpReduceBool{}); }

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

__global__ void applyAtomicMinFloat(float* addr, float value)
{
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    atomicMinFloat(addr, index == 137 ? value : 2025.0f);
}

TEST(WarpScan, atomicMinFloat)
{
    thrust::device_vector<float> d_value(1);

    // check especially corner cases -0.0f, 0.0f
    for (float firstSign : {-1.0f, -0.0f, 0.0f, 1.0f})
        for (float secondSign : {-1.0f, -0.0f, 0.0f, 1.0f})
        {
            d_value[0] = 42.0f * firstSign;
            applyAtomicMinFloat<<<2, 128>>>(rawPtr(d_value), 37.5f * secondSign);
            EXPECT_EQ(float(d_value[0]), std::min(42.0f * firstSign, 37.5f * secondSign));
        }
}

__global__ void applyAtomicMaxFloat(float* addr, float value)
{
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    atomicMaxFloat(addr, index == 137 ? value : -2025.0f);
}

TEST(WarpScan, atomicMaxFloat)
{
    thrust::device_vector<float> d_value(1);

    // check especially corner cases -0.0f, 0.0f
    for (float firstSign : {1.0f, -0.0f, 0.0f, 1.0f})
        for (float secondSign : {-1.0f, -0.0f, 0.0f, 1.0f})
        {
            d_value[0] = 42.0f * firstSign;
            applyAtomicMaxFloat<<<2, 128>>>(rawPtr(d_value), 37.5f * secondSign);
            EXPECT_EQ(float(d_value[0]), std::max(42.0f * firstSign, 37.5f * secondSign));
        }
}
