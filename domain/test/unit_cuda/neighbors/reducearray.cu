/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Fast array warp-reductions tests
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#include <array>
#include <tuple>
#include <type_traits>

#include <thrust/universal_vector.h>

#include "gtest/gtest.h"

#include "cstone/cuda/thrust_util.cuh"
#include "cstone/reducearray.cuh"

using namespace cstone;

template<unsigned ReductionSize, bool Interleave, std::size_t ArraySize>
__global__ void runReduction(std::array<int, ArraySize> const* __restrict__ in,
                             std::array<int, ArraySize>* __restrict__ out)
{
    constexpr unsigned reductions = GpuConfig::warpSize / ReductionSize;
    const unsigned i              = threadIdx.x;
    const int res                 = reduceArray<ReductionSize, Interleave>(in[i], [](auto a, auto b) { return a + b; });
    const unsigned r              = Interleave ? i % reductions : i / ReductionSize;
    const unsigned j              = Interleave ? i / reductions : i % ReductionSize;
    if (j < ArraySize) out[r][j] = res;
}

template<unsigned ReductionSize, bool Interleave, std::size_t ArraySize>
thrust::universal_vector<std::array<int, ArraySize>>
reference(const thrust::universal_vector<std::array<int, ArraySize>>& in)
{
    constexpr unsigned reductions = GpuConfig::warpSize / ReductionSize;
    thrust::universal_vector<std::array<int, ArraySize>> res(reductions);
    for (unsigned i = 0; i < reductions; ++i)
    {
        auto& resI = res[i];
        for (unsigned j = 0; j < ReductionSize; ++j)
        {
            auto const& inJ = in[Interleave ? i + reductions * j : ReductionSize * i + j];
            for (unsigned k = 0; k < ArraySize; ++k)
                resI[k] += inJ[k];
        }
    }
    return res;
}

template<unsigned N>
thrust::universal_vector<std::array<int, N>> testData()
{
    thrust::universal_vector<std::array<int, N>> data(GpuConfig::warpSize);
    int value = 0;
    for (unsigned i = 0; i < data.size(); ++i)
        for (unsigned j = 0; j < N; ++j)
            data[i][j] = value++;
    return data;
}

template<unsigned ReductionSize, bool Interleave, std::size_t ArraySize>
struct Param
{
    constexpr static unsigned reductionSize = ReductionSize;
    constexpr static bool interleave        = Interleave;
    constexpr static std::size_t arraySize  = ArraySize;
};

using TestTypes = ::testing::Types<Param<1, false, 1>,
                                   Param<2, false, 1>,
                                   Param<4, false, 1>,
                                   Param<8, false, 1>,
                                   Param<16, false, 1>,
                                   Param<32, false, 1>,
                                   Param<1, true, 1>,
                                   Param<2, true, 1>,
                                   Param<4, true, 1>,
                                   Param<8, true, 1>,
                                   Param<16, true, 1>,
                                   Param<32, true, 1>,
                                   Param<2, false, 2>,
                                   Param<4, false, 2>,
                                   Param<8, false, 2>,
                                   Param<16, false, 2>,
                                   Param<32, false, 2>,
                                   Param<2, true, 2>,
                                   Param<4, true, 2>,
                                   Param<8, true, 2>,
                                   Param<16, true, 2>,
                                   Param<32, true, 2>,
                                   Param<4, false, 3>,
                                   Param<8, false, 3>,
                                   Param<16, false, 3>,
                                   Param<32, false, 3>,
                                   Param<4, true, 3>,
                                   Param<8, true, 3>,
                                   Param<16, true, 3>,
                                   Param<32, true, 3>>;

template<class T>
struct ReduceArrayGpu : testing::Test
{
};

TYPED_TEST_SUITE(ReduceArrayGpu, TestTypes);

TYPED_TEST(ReduceArrayGpu, full)
{
    thrust::universal_vector<std::array<int, TypeParam::arraySize>> in = testData<TypeParam::arraySize>();
    const auto ref = reference<TypeParam::reductionSize, TypeParam::interleave>(in);
    thrust::universal_vector<std::array<int, TypeParam::arraySize>> out(ref.size());
    runReduction<TypeParam::reductionSize, TypeParam::interleave><<<1, GpuConfig::warpSize>>>(rawPtr(in), rawPtr(out));
    checkGpuErrors(cudaDeviceSynchronize());
#ifdef __HIP_PLATFORM_AMD__
    // Workaround for compiler bug in HIP/ROCm 6.3
    EXPECT_TRUE(std::equal(out.begin(), out.end(), ref.begin()));
#else
    EXPECT_EQ(out, ref);
#endif
}
