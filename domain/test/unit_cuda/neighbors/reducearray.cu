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
 * @brief Fast array warp-reductions tests
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#include <tuple>
#include <type_traits>

#include <thrust/universal_vector.h>

#include "gtest/gtest.h"

#include "cstone/cuda/thrust_util.cuh"
#include "cstone/reducearray.cuh"

using namespace cstone;

template<unsigned ReductionSize, bool Interleave, std::size_t ArraySize>
__global__ void runReduction(util::array<int, ArraySize> const* __restrict__ in,
                             util::array<int, ArraySize>* __restrict__ out)
{
    constexpr unsigned reductions = GpuConfig::warpSize / ReductionSize;
    const unsigned i              = threadIdx.x;
    const int res                 = reduceArray<ReductionSize, Interleave>(in[i], [](auto a, auto b) { return a + b; });
    const unsigned r              = Interleave ? i % reductions : i / ReductionSize;
    const unsigned j              = Interleave ? i / reductions : i % ReductionSize;
    if (j < ArraySize) out[r][j] = res;
}

template<unsigned ReductionSize, bool Interleave, std::size_t ArraySize>
thrust::universal_vector<util::array<int, ArraySize>>
reference(const thrust::universal_vector<util::array<int, ArraySize>>& in)
{
    constexpr unsigned reductions = GpuConfig::warpSize / ReductionSize;
    thrust::universal_vector<util::array<int, ArraySize>> res(reductions);
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
thrust::universal_vector<util::array<int, N>> testData()
{
    thrust::universal_vector<util::array<int, N>> data(GpuConfig::warpSize);
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
    thrust::universal_vector<util::array<int, TypeParam::arraySize>> in = testData<TypeParam::arraySize>();
    const auto ref = reference<TypeParam::reductionSize, TypeParam::interleave>(in);
    thrust::universal_vector<util::array<int, TypeParam::arraySize>> out(ref.size());
    runReduction<TypeParam::reductionSize, TypeParam::interleave><<<1, GpuConfig::warpSize>>>(rawPtr(in), rawPtr(out));
    checkGpuErrors(cudaDeviceSynchronize());
    EXPECT_EQ(out, ref);
}
