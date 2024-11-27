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
 * @brief Cornerstone octree GPU testing
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 */

#include <numeric>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include "gtest/gtest.h"

#include "cstone/cuda/thrust_util.cuh"
#include "cstone/primitives/primitives_gpu.h"

using namespace cstone;

TEST(PrimitivesGpu, MinMax)
{
    using thrust::raw_pointer_cast;

    thrust::device_vector<double> v = std::vector<double>{1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};

    auto minMax = MinMaxGpu<double>{}(raw_pointer_cast(v.data()), raw_pointer_cast(v.data()) + v.size());

    EXPECT_EQ(std::get<0>(minMax), 1.);
    EXPECT_EQ(std::get<1>(minMax), 10.);
}

TEST(PrimitivesGpu, segmentMax)
{
    unsigned numElements = 1000;
    thrust::device_vector<double> v(numElements);

    thrust::sequence(v.begin(), v.end(), 0);

    unsigned numSegments = 120;
    std::vector<unsigned> h_segments(numSegments + 1, numElements / numSegments);
    h_segments[numSegments - 1] += numElements % numSegments;
    std::exclusive_scan(h_segments.begin(), h_segments.end(), h_segments.begin(), 0);

    thrust::device_vector<unsigned> segments = h_segments;
    thrust::device_vector<double> output(numSegments);

    segmentMax(rawPtr(v), rawPtr(segments), numSegments, rawPtr(output));

    thrust::host_vector<double> h_output = output;

    for (unsigned i = 0; i < numSegments; ++i)
    {
        EXPECT_EQ(h_segments[i + 1] - 1, h_output[i]);
    }
}
