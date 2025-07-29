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
 * @brief Unit tests for DeviceParticlesData
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <iostream>

#include "gtest/gtest.h"

#include "sph/particles_data_gpu.cuh"

using namespace sphexa;

TEST(DeviceParticlesData, resize)
{
    DeviceParticlesData dev;

    dev.setConserved("x", "y", "z");
    dev.setDependent("du");

    size_t size = 10;
    dev.resize(10, 1.01);

    EXPECT_EQ(dev.x.size(), size);
    EXPECT_EQ(dev.y.size(), size);
    EXPECT_EQ(dev.z.size(), size);
    EXPECT_EQ(dev.du.size(), size);
}
