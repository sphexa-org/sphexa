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
 * @brief Switch and stub classes for ParticlesData to abstract and manage GPU device acceleration behavior
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/primitives/accel_switch.hpp"

template<class T>
class pinned_allocator;

namespace sphexa
{

//! @brief std::allocator on the CPU, pinned_allocator on the GPU
template<class Accelerator, class T>
using PinnedAlloc_t = typename cstone::AccelSwitchType<Accelerator, std::allocator, pinned_allocator>::template type<T>;

//! @brief stub for use in CPU code
struct DeviceDataFacade
{
    void   resize(size_t, float) {}
    size_t size() { return 0; }

    template<class... Ts>
    void setConserved(Ts...)
    {
    }

    template<class... Ts>
    void setDependent(Ts...)
    {
    }

    template<class... Ts>
    void release(Ts...)
    {
    }

    template<class... Ts>
    void acquire(Ts...)
    {
    }

    template<class Table>
    void uploadTables(const Table&, const Table&)
    {
    }

    inline static constexpr std::array fieldNames{0};
};

class DeviceParticlesData;

//! @brief Just a facade on the CPU, DeviceParticlesData on the GPU
template<class Accelerator>
using DeviceData_t = typename cstone::AccelSwitchTypeSimple<Accelerator, DeviceDataFacade, DeviceParticlesData>::type;

} // namespace sphexa
