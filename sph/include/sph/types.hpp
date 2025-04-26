/*
 * MIT License
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *               2024 University of Basel
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
 * @brief Types and definitions for SPH interfaces
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <cstdint>
#include <variant>
#include <vector>

#include "cstone/util/type_list.hpp"

namespace sph
{

struct SphTypes
{
    using KeyType        = uint64_t;
    using CoordinateType = double;
    using HydroType      = float;
    using XM1Type        = float;
    using Tmass          = float;
};

//TODO: does it make sense to have these definitions here? Same for particles_data_gpu.cuh
template<class ValueType>
using FieldVector = std::vector<ValueType, std::allocator<ValueType>>;

using FieldVariant = std::variant<FieldVector<float>*, FieldVector<double>*, FieldVector<unsigned>*,
                                  FieldVector<uint64_t>*, FieldVector<uint8_t>*>;

//TODO: this definition is not strictly related to SPH stuff, I put it here to avoid the creation of a new include
// In general I need it to have consistency with the types used by ParticlesData for the fields.
//TODO: what about having something similar for DevVector based variant in particles_data_gpu.cuh?
using BufferFieldVariant = util::Reduce<std::variant, util::Map<std::remove_pointer_t, FieldVariant>>;

} // namespace sph