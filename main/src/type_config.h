/*
 * SPH-EXA
 *
 * Copyright (c) 2026 CSCS, ETH Zurich
 *               2026 University of Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief List of builtin types supported in IO operations
 *
 * @author Noah Kubli <noah.kubli@uzh.ch>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/util/type_list.hpp"

namespace sphexa
{

struct IO
{
    template<class T>
    using ConstPtr = const T*;

    using Types = util::TypeList<double, float, char, unsigned char, signed char, short, int, long, long long,
                                 unsigned short, unsigned, unsigned long, unsigned long long>;
};

} // namespace sphexa
