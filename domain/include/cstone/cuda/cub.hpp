/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

#pragma once

#ifdef __HIPCC__

#include <hipcub/hipcub.hpp>

namespace cub = hipcub;

#else

#include <cub/cub.cuh>

#endif
