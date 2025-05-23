/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

#pragma once

#if defined(USE_CUDA) || defined(__CUDACC__) || defined(__HIPCC__)
#include "cuda_utils.cuh"
#else
#include "cuda_stubs.h"
#endif
