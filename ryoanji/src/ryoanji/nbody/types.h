/*
 * Ryoanji N-body solver
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  helper types
 *
 * @author Rio Yokota <rioyokota@gsic.titech.ac.jp>
 */

#pragma once

#include "cstone/cuda/annotation.hpp"
#include "cstone/sfc/box.hpp"
#include "cstone/util/array.hpp"

namespace ryoanji
{

template<class T>
using Vec3 = cstone::Vec3<T>;

template<class T>
using Vec4 = cstone::Vec4<T>;

using TreeNodeIndex = cstone::TreeNodeIndex;
using LocalIndex    = cstone::LocalIndex;

} // namespace ryoanji
