/*
 * MIT License
 *
 * Copyright (c) 2025 CSCS, ETH Zurich, University of Basel, University of Zurich
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
 * @brief Id tagging setup utilities
 *
 * @author Christopher Bignamini <christopher.bignamini@gmail.com>
 */

#pragma once

#include <string>
#include <vector> // TODO: to be removed after replacing IdSubsets types

#include "init/settings.hpp"

namespace sphexa
{
    // TODO: placeholder type, to be replaced with actual subsets data structure
    using IdSubsets = std::map<std::string, std::vector<uint64_t>>;

    /*! @brief Id tagging setup checks and defaults assignment
     *
     * @param[inout]  settings    settings for id tagging
     */
    // TODO: Only pass the relevant subset of settings?
    void idTaggingSetupInit(InitSettings& settings);
} 