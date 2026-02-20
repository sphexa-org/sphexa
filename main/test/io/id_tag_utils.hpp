/*
 * MIT License
 *
 * SPH-EXA
 * Copyright (c) 2024 CSCS, ETH Zurich, University of Basel, University of Zurich
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
 * @brief Utility functions for id tagging related functionality unit tests
 *
 * @author Christopher Bignamini <christopher.bignamini@gmail.com>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <vector>

#include "io/id_tag_utils.hpp"

void makeParticleDistribution(std::vector<sphexa::CoordinateType>& x, std::vector<sphexa::CoordinateType>& y,
                              std::vector<sphexa::CoordinateType>& z, std::size_t numParticles)
{
    x.resize(numParticles);
    y.resize(numParticles);
    z.resize(numParticles);

    unsigned int gridSize = std::cbrt(numParticles);
    double       step     = 2.0 / (gridSize - 1);
    unsigned int index    = 0;
    for (unsigned int i = 0; i < gridSize; ++i)
    {
        for (unsigned int j = 0; j < gridSize; ++j)
        {
            for (unsigned int k = 0; k < gridSize; ++k)
            {
                if (index >= numParticles) break;
                x[index] = -1.0 + i * step;
                y[index] = -1.0 + j * step;
                z[index] = -1.0 + k * step;
                ++index;
            }
        }
    }
}
