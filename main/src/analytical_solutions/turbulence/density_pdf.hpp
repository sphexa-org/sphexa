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
#include <vector>
#include <algorithm>
#include <math.h>

template<class T>
std::vector<double> computeProbabilityDistribution(std::vector<T>& data, const double referenceValue, size_t binCount,
                                                   T binStart, T binEnd)
{
    std::vector<double> bins(binCount);

#pragma omp for schedule(static)
    for (size_t i = 0; i < data.size(); ++i)
    {
        data[i] = std::log(data[i] / referenceValue);
    }

    T binSize = (binEnd - binStart) / binCount;
    for (size_t bin = 0; bin < binCount; bin++)
    {
        bins[bin] = std::count_if(data.begin(), data.end(), [bin, binStart, binSize](T i)
                                  { return i > binSize * bin + binStart && i <= binSize * (bin + 1) + binStart; });
    }
    return bins;
}
