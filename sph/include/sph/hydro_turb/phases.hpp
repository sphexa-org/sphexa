/*
 * MIT License
 *
 * Copyright (c) 2022 Politechnical University of Catalonia UPC
 *               2022 University of Basel
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
 * @brief  Implementation of stirring phase update
 * @author Axel Sanz <axel.sanz@estudiantat.upc.edu>
 */

#pragma once

namespace sph
{

/*! @brief Update stirring phases from the OU phases by applying the projection operator
 *
 * @param[in]   numModes      computed number of modes
 * @param[in]   numDim        number of dimensions
 * @param[in]   OUphases      Ornstein-Uhlenbeck phases
 * @param[in]   solWeight     solenoidal weight
 * @param[in]   modes         vector containing modes
 * @param[out]  phasesReal    real part of phases
 * @param[out]  phasesImag    imaginary part of phases
 */
template<class T>
void computePhases(size_t numModes, size_t numDim, const std::vector<T>& OUPhases, T solWeight,
                   const std::vector<T>& modes, std::vector<T>& phasesReal, std::vector<T>& phasesImag)
{
    for (size_t i = 0; i < numModes; i++)
    {
        T ka = 0.0;
        T kb = 0.0;
        T kk = 0.0;
        for (size_t j = 0; j < numDim; j++)
        {
            kk = kk + modes[3 * i + j] * modes[3 * i + j];
            ka = ka + modes[3 * i + j] * OUPhases[6 * i + 2 * j + 1];
            kb = kb + modes[3 * i + j] * OUPhases[6 * i + 2 * j];
        }
        for (size_t j = 0; j < numDim; j++)
        {
            T diva  = modes[3 * i + j] * ka / kk;
            T divb  = modes[3 * i + j] * kb / kk;
            T curla = OUPhases[6 * i + 2 * j] - divb;
            T curlb = OUPhases[6 * i + 2 * j + 1] - diva;

            phasesReal[3 * i + j] = solWeight * curla + (1.0 - solWeight) * divb;
            phasesImag[3 * i + j] = solWeight * curlb + (1.0 - solWeight) * diva;
        }
    }
}

} // namespace sph
