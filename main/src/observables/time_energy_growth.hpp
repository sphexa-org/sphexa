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
 * @brief output and calculate energies and growth rate for Kelvin-Helmholtz tests
 *        This calculation for the growth rate was taken from McNally et al. ApJSS, 201 (2012)
 *
 * @author Lukas Schmidt
 */

#include "iobservables.hpp"
#include "sph/math.hpp"
#include <mpi.h>
#include "io/ifile_writer.hpp"


namespace sphexa {


template<typename T, class Dataset>
void localGrowthRate(size_t startIndex, size_t endIndex, Dataset& d, T* sumsi, T* sumci, T* sumdi, const cstone::Box<T>& box)
{
    const T* x  =   d.x.data();
    const T* y  =   d.y.data();
    const T* vy =   d.vy.data();
    const T* rho =  d.rho.data();
    const T* m =    d.m.data();
    const T* kx =   d.kx.data();
    const T ybox =  box.ly();

    T sumsiThread = 0.0, sumciThread = 0.0, sumdiThread = 0.0;
#pragma omp parallel for reduction(+ : sumsiThread, sumciThread, sumdiThread)
    for (size_t i = startIndex; i < endIndex; i++)
    {
        T voli = m[i] / (rho[i] * kx[i]);
        T si;
        T ci;
        T di;
        T aux;
        if(y[i] > ybox * 0.5e0)
        {
            aux = std::exp(-4.e0 * PI * std::abs(y[i] - 0.25e0));
        } else {
            aux = std::exp(-4.e0 * PI * std::abs((ybox - y[i]) - 0.25e0));
        }
        si = vy[i] * voli * std::sin(4.e0 * PI * x[i]) * aux;
        ci = vy[i] * voli * std::cos(4.e0 * PI * x[i]) * aux;
        di = voli * aux;

        sumsiThread += si;
        sumciThread += ci;
        sumdiThread += di;
    }

    *sumsi = sumsiThread;
    *sumci = sumciThread;
    *sumdi = sumdiThread;
}

/*! @brief global calculation of the growth rate
*
* @tparam        T            double or float
* @tparam        Dataset
* @tparam        Box
* @param[in]     startIndex   first locally assigned particle index of buffers in @p d
* @param[in]     endIndex     last locally assigned particle index of buffers in @p d
* @param[in]     d            particle data set
* @param[in]     box          bounding box 
*/
template<typename T, class Dataset>
T computeKHGrowthRate(size_t startIndex, size_t endIndex, Dataset& d, const cstone::Box<T>& box)
{
    T sum[3], globalSum[3];
    localGrowthRate(startIndex, endIndex, d, sum + 0, sum + 1, sum + 2, box);

    int rootRank = 0;
    MPI_Reduce(sum, globalSum, 3, MpiType<T>{}, MPI_SUM, rootRank, MPI_COMM_WORLD);
    return 2.e0 * std::sqrt((globalSum[0]/globalSum[2])*(globalSum[0]/globalSum[2]) + (globalSum[1]/globalSum[2])*(globalSum[1]/globalSum[2]));
    ;
}



//! @brief Observables that includes times, energies and Kelvin-Helmholtz growth rate
template<class Dataset>
class TimeEnergyGrowth : public IObservables<Dataset>
{
    std::ofstream& constantsFile;

public:

    TimeEnergyGrowth(std::ofstream& constPath) : constantsFile(constPath){}

    using T = typename Dataset::RealType;
    

    void computeAndWrite(Dataset& d, size_t firstIndex, size_t lastIndex,
                    cstone::Box<T>& box)
    {
        T khgr = computeKHGrowthRate<T>(firstIndex, lastIndex, d, box);

        int rank;
        MPI_Comm_rank(d.comm, &rank);

        if(rank == 0) 
        {
            fileutils::writeColumns(constantsFile, ' ', d.iteration, d.ttot, d.minDt, d.etot, d.ecin, d.eint, d.egrav, khgr);
        }
    }
    
};







} //namespace sphexa
