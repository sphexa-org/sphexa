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
 * @brief output time and energies each iteration (default)
 * @author Lukas Schmidt
 */

#include "conserved_quantities.hpp"
#include "iobservables.hpp"
#include "io/file_utils.hpp"

namespace sphexa
{

template<class Dataset>
class TimeAndEnergy : public IObservables<Dataset>
{
    std::ostream& constantsFile;
    using T = typename Dataset::RealType;

public:
    explicit TimeAndEnergy(std::ostream& constPath)
        : constantsFile(constPath)
    {
    }

    void computeAndWrite(Dataset& simData, size_t firstIndex, size_t lastIndex, const cstone::Box<T>& /*box*/) override
    {
        int rank;
        MPI_Comm_rank(simData.comm, &rank);
        auto& d = simData.hydro;

        computeConservedQuantities(firstIndex, lastIndex, d, simData.comm);

        if (rank == 0)
        {
            fileutils::writeColumns(constantsFile, ' ', d.iteration, d.ttot, d.minDt, d.etot, d.ecin, d.eint, d.egrav,
                                    d.linmom, d.angmom);
        }
    }
};

} // namespace sphexa
