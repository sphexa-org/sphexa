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

#include <filesystem>
#include <fstream>
#include <numeric>

#include "density_pdf.hpp"
#include "io/arg_parser.hpp"
#include "io/factory.hpp"
#include "util/utils.hpp"
#include "cstone/primitives/mpi_wrappers.hpp"

using namespace sphexa;
using T = float;

int main(int argc, char** argv)
{
    auto [rank, numRanks] = initMpi();
    const ArgParser parser(argc, (const char**)argv);

    // The default pdf range was taken in comparison to Federrath et al. 2021, DOI 10.1038/s41550-020-01282-z
    const std::string inputFile  = parser.get("--file");
    const size_t      nBins      = parser.get("-n", 50);
    const int         step       = parser.get("-s", -1);
    std::string       outputFile = parser.get("-o", std::string("density_pdf.txt"));
    const std::string sph_type   = parser.get("--sph", std::string("std"));
    const T           minValue   = parser.get("--min", -8.0);
    const T           maxValue   = parser.get("--max", 6.0);

    if (!std::filesystem::exists(inputFile))
    {
        printf("Please provide a existing file: no file found at %s\n", inputFile.c_str());
        return exitSuccess();
    }

    std::vector<T>      rho;
    std::vector<double> bins;

    auto h5reader = fileReaderFactory(false, MPI_COMM_WORLD);
    h5reader->setStep(inputFile, step, FileMode::collective);

    const size_t localNumParticles  = h5reader->localNumParticles();
    const size_t globalNumParticles = h5reader->globalNumParticles();
    if (rank == 0)
    {
        printf("Density-PDF: local particles: %lu \t global particles: %lu\n", localNumParticles, globalNumParticles);
    }

    rho.resize(localNumParticles);

    if (sph_type == "std") { h5reader->readField("rho", rho.data()); }
    else
    {
        std::vector<T> m(localNumParticles);
        std::vector<T> xm(localNumParticles);
        h5reader->readField("kx", rho.data());
        h5reader->readField("xm", xm.data());
        h5reader->readField("m", m.data());
#pragma omp for schedule(static)
        for (size_t i = 0; i < localNumParticles; ++i)
        {
            rho[i] = rho[i] * m[i] / xm[i];
        }
    }
    rho.shrink_to_fit();
    if (rho.size() != localNumParticles)
    {
        throw std::runtime_error("rho length doesn't match local count: " + std::to_string(rho.size()) + "\t" +
                                 std::to_string(localNumParticles));
    }

    h5reader->closeStep();

    double localTotalDensity = 0.0;
#pragma omp parallel for reduction(+ : localTotalDensity)
    for (size_t i = 0; i < localNumParticles; ++i)
    {
        localTotalDensity += rho[i];
    }

    printf("rank %i, local average  density: %f \n", rank, localTotalDensity / localNumParticles);
    double referenceDensity = 0.0;
    MPI_Allreduce(&localTotalDensity, &referenceDensity, 1, MpiType<double>{}, MPI_SUM, MPI_COMM_WORLD);
    referenceDensity /= globalNumParticles;

    if (rank == 0) { printf("starting PDF calculation with reference density %f\n", referenceDensity); }

    bins = computeProbabilityDistribution(rho, referenceDensity, nBins, minValue, maxValue);
    std::vector<double> reduced_bins(nBins, 0.0);
    MPI_Reduce(bins.data(), reduced_bins.data(), nBins, MpiType<double>{}, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        T binSize = (maxValue - minValue) / nBins;
        std::for_each(reduced_bins.begin(), reduced_bins.end(),
                      [globalNumParticles, binSize](double& i) { i /= globalNumParticles * binSize; });
        std::ofstream outFile(std::filesystem::path(outputFile), std::ofstream::out);

        T firstMiddle = minValue + 0.5 * binSize;

        // header line containing metadata
        outFile << nBins << ' ' << binSize << ' ' << referenceDensity << std::endl;

        for (size_t i = 0; i < nBins; i++)
        {
            T binCenter = binSize * i + firstMiddle;
            outFile << binCenter << ' ' << reduced_bins[i] << std::endl;
        }
        printf("Calculated PDF for %lu particles in %lu bins.\n", globalNumParticles, nBins);
    }

    exitSuccess();
}
