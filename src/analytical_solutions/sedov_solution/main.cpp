/*
 * MIT License
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
 *
 * @brief This program generate the analytical sedov solution based in the time and the initial conditions
 *
 * @author Jose A. Escartin <ja.escartin@gmail.com>
 *
 */

#include <string>
#include <iostream>
#include <cmath>
#include <vector>
#include <filesystem>

#include "arg_parser.hpp"
#include "file_utils.hpp"

#include "sedov/sedov_generator.hpp"

#include "sedov_solution.hpp"

using namespace std;
using namespace sphexa;

using Real    = double;
using KeyType = uint64_t;

void printHelp(char* binName);
void writeColumns1D(const std::string& path);

int main(int argc, char** argv)
{
    const ArgParser parser(argc, argv);

    if (parser.exists("-h") || parser.exists("--h") || parser.exists("-help") || parser.exists("--help"))
    {
        printHelp(argv[0]);
        exit(EXIT_SUCCESS);
    }

    // Get command line parameters
    const double time      = parser.getDouble("--time", 0.);
    const string outDir    = parser.getString("--outDir", "./");
    const bool   normalize = parser.exists("--normalize");

    // Get time without rounding
    ostringstream time_long;
    time_long << time;
    string time_str = time_long.str();

    const string solFile =
        parser.exists("--out") ? parser.getString("--out") : outDir + "sedov_solution_" + time_str + ".dat";

    // Calculate and write theoretical solution profile in one dimension
    const size_t dim    = SedovDataGenerator<Real, KeyType>::dim;
    const double r0     = SedovDataGenerator<Real, KeyType>::r0;
    const double r1     = SedovDataGenerator<Real, KeyType>::r1;
    const double eblast = SedovDataGenerator<Real, KeyType>::energyTotal;
    const double gamma  = SedovDataGenerator<Real, KeyType>::gamma;
    const double omega  = SedovDataGenerator<Real, KeyType>::omega;
    const double rho0   = SedovDataGenerator<Real, KeyType>::rho0;
    const double u0     = SedovDataGenerator<Real, KeyType>::u0;
    const double p0     = SedovDataGenerator<Real, KeyType>::p0;
    const double vr0    = SedovDataGenerator<Real, KeyType>::vr0;
    const double cs0    = SedovDataGenerator<Real, KeyType>::cs0;

    double shockFront;
    {
        std::vector<double> rDummy(1, 0.1);
        std::vector<Real>   rho(1), p(1), u(1), vel(1), cs(1);
        shockFront = SedovSolution::sedovSol(
            dim, time, eblast, omega, gamma, rho0, u0, p0, vr0, cs0, rDummy, rho, p, u, vel, cs);
    }

    // Set the positions for calculating the solution
    size_t         nSteps   = 100000;
    size_t         nSamples = nSteps + 2;
    vector<double> rSol(nSamples);

    const double rMax  = 2. * r1;
    const double rStep = (rMax - r0) / nSteps;

    for (size_t i = 0; i < nSteps; i++)
    {
        rSol[i] = (r0 + (0.5 * rStep) + (i * rStep));
    }
    rSol[nSamples - 2] = shockFront;
    rSol[nSamples - 1] = shockFront + 1e-7;
    std::sort(begin(rSol), end(rSol));

    // analytical solution output
    std::vector<Real> rho(nSamples), p(nSamples), u(nSamples), vel(nSamples), cs(nSamples);

    // Calculate theoretical solution
    SedovSolution::sedovSol(dim, time, eblast, omega, gamma, rho0, u0, p0, vr0, cs0, rSol, rho, p, u, vel, cs);

    if (normalize)
    {
        std::for_each(begin(rho), end(rho), [](auto& val) { val /= SedovSolution::rho_shock; });
        std::for_each(begin(u), end(u), [](auto& val) { val /= SedovSolution::u_shock; });
        std::for_each(begin(p), end(p), [](auto& val) { val /= SedovSolution::p_shock; });
        std::for_each(begin(vel), end(vel), [](auto& val) { val /= SedovSolution::vel_shock; });
        std::for_each(begin(cs), end(cs), [](auto& val) { val /= SedovSolution::cs_shock; });
    }

    writeColumns1D(solFile);
    fileutils::writeAscii<Real>(0,
                                nSteps,
                                solFile,
                                true,
                                {rSol.data(), rho.data(), u.data(), p.data(), vel.data(), cs.data()},
                                std::setw(16),
                                std::setprecision(7),
                                std::scientific);

    cout << "Created solution file: '" << solFile << std::endl;

    return EXIT_SUCCESS;
}

void printHelp(char* binName)
{
    printf("\nUsage:\n\n");
    printf("%s [OPTIONS]\n", binName);
    printf("\nWhere possible options are:\n\n");

    printf("\t--time     NUM  \t\t Time where the solution is calculated (secs) [0.]\n\n");

    printf("\t--outPath  PATH \t\t Path to directory where output will be saved [./].\
                \n\t\t\t\t Note that directory must exist and be provided with ending slash.\
                \n\t\t\t\t Example: --outDir /home/user/folderToSaveOutputFiles/\n\n");

    printf("\t--complete FLAG \t\t Calculate the solution for each particle [False]\n");
}

void writeColumns1D(const std::string& path)
{
    ofstream out(path, std::ofstream::out);

    out << setw(16) << "#           01:r"    // Column : position 1D     (Real value)
        << setw(16) << "02:rho"              // Column : density         (Real value)
        << setw(16) << "03:u"                // Column : internal energy (Real value)
        << setw(16) << "04:p"                // Column : pressure        (Real value)
        << setw(16) << "05:vel"              // Column : velocity 1D     (Real value)
        << setw(16) << "06:cs" << std::endl; // Column : sound speed     (Real value)

    out.close();
}
