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
 * @brief Kelvin Helmholtz simulation data initialization
 *
 * @author Lukas Schmidt
 */

#pragma once

#include "cstone/sfc/box.hpp"
#include "cstone/sfc/sfc.hpp"
#include "cstone/primitives/gather.hpp"
#include "io/mpi_file_utils.hpp"
#include "isim_init.hpp"
#include "utils.hpp"

#include "grid.hpp"

namespace sphexa
{

template<class T, class Dataset>
void initKelvinHelmholtzFields(Dataset& d, const std::map<std::string, double>& constants, T massPart)
{
    T rhoInt        = constants.at("rhoInt");
    T rhoExt        = constants.at("rhoExt");
    T firstTimeStep = constants.at("firstTimeStep");
    T omega0        = constants.at("omega0");
    T gamma         = constants.at("gamma");
    T p             = constants.at("p");
    T vxInt         = constants.at("vxInt");
    T vxExt         = constants.at("vxExt");

    T uInt = p / ((gamma - 1.) * rhoInt);
    T uExt = p / ((gamma - 1.) * rhoExt);
    T vDif = 0.5 * (vxExt - vxInt);
    T ls   = 0.025;

    size_t ng0  = 100;
    T      hInt = 0.5 * std::cbrt(3. * ng0 * massPart / 4. / M_PI / rhoInt);
    T      hExt = 0.5 * std::cbrt(3. * ng0 * massPart / 4. / M_PI / rhoExt);

    std::fill(d.m.begin(), d.m.end(), massPart);
    std::fill(d.du_m1.begin(), d.du_m1.end(), 0.0);
    std::fill(d.mue.begin(), d.mue.end(), 2.0);
    std::fill(d.mui.begin(), d.mui.end(), 10.0);
    std::fill(d.alpha.begin(), d.alpha.end(), d.alphamax);
    std::fill(d.vz.begin(), d.vz.end(), 0.0);

    d.minDt    = firstTimeStep;
    d.minDt_m1 = firstTimeStep;

    auto cv = sph::idealGasCv(d.muiConst, gamma);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < d.x.size(); i++)
    {
        d.vy[i] = omega0 * std::sin(4 * M_PI * d.x[i]);

        if (d.y[i] < 0.75 && d.y[i] > 0.25)
        {
            d.h[i]    = hInt;
            d.temp[i] = uInt / cv;
            if (d.y[i] > 0.5) { d.vx[i] = vxInt + vDif * std::exp((d.y[i] - 0.75) / ls); }
            else { d.vx[i] = vxInt + vDif * std::exp((0.25 - d.y[i]) / ls); }
        }
        else
        {
            d.h[i]    = hExt;
            d.temp[i] = uExt / cv;
            if (d.y[i] < 0.25) { d.vx[i] = vxExt - vDif * std::exp((d.y[i] - 0.25) / ls); }
            else { d.vx[i] = vxExt - vDif * std::exp((0.75 - d.y[i]) / ls); }
        }

        d.x_m1[i] = d.vx[i] * firstTimeStep;
        d.y_m1[i] = d.vy[i] * firstTimeStep;
        d.z_m1[i] = d.vz[i] * firstTimeStep;
    }
}

std::map<std::string, double> KelvinHelmholtzConstants()
{
    return {{"rhoInt", 2.},     {"rhoExt", 1.},          {"vxExt", 0.5}, {"vxInt", -0.5},
            {"gamma", 5. / 3.}, {"firstTimeStep", 1e-7}, {"p", 2.5},     {"omega0", 0.01}};
}

template<class Dataset>
class KelvinHelmholtzGlass : public ISimInitializer<Dataset>
{
    std::string                   glassBlock;
    std::map<std::string, double> constants_;

public:
    KelvinHelmholtzGlass(std::string initBlock)
        : glassBlock(initBlock)
    {
        constants_ = KelvinHelmholtzConstants();
    }

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cbrtNumPart,
                                                 Dataset& simData) const override
    {
        using KeyType = typename Dataset::KeyType;
        using T       = typename Dataset::RealType;
        auto& d       = simData.hydro;

        T rhoInt = constants_.at("rhoInt");

        std::vector<T> xBlock, yBlock, zBlock;
        fileutils::readTemplateBlock(glassBlock, xBlock, yBlock, zBlock);
        size_t blockSize = xBlock.size();

        cstone::Box<T> globalBox(0, 1, 0, 1, 0, 0.0625, cstone::BoundaryType::periodic, cstone::BoundaryType::periodic,
                                 cstone::BoundaryType::periodic);
        auto [keyStart, keyEnd] = partitionRange(cstone::nodeRange<KeyType>(0), rank, numRanks);

        auto [xHalf, yHalf, zHalf] = makeLessDenseTemplate<T, Dataset>(2, xBlock, yBlock, zBlock, blockSize);

        int               multi1D    = std::rint(cbrtNumPart / std::cbrt(blockSize));
        cstone::Vec3<int> innerMulti = {16 * multi1D, 8 * multi1D, multi1D};
        cstone::Vec3<int> outerMulti = {16 * multi1D, 4 * multi1D, multi1D};

        cstone::Box<T> layer1(0, 1, 0, 0.25, 0, 0.0625, cstone::BoundaryType::periodic, cstone::BoundaryType::periodic,
                              cstone::BoundaryType::periodic);
        cstone::Box<T> layer2(0, 1, 0.25, 0.75, 0, 0.0625, cstone::BoundaryType::periodic,
                              cstone::BoundaryType::periodic, cstone::BoundaryType::periodic);
        cstone::Box<T> layer3(0, 1, 0.75, 1, 0, 0.0625, cstone::BoundaryType::periodic, cstone::BoundaryType::periodic,
                              cstone::BoundaryType::periodic);

        assembleCuboid<T>(keyStart, keyEnd, layer1, outerMulti, xHalf, yHalf, zHalf, d.x, d.y, d.z);
        assembleCuboid<T>(keyStart, keyEnd, layer2, innerMulti, xBlock, yBlock, zBlock, d.x, d.y, d.z);
        assembleCuboid<T>(keyStart, keyEnd, layer3, outerMulti, xHalf, yHalf, zHalf, d.x, d.y, d.z);

        size_t npartInner   = 128 * xBlock.size();
        T      volumeHD     = 0.5 * 0.0625;
        T      particleMass = volumeHD * rhoInt / npartInner;

        // size_t totalNPart = 128 * (xBlock.size() + xHalf.size());
        d.resize(d.x.size());
        initKelvinHelmholtzFields(d, constants_, particleMass);

        d.numParticlesGlobal = d.x.size();
        MPI_Allreduce(MPI_IN_PLACE, &d.numParticlesGlobal, 1, MpiType<size_t>{}, MPI_SUM, simData.comm);

        return globalBox;
    }

    const std::map<std::string, double>& constants() const override { return constants_; }
};

} // namespace sphexa