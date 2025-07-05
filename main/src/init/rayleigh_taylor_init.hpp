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
 * @brief Rayleigh Taylor simulation data initialization
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 * @author Jose A. Escartin <ja.escartin@gmail.com>
 * @author Lukas Schmidt
 *
 */

#pragma once

#include "cstone/sfc/box.hpp"
#include "isim_init.hpp"

#include "utils.hpp"
#include "grid.hpp"

namespace sphexa
{

template<class T, class Dataset>
void initRayleighTaylorFields(Dataset& d, const std::map<std::string, double>& constants, T massPart)
{
    T rhoUp         = constants.at("rhoUp");
    T rhoDown       = 1.;
    T firstTimeStep = constants.at("firstTimeStep");
    T omega0        = constants.at("omega0");
    T gamma         = constants.at("gamma");
    T p0            = constants.at("p0");
    T y0            = constants.at("y0");
    T ymax          = constants.at("ySize");
    T ymin          = 0.0;
    T g             = constants.at("gravityConstant");

    size_t ng0   = 100;
    T      hUp   = 0.5 * std::cbrt(3. * ng0 * massPart / 4. / M_PI / rhoUp);
    T      hDown = 0.5 * std::cbrt(3. * ng0 * massPart / 4. / M_PI / rhoDown);

    std::fill(d.m.begin(), d.m.end(), massPart);
    std::fill(d.du_m1.begin(), d.du_m1.end(), 0.0);
    std::fill(d.mue.begin(), d.mue.end(), 2.0);
    std::fill(d.mui.begin(), d.mui.end(), 10.0);
    std::fill(d.alpha.begin(), d.alpha.end(), d.alphamin);
    std::fill(d.vx.begin(), d.vx.end(), 0.0);
    std::fill(d.vz.begin(), d.vz.end(), 0.0);
    std::fill(d.x_m1.begin(), d.x_m1.end(), 0.0);
    std::fill(d.y_m1.begin(), d.y_m1.end(), 0.0);

    d.minDt    = firstTimeStep;
    d.minDt_m1 = firstTimeStep;

    auto cv = sph::idealGasCv(d.muiConst, d.gamma);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < d.x.size(); i++)
    {

        d.vy[i] = omega0 * (1. - std::cos(4 * M_PI * d.x[i])) * (1. - std::cos(4 * M_PI * d.y[i] / 3.));

        if (d.y[i] < y0)
        {
            // to initialize fixed boundary particles to sensible values
            T y_init = std::max(d.y[i], ymin);
            T p      = p0 - rhoDown * (y_init - y0) * g;
            T u      = p / (rhoDown * (gamma - 1.));

            d.h[i]    = hDown;
            d.temp[i] = u / cv;
        }
        else
        {
            // to initialize fixed boundary particles to sensible values
            T y_init = std::min(d.y[i], ymax);
            T p      = p0 - rhoUp * (y_init - y0) * g;
            T u      = p / (rhoUp * (gamma - 1.));

            d.h[i]    = hUp;
            d.temp[i] = u / cv;
        }

        d.y_m1[i] = d.vy[i] * firstTimeStep;
    }
}

std::map<std::string, double> RayleighTaylorConstants()
{
    return {
        {"rhoUp", 2.},        {"gamma", 1.4},        {"firstTimeStep", 1e-6},  {"y0", 0.75},         {"omega0", 0.0025},
        {"ay0", -0.5},        {"blockSize", 0.0625}, {"xSize", 0.5},           {"ySize", 1.5},       {"zSize", 0.0625},
        {"fbcThickness", 8.}, {"p0", 2.5},           {"gravityConstant", 0.5}, {"readPregenIC", .0}, {"RT", 1.}};
}

template<class Dataset>
class RayleighTaylorGlass : public ISimInitializer<Dataset>
{
    std::string          glassBlock;
    mutable InitSettings settings_;

public:
    RayleighTaylorGlass(std::string glassBlock, std::string settingsFile, IFileReader* reader)
        : glassBlock(std::move(glassBlock))
    {
        Dataset d;
        settings_ = buildSettings(d, RayleighTaylorConstants(), settingsFile, reader);
    }

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cbrtNumPart, Dataset& simData,
                                                 IFileReader* reader) const override
    {

        using KeyType = typename Dataset::KeyType;
        using T       = typename Dataset::RealType;
        auto& d       = simData.hydro;

        if (d.propagator != "RT-ve")
        {
            throw std::runtime_error("ERROR: For the Rayleigh Taylor test (--init RT) the SPH propagator has to be "
                                     "RT-ve. Please restart with '--prop RT-ve'\n");
        }

        T rhoUp = settings_.at("rhoUp");

        T    blockSize    = settings_.at("blockSize");
        T    xSize        = settings_.at("xSize");
        T    ySize        = settings_.at("ySize");
        T    zSize        = settings_.at("zSize");
        T    fbcThickness = settings_.at("fbcThickness");
        bool readPregenIC = settings_.at("readPregenIC") == 1.;

        cstone::Box<T> globalBox(0, xSize, 0, ySize, 0, zSize, cstone::BoundaryType::periodic,
                                 cstone::BoundaryType::fixed, cstone::BoundaryType::periodic);

        if (readPregenIC)
        {
            reader->setStep(glassBlock, -1, FileMode::collective);
            size_t blockSize = reader->numParticles();
            d.x.resize(blockSize);
            d.y.resize(blockSize);
            d.z.resize(blockSize);

            reader->readField("x", d.x.data());
            reader->readField("y", d.y.data());
            reader->readField("z", d.z.data());

            reader->closeStep();
        }
        else
        {

            int xBlocks = xSize / blockSize;
            int yBlocks = ySize / blockSize;
            int zBlocks = zSize / blockSize;

            size_t nBlocks    = xBlocks * yBlocks * zBlocks;
            size_t halfBlocks = nBlocks / 2;

            std::vector<T> xBlock, yBlock, zBlock;
            readTemplateBlock(glassBlock, reader, xBlock, yBlock, zBlock);

            unsigned level             = cstone::log8ceil<KeyType>(100 * numRanks);
            auto     initialBoundaries = cstone::initialDomainSplits<KeyType>(numRanks, level);
            KeyType  keyStart          = initialBoundaries[rank];
            KeyType  keyEnd            = initialBoundaries[rank + 1];

            int               multi1D      = std::lround(cbrtNumPart / std::cbrt(xBlock.size()));
            cstone::Vec3<int> multiplicity = {xBlocks * multi1D, yBlocks / 2 * multi1D, multi1D};

            cstone::Box<T> layer1(0, xSize, 0, ySize / 2., 0, zSize, cstone::BoundaryType::periodic,
                                  cstone::BoundaryType::periodic, cstone::BoundaryType::periodic);
            cstone::Box<T> layer2(0, xSize, ySize / 2., ySize, 0, zSize, cstone::BoundaryType::periodic,
                                  cstone::BoundaryType::periodic, cstone::BoundaryType::periodic);

            assembleCuboid<T>(keyStart, keyEnd, layer2, multiplicity, xBlock, yBlock, zBlock, d.x, d.y, d.z);

            std::vector<T> xStretch, yStretch, zStretch;
            assembleCuboid<T>(keyStart, keyEnd, layer1, multiplicity, xBlock, yBlock, zBlock, xStretch, yStretch,
                              zStretch);

            T    stretch  = std::cbrt(rhoUp);
            auto inLayer1 = [b = layer1](T x, T y, T z)
            { return x >= b.xmin() && x < b.xmax() && y >= b.ymin() && y < b.ymax() && z >= b.zmin() && z < b.zmax(); };

            for (size_t i = 0; i < xStretch.size(); i++)
            {
                cstone::Vec3<T> X{xStretch[i], yStretch[i], zStretch[i]};
                X *= stretch;
                if (inLayer1(X[0], X[1], X[2]))
                {
                    d.x.push_back(X[0]);
                    d.y.push_back(X[1]);
                    d.z.push_back(X[2]);
                }
            }
        }

        size_t numParticlesGlobal = d.x.size();
        MPI_Allreduce(MPI_IN_PLACE, &numParticlesGlobal, 1, MpiType<size_t>{}, MPI_SUM, simData.comm);

        syncCoords<KeyType>(rank, numRanks, numParticlesGlobal, d.x, d.y, d.z, globalBox);

        d.resize(d.x.size());

        settings_["numParticlesGlobal"] = double(numParticlesGlobal);
        BuiltinWriter attributeSetter(settings_);
        d.loadOrStoreAttributes(&attributeSetter);

        T volumeHD = xSize * settings_.at("y0") * zSize; // (x_size * y_size * z_size) in the high-density zone
        T particleFractionHD =
            rhoUp / (rhoUp + 1) * numParticlesGlobal;           // number of particles in the high density region
        T particleMass = volumeHD * rhoUp / particleFractionHD; // mass per particle to match given density ratio

        initRayleighTaylorFields(d, settings_, particleMass);

        return globalBox;
    }

    [[nodiscard]] const InitSettings& constants() const override { return settings_; }
};

} // namespace sphexa
