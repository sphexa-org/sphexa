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
 * @brief Isobaric cube simulation data initialization
 *
 * @author Jose A. Escartin <ja.escartin@gmail.com>
 */

#pragma once

#include <map>
#include <cmath>
#include <algorithm>

#include "cstone/sfc/box.hpp"
#include "sph/eos.hpp"

#include "isim_init.hpp"
#include "grid.hpp"
#include "utils.hpp"

namespace sphexa
{

InitSettings IsobaricCubeConstants()
{
    return {{"r", .25},
            {"rDelta", .25},
            {"dim", 3},
            {"gamma", 5.0 / 3.0},
            {"rhoExt", 1.},
            {"rhoInt", 8.},
            {"pIsobaric", 2.5},
            {"minDt", 1e-4},
            {"minDt_m1", 1e-4},
            {"epsilon", 1e-15},
            {"pairInstability", 0.},
            {"mui", 10.0},
            {"gravConstant", 0.0},
            {"ng0", 100},
            {"ngmax", 150}};
}

template<class Dataset>
void initIsobaricCubeFields(Dataset& d, const std::map<std::string, double>& constants, double massPart)
{
    using T = typename Dataset::RealType;

    T r         = constants.at("r");
    T rhoInt    = constants.at("rhoInt");
    T rhoExt    = constants.at("rhoExt");
    T hInt      = 0.5 * std::pow(3. * d.ng0 * massPart / 4. / M_PI / rhoInt, 1. / 3.);
    T hExt      = 0.5 * std::pow(3. * d.ng0 * massPart / 4. / M_PI / rhoExt, 1. / 3.);
    T pIsobaric = constants.at("pIsobaric");
    T gamma     = constants.at("gamma");
    T uInt      = pIsobaric / (gamma - 1.) / rhoInt;
    T uExt      = pIsobaric / (gamma - 1.) / rhoExt;
    T epsilon   = constants.at("epsilon");

    auto cv = sph::idealGasCv(d.muiConst, d.gamma);

    std::fill(d.m.begin(), d.m.end(), massPart);
    std::fill(d.du_m1.begin(), d.du_m1.end(), 0.0);
    std::fill(d.mui.begin(), d.mui.end(), d.muiConst);
    std::fill(d.alpha.begin(), d.alpha.end(), d.alphamin);
    std::fill(d.vx.begin(), d.vx.end(), 0.0);
    std::fill(d.vy.begin(), d.vy.end(), 0.0);
    std::fill(d.vz.begin(), d.vz.end(), 0.0);

    generateParticleIDs(d.id);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < d.x.size(); i++)
    {
        T xi = std::abs(d.x[i]);
        T yi = std::abs(d.y[i]);
        T zi = std::abs(d.z[i]);

        if ((xi > r + epsilon) || (yi > r + epsilon) || (zi > r + epsilon))
        {
            if ((xi > r + 2 * hExt) || (yi > r + 2 * hExt) || (zi > r + 2 * hExt))
            {
                // more than two smoothing lengths away from the inner cube
                d.h[i] = hExt;
            }
            else
            {
                T dist = std::max({xi - r, yi - r, zi - r});
                // reduce smoothing lengths for particles outside, but close to the inner cube
                d.h[i] = hInt * (1 - dist / (2 * hExt)) + hExt * dist / (2 * hExt);
            }

            d.temp[i] = uExt / cv;
        }
        else
        {
            d.h[i]    = hInt;
            d.temp[i] = uInt / cv;
        }

        d.x_m1[i] = d.vx[i] * constants.at("minDt");
        d.y_m1[i] = d.vy[i] * constants.at("minDt");
        d.z_m1[i] = d.vz[i] * constants.at("minDt");
    }
}

template<class T>
void compressCenterCube(std::span<T> x, std::span<T> y, std::span<T> z, T rInt, T s, T rExt, T epsilon)
{
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x.size(); i++)
    {
        if ((std::abs(x[i]) - s > epsilon) || (std::abs(y[i]) - s > epsilon) || (std::abs(z[i]) - s > epsilon))
        {
            cstone::Vec3<T> X{x[i], y[i], z[i]};

            T scaleFactor = cappedPyramidStretch(X, rInt, s, rExt);

            x[i] *= scaleFactor;
            y[i] *= scaleFactor;
            z[i] *= scaleFactor;
        }
        else
        {
            // particle in inner high-density region get contracted towards the center
            x[i] *= rInt / s;
            y[i] *= rInt / s;
            z[i] *= rInt / s;
        }
    }
}

template<class Dataset>
class IsobaricCubeGlass : public ISimInitializer<Dataset>
{
    std::string          glassBlock;
    mutable InitSettings settings_;

public:
    explicit IsobaricCubeGlass(std::string initBlock, std::string settingsFile, IFileReader* reader)
        : glassBlock(std::move(initBlock))
    {
        Dataset d;
        settings_ = buildSettings(d, IsobaricCubeConstants(), settingsFile, reader);
    }

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cbrtNumPart, Dataset& simData,
                                                 IFileReader* reader) const override
    {
        auto& d       = simData.hydro;
        using KeyType = typename Dataset::KeyType;
        using T       = typename Dataset::RealType;

        T r       = settings_.at("r");
        T rhoInt  = settings_.at("rhoInt");
        T rhoExt  = settings_.at("rhoExt");
        T epsilon = settings_.at("pairInstability");

        std::vector<T> xBlock, yBlock, zBlock;
        readTemplateBlock(glassBlock, reader, xBlock, yBlock, zBlock);
        size_t blockSize = xBlock.size();

        int               multi1D            = std::rint(cbrtNumPart / std::cbrt(blockSize));
        cstone::Vec3<int> multiplicity       = {multi1D, multi1D, multi1D};
        size_t            numParticlesGlobal = multi1D * multi1D * multi1D * blockSize;

        cstone::Box<T> globalBox(-2 * r, 2 * r, cstone::BoundaryType::periodic);
        auto [keyStart, keyEnd] = equiDistantSfcSegments<KeyType>(rank, numRanks, 100);
        assembleCuboid<T>(keyStart, keyEnd, globalBox, multiplicity, xBlock, yBlock, zBlock, d.x, d.y, d.z);

        T s = computeStretchFactor(r, 2 * r, rhoInt / rhoExt);
        compressCenterCube<T>(d.x, d.y, d.z, r, s, 2. * r, epsilon);

        size_t numParticlesInternal = numParticlesGlobal * std::pow(s / (2. * r), 3);

        // Calculate particle mass with the internal cube
        T innerSide   = 2. * r;
        T innerVolume = innerSide * innerSide * innerSide;
        T massPart    = innerVolume * rhoInt / numParticlesInternal;

        // Initialize isobaric cube domain variables
        d.resize(d.x.size());

        settings_["numParticlesGlobal"] = double(numParticlesGlobal);
        BuiltinWriter attributeSetter(settings_);
        d.loadOrStoreAttributes(&attributeSetter);

        initIsobaricCubeFields(d, settings_, massPart);

        return globalBox;
    }

    const std::map<std::string, double>& constants() const override { return settings_; }
};

} // namespace sphexa
