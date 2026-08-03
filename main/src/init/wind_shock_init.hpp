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
 * @brief Wind shock simulation data initialization
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 * @author Jose A. Escartin <ja.escartin@gmail.com>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 */

#pragma once

#include <map>
#include <cmath>
#include <algorithm>

#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/sfc/box.hpp"
#include "sph/eos.hpp"

#include "isim_init.hpp"
#include "grid.hpp"
#include "utils.hpp"

namespace sphexa
{

InitSettings WindShockConstants()
{
    return {{"r", .125},           {"rSphere", .025},   {"rhoInt", 10.}, {"rhoExt", 1.},     {"uExt", 3. / 2.},
            {"vxExt", 2.7},        {"vyExt", .0},       {"vzExt", .0},   {"dim", 3},         {"gamma", 5. / 3.},
            {"minDt", 1e-10},      {"minDt_m1", 1e-10}, {"Kcour", 0.2},  {"epsilon", 0.},    {"mui", 10.},
            {"gravConstant", 0.0}, {"ng0", 100},        {"ngmax", 150},  {"wind-shock", 1.0}};
}

template<class Dataset>
void initWindShockFields(Dataset& d, const std::map<std::string, double>& constants, double massPart)
{
    using T         = Dataset::RealType;
    using HydroType = Dataset::HydroType;

    T r       = constants.at("r");
    T rSphere = constants.at("rSphere");
    T rhoInt  = constants.at("rhoInt");
    T rhoExt  = constants.at("rhoExt");
    T uExt    = constants.at("uExt");
    T vxExt   = constants.at("vxExt");
    T vyExt   = constants.at("vyExt");
    T vzExt   = constants.at("vzExt");
    T epsilon = constants.at("epsilon");

    T hInt = 0.5 * std::cbrt(3. * d.ng0 * massPart / 4. / M_PI / rhoInt);
    T hExt = 0.5 * std::cbrt(3. * d.ng0 * massPart / 4. / M_PI / rhoExt);

    auto cv = sph::idealGasCv(d.muiConst, d.gamma);
    initFieldsAtRest(d, massPart);

    T uInt = uExt / (rhoInt / rhoExt);

    T k = d.ngmax / r;

    util::array<T, 3> blobCenter{r, r, r};
    auto&&            x = cstone::toHost(d.x);
    auto&&            y = cstone::toHost(d.y);
    auto&&            z = cstone::toHost(d.z);

    std::vector<HydroType> h(d.h.size());
    std::vector<T>         u(d.x.size());
    std::vector<HydroType> vx(d.vx.size(), 0), vy(d.vy.size(), 0), vz(d.vz.size(), 0);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < d.x.size(); i++)
    {
        util::array<T, 3> X{x[i], y[i], z[i]};

        T rPos = std::sqrt(norm2(X - blobCenter));

        if (rPos > rSphere + epsilon)
        {
            if (rPos > rSphere + 2 * hExt)
            {
                // more than two smoothing lengths away from the inner sphere
                h[i] = hExt;
            }
            else
            {
                // reduce smoothing lengths for particles outside, but close to the inner sphere
                h[i] = hInt + 0.5 * (hExt - hInt) * (1. + std::tanh(k * (rPos - rSphere - hExt)));
            }

            u[i]  = uExt;
            vx[i] = vxExt;
            vy[i] = vyExt;
            vz[i] = vzExt;
        }
        else
        {
            h[i] = hInt;
            u[i] = uInt;
        }
    }
    d.h  = std::move(h);
    d.vx = std::move(vx);
    d.vy = std::move(vy);
    d.vz = std::move(vz);
    cstone::scale(d.exec, d.vx.data(), d.vx.data() + d.vx.size(), d.x_m1.data(), constants.at("minDt"));
    cstone::scale(d.exec, d.vy.data(), d.vy.data() + d.vy.size(), d.y_m1.data(), constants.at("minDt"));
    cstone::scale(d.exec, d.vz.data(), d.vz.data() + d.vz.size(), d.z_m1.data(), constants.at("minDt"));

    if (d.u.empty())
    {
        std::for_each(u.begin(), u.end(), [cvm1 = 1.0 / cv](auto& t) { t *= cvm1; });
        d.temp = std::move(u);
    }
    else { d.u = std::move(u); }
}

template<class Dataset>
class WindShockGlass : public ISimInitializer<Dataset>
{
    std::string          glassBlock;
    mutable InitSettings settings_;

public:
    WindShockGlass(std::string initBlock, std::string settingsFile, IFileReader* reader)
        : ISimInitializer<Dataset>(settingsFile)
        , glassBlock(std::move(initBlock))
    {
        Dataset d;
        settings_ = buildSettings(d, WindShockConstants(), settingsFile, reader);
    }

    cstone::Box<typename Dataset::RealType> initImpl(int rank, int numRanks, size_t cbrtNumPart, Dataset& simData,
                                                     IFileReader* reader) const override
    {
        auto& d       = simData.hydro;
        using KeyType = typename Dataset::KeyType;
        using T       = typename Dataset::RealType;

        T r       = settings_.at("r");
        T rSphere = settings_.at("rSphere");
        T rhoInt  = settings_.at("rhoInt");
        T rhoExt  = settings_.at("rhoExt");

        T densityRatio   = rhoInt / rhoExt;
        T cubeVolume     = std::pow(2 * r, 3);
        T blobMultiplier = std::cbrt(cubeVolume / densityRatio) / (2. * rSphere);

        std::vector<T> xBlock, yBlock, zBlock;
        readTemplateBlock(glassBlock, reader, xBlock, yBlock, zBlock);
        size_t blockSize = xBlock.size();

        int               multi1D          = std::rint(cbrtNumPart / std::cbrt(blockSize));
        cstone::Vec3<int> surroundingMulti = {4 * multi1D, multi1D, multi1D};

        auto           pbc = cstone::BoundaryType::periodic;
        cstone::Box<T> globalBox(0, 8 * r, 0, 2 * r, 0, 2 * r, pbc, pbc, pbc);

        auto [keyStart, keyEnd] = equiDistantSfcSegments<KeyType>(rank, numRanks, 100);
        std::vector<T> x, y, z;
        assembleCuboid<T>(keyStart, keyEnd, globalBox, surroundingMulti, xBlock, yBlock, zBlock, x, y, z);

        auto cutSphereOut = [r, rSphere](auto x, auto y, auto z)
        {
            using T_ = decltype(x);
            util::array<T_, 3> X{x, y, z};
            util::array<T_, 3> center{r, r, r};
            return std::sqrt(norm2(X - center)) > rSphere;
        };
        selectParticles(x, y, z, cutSphereOut);

        // create the high-density blob
        cstone::Vec3<int> blobMulti = {multi1D, multi1D, multi1D};
        std::vector<T>    xBlob, yBlob, zBlob;
        cstone::Box<T>    boxS(r - blobMultiplier * rSphere, r + blobMultiplier * rSphere);
        assembleCuboid<T>(keyStart, keyEnd, boxS, blobMulti, xBlock, yBlock, zBlock, xBlob, yBlob, zBlob);
        auto keepSphere = [r, rSphere](auto x, auto y, auto z)
        {
            using T_ = decltype(x);
            util::array<T_, 3> X{x, y, z};
            util::array<T_, 3> center{r, r, r};
            return std::sqrt(norm2(X - center)) < rSphere;
        };
        selectParticles(xBlob, yBlob, zBlob, keepSphere);
        std::copy(xBlob.begin(), xBlob.end(), std::back_inserter(x));
        std::copy(yBlob.begin(), yBlob.end(), std::back_inserter(y));
        std::copy(zBlob.begin(), zBlob.end(), std::back_inserter(z));

        // Calculate particle mass with the internal sphere
        T innerSide   = rSphere;
        T innerVolume = (4. / 3.) * M_PI * innerSide * innerSide * innerSide;

        size_t numParticlesInternal = xBlob.size();
        MPI_Allreduce(MPI_IN_PLACE, &numParticlesInternal, 1, MpiType<size_t>{}, MPI_SUM, simData.comm);
        T massPart = innerVolume * rhoInt / numParticlesInternal;

        size_t numParticlesGlobal = x.size();
        MPI_Allreduce(MPI_IN_PLACE, &numParticlesGlobal, 1, MpiType<size_t>{}, MPI_SUM, simData.comm);

        d.x = x; // uploads to GPU if active
        d.y = y;
        d.z = z;
        syncCoords<KeyType>(rank, numRanks, numParticlesGlobal, d.x, d.y, d.z, globalBox);

        d.resize(d.x.size());

        settings_["numParticlesGlobal"] = double(numParticlesGlobal);
        BuiltinWriter attributeSetter(settings_);
        d.loadOrStoreAttributes(&attributeSetter);

        initWindShockFields(d, settings_, massPart);

        return globalBox;
    }

    [[nodiscard]] const InitSettings& constants() const override { return settings_; }
};

} // namespace sphexa
