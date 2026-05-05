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
 * @brief Sedov blast simulation data initialization
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <map>

#include "cstone/sfc/box.hpp"
#include "sph/eos.hpp"

#include "isim_init.hpp"
#include "early_sync.hpp"
#include "grid.hpp"
#include "utils.hpp"

namespace sphexa
{

std::map<std::string, double> StarRelaxationConstants()
{
    return {{"rSun", 6.96e10},  // radius of the sun
            {"mSun", 1.969e33}, // solar mass
            {"dim", 3},
            {"minDt", 1e-6},
            {"minDt_m1", 1e-6},
            {"gravConstant", 6.6726e-8},
            {"eosChoice", sph::EosType::helmholtz},
            {"ng0", 100},
            {"ngmax", 150}};
}

template<class Dataset>
void readICFile(Dataset& d, IFileReader* reader, std::string ICFile)
{
    reader->setStep(ICFile, -1, FileMode::collective);
    size_t numParticles  = reader->numParticles();
    d.numParticlesGlobal = numParticles;

    d.x.resize(numParticles);
    d.y.resize(numParticles);
    d.z.resize(numParticles);
    d.h.resize(numParticles);
    d.temp.resize(numParticles);
    d.abar.resize(numParticles);
    d.zbar.resize(numParticles);

    // Read IC file: Columns are: x,y,z,h,temp,abar,zbar, and li3 abundance.
    reader->readField("x", d.x.data());
    reader->readField("y", d.y.data());
    reader->readField("z", d.z.data());
    reader->readField("h", d.h.data());
    reader->readField("temp", d.temp.data());
    reader->readField("abar", d.abar.data());
    reader->readField("zbar", d.zbar.data());
    // reader->readField("li3", d.li3.data());

    reader->closeStep();
}

template<class Dataset>
void initStarRelaxationFields(Dataset& d, const std::map<std::string, double>& constants)
{
    using T = typename Dataset::RealType;

    double mPart = 1.0 * constants.at("mSun") / d.numParticlesGlobal;

    std::fill(d.m.begin(), d.m.end(), mPart);
    d.mui.resize(d.zbar.size());

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < d.zbar.size(); i++)
    {
        double ye = std::max(1e-16, d.zbar[i] / d.abar[i]);
        d.mui[i]  = 1.0 / (1.0 / d.abar[i] + ye); // type match
    }

    std::fill(d.du_m1.begin(), d.du_m1.end(), 0.0);
    std::fill(d.alpha.begin(), d.alpha.end(), d.alphamin);

    std::fill(d.vx.begin(), d.vx.end(), 0.0);
    std::fill(d.vy.begin(), d.vy.end(), 0.0);
    std::fill(d.vz.begin(), d.vz.end(), 0.0);

    // general form: d.x_m1[i] = d.vx[i] * firstTimeStep;
    std::fill(d.x_m1.begin(), d.x_m1.end(), 0.0);
    std::fill(d.y_m1.begin(), d.y_m1.end(), 0.0);
    std::fill(d.z_m1.begin(), d.z_m1.end(), 0.0);

    generateParticleIDs(d.id);

    // If temperature is not allocated, we can still use this initializer for just the coordinates
    if (d.temp.empty() && d.u.empty()) { return; }

    // #pragma omp parallel for schedule(static)
    //     for (size_t i = 0; i < d.x.size(); i++)
    //     {
    //         T xi = d.x[i];
    //         T yi = d.y[i];
    //         T zi = d.z[i];
    //         T r2 = xi * xi + yi * yi + zi * zi;

    //         T ui = constants.at("ener0") * exp(-(r2 / width2)) + constants.at("u0");
    //         if (d.temp.empty()) { d.u[i] = ui; }
    //         else { d.temp[i] = ui / cv; }
    //     }
}

template<class Dataset>
class StarRelaxation : public ISimInitializer<Dataset>
{
    std::string          ICFile;
    mutable InitSettings settings_;

public:
    StarRelaxation(std::string ICFile, std::string settingsFile, IFileReader* reader)
        : ICFile(std::move(ICFile))
    {
        Dataset d;
        settings_ = buildSettings(d, StarRelaxationConstants(), settingsFile, reader);
    }

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cubeSide, Dataset& simData,
                                                 IFileReader* reader) const override
    {
        auto& d       = simData.hydro;
        using KeyType = typename Dataset::KeyType;
        using T       = typename Dataset::RealType;
        // size_t numParticlesGlobal = cubeSide * cubeSide * cubeSide;

        auto [first, last] = partitionRange(d.numParticlesGlobal, rank, numRanks);
        d.resize(last - first);

        T              r = settings_.at("rSun") * 1.3;
        cstone::Box<T> globalBox(-r, r, cstone::BoundaryType::open);
        // read from file here
        readICFile(d, reader, ICFile);

        syncCoords<KeyType>(rank, numRanks, d.numParticlesGlobal, d.x, d.y, d.z, globalBox);

        settings_["numParticlesGlobal"] = double(d.numParticlesGlobal);
        BuiltinWriter attributeSetter(settings_);
        d.loadOrStoreAttributes(&attributeSetter);

        initStarRelaxationFields(d, settings_);

        return globalBox;
    }

    [[nodiscard]] const InitSettings& constants() const override { return settings_; }
};

} // namespace sphexa
