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

#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/sfc/box.hpp"
#include "sph/eos.hpp"

#include "isim_init.hpp"
#include "sedov_constants.hpp"
#include "early_sync.hpp"
#include "grid.hpp"
#include "utils.hpp"

namespace sphexa
{

template<class Dataset>
void initSedovFields(Dataset& d, const InitSettings& constants)
{
    using Exec  = typename Dataset::Exec;
    auto exec = Exec::Default();
    using T     = Dataset::RealType;

    double r           = constants.at("r1");
    double totalVolume = std::pow(2 * r, 3);
    double hInit       = std::cbrt(3.0 / (4 * M_PI) * d.ng0 * totalVolume / d.numParticlesGlobal) * 0.5;

    double mPart  = constants.at("mTotal") / d.numParticlesGlobal;
    double width  = T(2) * hInit;
    double width2 = width * width;

    // We distribute energy as exp(-r2 / width2), with width taken as the current 2h, so that the enery
    // is deposited in about ng0 neighbors.
    // ener0 is the constant that should multiply the Gaussian so that its integral equals energytotal
    double ener0 = constants.at("energyTotal") / std::pow(M_PI, 1.5) / width2 / width;

    initFieldsAtRest(d, mPart, exec);
    cstone::fill(exec, d.h.begin(), d.h.end(), hInit);

    auto cv = sph::idealGasCv(d.muiConst, d.gamma);

    // If temperature is not allocated, we can still use this initializer for just the coordinates
    if (d.temp.empty() && d.u.empty()) { return; }

    auto&& x = cstone::toHost(d.x);
    auto&& y = cstone::toHost(d.y);
    auto&& z = cstone::toHost(d.z);

    std::vector<T> u(d.x.size());
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < d.x.size(); i++)
    {
        T r2 = norm2(cstone::Vec3<T>{x[i], y[i], z[i]});
        u[i] = ener0 * exp(-(r2 / width2)) + constants.at("u0");
    }
    if (d.u.empty())
    {
        std::for_each(u.begin(), u.end(), [cvm1 = 1.0 / cv](auto& t) { t *= cvm1; });
        d.temp = std::move(u);
    }
    else
    {
        d.u = std::move(u);
    }
}

template<class Dataset>
class SedovGrid : public ISimInitializer<Dataset>
{
    mutable InitSettings settings_;

public:
    SedovGrid()
    {
        Dataset d;
        settings_ = buildSettings(d, sedovConstants(), {}, nullptr);
    }

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cubeSide, Dataset& simData,
                                                 IFileReader*) const override
    {
        auto& d                   = simData.hydro;
        using KeyType             = typename Dataset::KeyType;
        using T                   = typename Dataset::RealType;
        size_t numParticlesGlobal = cubeSide * cubeSide * cubeSide;

        auto [first, last] = partitionRange(numParticlesGlobal, rank, numRanks);
        d.resize(last - first);

        T              r = settings_.at("r1");
        cstone::Box<T> globalBox(-r, r, cstone::BoundaryType::periodic);
        std::vector<T> x(last - first), y(last - first), z(last - first);
        regularGrid(r, cubeSide, first, last, x, y, z);
        d.x = x; // uploads to GPU if active
        d.y = y;
        d.z = z;
        syncCoords<KeyType>(rank, numRanks, numParticlesGlobal, d.x, d.y, d.z, globalBox);
        d.resize(d.x.size());

        settings_["numParticlesGlobal"] = double(numParticlesGlobal);
        BuiltinWriter attributeSetter(settings_);
        d.loadOrStoreAttributes(&attributeSetter);

        initSedovFields(d, settings_);

        return globalBox;
    }

    [[nodiscard]] const InitSettings& constants() const override { return settings_; }
};

template<class Dataset>
class SedovGlass : public ISimInitializer<Dataset>
{
    std::string          glassBlock;
    mutable InitSettings settings_;

public:
    SedovGlass(std::string initBlock, std::string settingsFile, IFileReader* reader)
        : glassBlock(std::move(initBlock))
    {
        Dataset d;
        settings_ = buildSettings(d, sedovConstants(), settingsFile, reader);
    }

    /*! @brief initialize particle data with a constant density cube
     *
     * @param[in]    rank             MPI rank ID
     * @param[in]    numRanks         number of MPI ranks
     * @param[in]    cbrtNumPart      the cubic root of the global number of particles to generate
     * @param[inout] d                particle dataset
     * @return                        the global coordinate bounding box
     */
    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cbrtNumPart, Dataset& simData,
                                                 IFileReader* reader) const override
    {
        auto& d       = simData.hydro;
        using KeyType = typename Dataset::KeyType;
        using T       = typename Dataset::RealType;

        std::vector<T> xBlock, yBlock, zBlock;
        readTemplateBlock(glassBlock, reader, xBlock, yBlock, zBlock);
        size_t blockSize = xBlock.size();

        int               multi1D            = std::rint(cbrtNumPart / std::cbrt(blockSize));
        cstone::Vec3<int> multiplicity       = {multi1D, multi1D, multi1D};
        size_t            numParticlesGlobal = multi1D * multi1D * multi1D * blockSize;

        T              r = settings_.at("r1");
        cstone::Box<T> globalBox(-r, r, cstone::BoundaryType::periodic);

        auto [keyStart, keyEnd] = equiDistantSfcSegments<KeyType>(rank, numRanks, 100);

        std::vector<T> x, y, z;
        assembleCuboid<T>(keyStart, keyEnd, globalBox, multiplicity, xBlock, yBlock, zBlock, x, y, z);
        d.x = x; // uploads to GPU if active
        d.y = y;
        d.z = z;
        d.resize(d.x.size());

        settings_["numParticlesGlobal"] = double(numParticlesGlobal);
        BuiltinWriter attributeSetter(settings_);
        d.loadOrStoreAttributes(&attributeSetter);

        initSedovFields(d, settings_);

        return globalBox;
    }

    const InitSettings& constants() const override { return settings_; }
};

} // namespace sphexa
