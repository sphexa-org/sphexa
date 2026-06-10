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
 * @brief Noh implosion simulation data initialization
 *
 * @author Jose A. Escartin <ja.escartin@gmail.com>"
 */

#pragma once

#include <map>

#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/sfc/box.hpp"
#include "sph/eos.hpp"

#include "isim_init.hpp"
#include "grid.hpp"
#include "utils.hpp"

namespace sphexa
{

std::map<std::string, double> nohConstants()
{
    return {{"r0", 0},
            {"r1", 0.5},
            {"mTotal", 1.},
            {"dim", 3},
            {"gamma", 5.0 / 3.0},
            {"rho0", 1.},
            {"u0", 1e-20},
            {"p0", 0.},
            {"vr0", -1.},
            {"cs0", 0.},
            {"minDt", 1e-4},
            {"minDt_m1", 1e-4},
            {"gravConstant", 0.0},
            {"ng0", 100},
            {"ngmax", 150},
            {"mui", 10.}};
}

template<class Dataset>
void initNohFields(Dataset& d, const std::map<std::string, double>& constants)
{
    using Exec      = typename Dataset::Exec;
    auto exec       = cstone::execution::defaultExec<Exec>;
    using T         = typename Dataset::RealType;
    using HydroType = typename Dataset::HydroType;

    double r           = constants.at("r1");
    double totalVolume = 4. * M_PI / 3. * r * r * r;
    double hInit       = std::cbrt(3.0 / (4 * M_PI) * d.ng0 * totalVolume / d.numParticlesGlobal) * 0.5;
    double mPart       = constants.at("mTotal") / d.numParticlesGlobal;

    auto cv    = sph::idealGasCv(d.muiConst, d.gamma);
    auto temp0 = constants.at("u0") / cv;

    cstone::fill(exec, d.m.begin(), d.m.end(), mPart);
    cstone::fill(exec, d.h.begin(), d.h.end(), hInit);
    cstone::fill(exec, d.du_m1.begin(), d.du_m1.end(), 0.0);
    cstone::fill(exec, d.mui.begin(), d.mui.end(), d.muiConst);
    cstone::fill(exec, d.temp.begin(), d.temp.end(), temp0);
    cstone::fill(exec, d.u.begin(), d.u.end(), constants.at("u0"));
    cstone::fill(exec, d.alpha.begin(), d.alpha.end(), d.alphamin);

    generateParticleIDs(d.id, exec);

    auto&& x = cstone::toHost(d.x);
    auto&& y = cstone::toHost(d.y);
    auto&& z = cstone::toHost(d.z);

    std::vector<HydroType> vx(d.vx.size()), vy(d.vy.size()), vz(d.vz.size());

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < d.x.size(); i++)
    {
        T radius = std::sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
        radius   = std::max(radius, T(1e-10));

        vx[i] = constants.at("vr0") * (x[i] / radius);
        vy[i] = constants.at("vr0") * (y[i] / radius);
        vz[i] = constants.at("vr0") * (z[i] / radius);
    }
    d.vx = std::move(vx);
    d.vy = std::move(vy);
    d.vz = std::move(vz);
    cstone::scale(exec, d.vx.data(), d.vx.data() + d.vx.size(), d.x_m1.data(), constants.at("minDt"));
    cstone::scale(exec, d.vy.data(), d.vy.data() + d.vy.size(), d.y_m1.data(), constants.at("minDt"));
    cstone::scale(exec, d.vz.data(), d.vz.data() + d.vz.size(), d.z_m1.data(), constants.at("minDt"));
}

template<class Dataset>
class NohGlassSphere : public ISimInitializer<Dataset>
{
    std::string          glassBlock;
    mutable InitSettings settings_;

public:
    NohGlassSphere(std::string initBlock, std::string settingsFile, IFileReader* reader)
        : glassBlock(std::move(initBlock))
    {
        Dataset d;
        settings_ = buildSettings(d, nohConstants(), settingsFile, reader);
    }

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cbrtNumPart, Dataset& simData,
                                                 IFileReader* reader) const override
    {
        auto& d       = simData.hydro;
        using KeyType = typename Dataset::KeyType;
        using T       = typename Dataset::RealType;

        std::vector<T> xBlock, yBlock, zBlock;
        readTemplateBlock(glassBlock, reader, xBlock, yBlock, zBlock);
        size_t blockSize = xBlock.size();

        int               multi1D      = std::rint(cbrtNumPart / std::cbrt(blockSize));
        cstone::Vec3<int> multiplicity = {multi1D, multi1D, multi1D};

        T              r = settings_.at("r1");
        cstone::Box<T> globalBox(-r, r, cstone::BoundaryType::open);

        auto [keyStart, keyEnd] = equiDistantSfcSegments<KeyType>(rank, numRanks, 100);
        std::vector<T> x, y, z;
        assembleCuboid<T>(keyStart, keyEnd, globalBox, multiplicity, xBlock, yBlock, zBlock, x, y, z);
        cutSphere(r, x, y, z);
        d.x = x; // uploads to GPU if active
        d.y = y;
        d.z = z;

        size_t numParticlesGlobal = d.x.size();
        MPI_Allreduce(MPI_IN_PLACE, &numParticlesGlobal, 1, MpiType<size_t>{}, MPI_SUM, simData.comm);

        d.resize(d.x.size());

        settings_["numParticlesGlobal"] = double(numParticlesGlobal);
        BuiltinWriter attributeSetter(settings_);
        d.loadOrStoreAttributes(&attributeSetter);

        initNohFields(d, settings_);

        return globalBox;
    }

    [[nodiscard]] const InitSettings& constants() const override { return settings_; }
};

} // namespace sphexa
