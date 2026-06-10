/*
 * MIT License
 *
 * Copyright (c) 2023 CSCS, ETH Zurich
 *               2023 University of Basel
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
 * @brief Gresho-Chan vortex simulation data initialization
 *
 * @author Lukas Schmidt
 */

#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/sfc/box.hpp"
#include "isim_init.hpp"
#include "grid.hpp"
#include "utils.hpp"
#include "sph/eos.hpp"

namespace sphexa
{

InitSettings GreshoChanSettings()
{
    return {{"R1", 0.2}, {"v0", 1.},     {"P0", 5.},   {"gamma", 5. / 3.}, {"minDt", 1e-7},       {"minDt_m1", 1e-7},
            {"rho", 1},  {"Kcour", 0.2}, {"ng0", 100}, {"ngmax", 150},     {"gravConstant", 0.0}, {"gresho-chan", 1.0}};
}

template<class Dataset, class T>
void initGreshoChanFields(Dataset& d, const std::map<std::string, double>& settings, T mPart)
{
    using Exec           = typename Dataset::Exec;
    auto stream          = Exec::Default();
    using HydroType      = Dataset::HydroType;
    using RealType       = Dataset::RealType;
    double ng0           = settings.at("ng0");
    double rho           = settings.at("rho");
    double hInit         = 0.5 * std::cbrt(3. * ng0 * mPart / 4. / M_PI / rho);
    double firstTimeStep = settings.at("minDt");

    d.gamma    = settings.at("gamma");
    d.minDt    = firstTimeStep;
    d.minDt_m1 = firstTimeStep;

    auto   cv = sph::idealGasCv(d.muiConst, d.gamma);
    double R1 = settings.at("R1");
    double v0 = settings.at("v0");
    double P0 = settings.at("P0");

    initFieldsAtRest(d, mPart, stream);
    cstone::fill(d.h.begin(), d.h.end(), hInit, stream);

    auto&& x = toHost(d.x);
    auto&& y = toHost(d.y);

    cstone::LocalIndex     numPartLoc = d.x.size();
    std::vector<RealType>  u(numPartLoc);
    std::vector<HydroType> vx(numPartLoc), vy(numPartLoc);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < d.x.size(); ++i)
    {
        T vi, pi;
        T xi    = x[i];
        T yi    = y[i];
        T psi   = std::sqrt(norm2(util::array<T, 2>{xi, yi})) / R1;
        T theta = std::atan2(yi, xi);

        if (psi <= 1.)
        {
            pi = P0 + 4 * v0 * v0 * psi * psi / 8;
            vi = v0 * psi;
        }
        else if (psi <= 2.)
        {
            pi = P0 + 4 * v0 * v0 * (psi * psi / 8 - psi + std::log(psi) + 1);
            vi = v0 * (2 - psi);
        }
        else
        {
            pi = P0 + 4 * v0 * v0 * (std::log(2) - 0.5);
            vi = 0.0;
        }

        u[i]  = pi / ((d.gamma - 1.) * rho);
        vx[i] = -1.0 * vi * std::sin(theta);
        vy[i] = vi * std::cos(theta);
    }
    d.vx = std::move(vx);
    d.vy = std::move(vy);
    cstone::scaleGpuAcc(d.vx.data(), d.vx.data() + d.vx.size(), d.x_m1.data(), firstTimeStep, stream);
    cstone::scaleGpuAcc(d.vy.data(), d.vy.data() + d.vy.size(), d.y_m1.data(), firstTimeStep, stream);

    if (d.temp.empty()) { d.u = std::move(u); }
    else
    {
        std::for_each(u.begin(), u.end(), [cvm1 = 1.0 / cv](auto& t) { t *= cvm1; }); // convert to temperature
        d.temp = std::move(u);
    }
}

template<class Dataset>
class GreshoChan : public ISimInitializer<Dataset>
{
    std::string          glassBlock;
    mutable InitSettings settings_;

public:
    GreshoChan(std::string initBlock, std::string settingsFile, IFileReader* reader)
        : glassBlock(std::move(initBlock))
    {
        Dataset d;
        settings_ = buildSettings(d, GreshoChanSettings(), settingsFile, reader);
    }

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cbrtNumPart, Dataset& simData,
                                                 IFileReader* reader) const override
    {
        using KeyType = typename Dataset::KeyType;
        using T       = typename Dataset::RealType;
        auto& d       = simData.hydro;

        std::vector<T> xBlock, yBlock, zBlock;
        readTemplateBlock(glassBlock, reader, xBlock, yBlock, zBlock);
        size_t            blockSize          = xBlock.size();
        int               multi1D            = std::rint(cbrtNumPart / std::cbrt(blockSize));
        cstone::Vec3<int> multiplicity       = {9 * multi1D, 9 * multi1D, multi1D};
        size_t            numParticlesGlobal = multiplicity[0] * multiplicity[1] * multiplicity[2] * blockSize;

        auto           pbc = cstone::BoundaryType::periodic;
        cstone::Box<T> globalBox(-0.5, 0.5, -0.5, 0.5, -0.0555, 0.0555, pbc, pbc, pbc);

        unsigned level             = cstone::log8ceil<KeyType>(100 * numRanks);
        auto     initialBoundaries = cstone::initialDomainSplits<KeyType>(numRanks, level);
        KeyType  keyStart          = initialBoundaries[rank];
        KeyType  keyEnd            = initialBoundaries[rank + 1];

        std::vector<T> x, y, z;
        assembleCuboid<T>(keyStart, keyEnd, globalBox, multiplicity, xBlock, yBlock, zBlock, x, y, z);
        d.x = x; // uploads to GPU if active
        d.y = y;
        d.z = z;
        d.resize(d.x.size());

        settings_["numParticlesGlobal"] = double(numParticlesGlobal);
        BuiltinWriter attributeSetter(settings_);
        d.loadOrStoreAttributes(&attributeSetter);

        T massPart = globalBox.lx() * globalBox.ly() * globalBox.lz() * settings_.at("rho") / d.numParticlesGlobal;
        initGreshoChanFields(d, settings_, massPart);

        return globalBox;
    }

    const InitSettings& constants() const override { return settings_; }
};

} // namespace sphexa
