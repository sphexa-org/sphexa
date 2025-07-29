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

#include "cstone/sfc/box.hpp"
#include "isim_init.hpp"
#include "grid.hpp"
#include "sph/eos.hpp"

namespace sphexa
{

InitSettings GreshoChanSettings()
{
    return {{"R1", 0.2},         {"v0", 1.}, {"P0", 5.},     {"gamma", 5. / 3.}, {"mTotal", 1.}, {"minDt", 1e-7},
            {"minDt_m1", 1e-7},  {"rho", 1}, {"Kcour", 0.2}, {"ng0", 100},       {"ngmax", 150}, {"gravConstant", 0.0},
            {"gresho-chan", 1.0}};
}

template<class T>
double twoDimRadius(T x, T y)
{
    return std::sqrt(x * x + y * y);
}

template<class Dataset, class T>
void initGreshoChanFields(Dataset& d, const std::map<std::string, double>& settings, T mPart)
{
    double ng0 = settings.at("ng0");
    double rho = settings.at("rho");
    // double mPart         = settings.at("mTotal") / d.numParticlesGlobal;
    double hInit         = 0.5 * std::cbrt(3. * ng0 * mPart / 4. / M_PI / rho);
    double firstTimeStep = settings.at("minDt");

    d.gamma    = settings.at("gamma");
    d.minDt    = firstTimeStep;
    d.minDt_m1 = firstTimeStep;

    auto   cv = sph::idealGasCv(d.muiConst, d.gamma);
    double R1 = settings.at("R1");
    double v0 = settings.at("v0");
    double P0 = settings.at("P0");

    std::fill(d.m.begin(), d.m.end(), mPart);
    std::fill(d.du_m1.begin(), d.du_m1.end(), 0.0);
    std::fill(d.h.begin(), d.h.end(), hInit);
    std::fill(d.mui.begin(), d.mui.end(), d.muiConst);
    std::fill(d.alpha.begin(), d.alpha.end(), d.alphamin);

    std::fill(d.z_m1.begin(), d.z_m1.end(), 0.0);

    generateParticleIDs(d.id);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < d.x.size(); ++i)
    {
        T vi, pi;
        T xi    = d.x[i];
        T yi    = d.y[i];
        T psi   = twoDimRadius(xi, yi) / R1;
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

        auto ui = pi / ((d.gamma - 1.) * rho);
        if (d.u.empty()) { d.temp[i] = ui / cv; }
        else { d.u[i] = ui; }
        d.vx[i] = -1.0 * vi * std::sin(theta);
        d.vy[i] = vi * std::cos(theta);
        d.vz[i] = 0.0;

        d.x_m1[i] = d.vx[i] * firstTimeStep;
        d.y_m1[i] = d.vy[i] * firstTimeStep;
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

        assembleCuboid<T>(keyStart, keyEnd, globalBox, multiplicity, xBlock, yBlock, zBlock, d.x, d.y, d.z);
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