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

#include "isim_init.hpp"
#include "grid.hpp"
#include "utils.hpp"

namespace sphexa
{

InitSettings KelvinHelmholtzConstants()
{
    return {{"rhoInt", 2.},        {"rhoExt", 1.},           {"vxExt", 0.5},
            {"vxInt", -0.5},       {"gamma", 5. / 3.},       {"p", 2.5},
            {"omega0", 0.01},      {"Kcour", 0.4},           {"ng0", 100},
            {"ngmax", 150},        {"minDt", 1e-7},          {"minDt_m1", 1e-7},
            {"gravConstant", 0.0}, {"kelvin-helmholtz", 1.0}};
}

template<class T, class Dataset>
void initKelvinHelmholtzFields(Dataset& d, const std::map<std::string, double>& constants, T massPart)
{
    T rhoInt = constants.at("rhoInt");
    T rhoExt = constants.at("rhoExt");
    T omega0 = constants.at("omega0");
    T gamma  = constants.at("gamma");
    T p      = constants.at("p");
    T vxInt  = constants.at("vxInt");
    T vxExt  = constants.at("vxExt");

    T uInt = p / ((gamma - 1.) * rhoInt);
    T uExt = p / ((gamma - 1.) * rhoExt);
    T vDif = 0.5 * (vxExt - vxInt);
    T ls   = 0.025;

    T hInt = 0.5 * std::cbrt(3. * d.ng0 * massPart / 4. / M_PI / rhoInt);
    T hExt = 0.5 * std::cbrt(3. * d.ng0 * massPart / 4. / M_PI / rhoExt);

    std::fill(d.m.begin(), d.m.end(), massPart);
    std::fill(d.du_m1.begin(), d.du_m1.end(), 0.0);
    std::fill(d.mue.begin(), d.mue.end(), 2.0);
    std::fill(d.mui.begin(), d.mui.end(), 10.0);
    std::fill(d.alpha.begin(), d.alpha.end(), d.alphamax);
    std::fill(d.vz.begin(), d.vz.end(), 0.0);

    generateParticleIDs(d.id);

    auto  cv     = sph::idealGasCv(d.muiConst, gamma);
    auto* u_or_t = d.u.empty() ? d.temp.data() : d.u.data();

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < d.x.size(); i++)
    {
        d.vy[i] = omega0 * std::sin(4 * M_PI * d.x[i]);

        if (d.y[i] < 0.75 && d.y[i] > 0.25)
        {
            d.h[i]    = hInt;
            u_or_t[i] = uInt;
            if (d.y[i] > 0.5) { d.vx[i] = vxInt + vDif * std::exp((d.y[i] - 0.75) / ls); }
            else { d.vx[i] = vxInt + vDif * std::exp((0.25 - d.y[i]) / ls); }
        }
        else
        {
            if (d.y[i] > 0.75 + 2 * hExt || d.y[i] < 0.25 - 2 * hExt)
            {
                // more than two smoothing lengths away from the high density band
                d.h[i] = hExt;
            }
            else
            {
                T dist = (d.y[i] > 0.75) ? d.y[i] - 0.75 : 0.25 - d.y[i];
                // linear interpolation from hInt to hExt for particles within 2 * hExt of the high density band
                d.h[i] = hInt * (1 - dist / (2 * hExt)) + hExt * dist / (2 * hExt);
            }

            u_or_t[i] = uExt;
            if (d.y[i] < 0.25) { d.vx[i] = vxExt - vDif * std::exp((d.y[i] - 0.25) / ls); }
            else { d.vx[i] = vxExt - vDif * std::exp((0.75 - d.y[i]) / ls); }
        }

        d.x_m1[i] = d.vx[i] * d.minDt;
        d.y_m1[i] = d.vy[i] * d.minDt;
        d.z_m1[i] = d.vz[i] * d.minDt;
    }
    if (d.u.empty())
    {
        std::for_each(d.temp.begin(), d.temp.end(), [cvm1 = 1.0 / cv](auto& t) { t *= cvm1; });
    }
}

template<class Dataset>
class KelvinHelmholtzGlass : public ISimInitializer<Dataset>
{
    std::string          glassBlock;
    mutable InitSettings settings_;

public:
    KelvinHelmholtzGlass(std::string initBlock, std::string settingsFile, IFileReader* reader)
        : glassBlock(initBlock)
    {
        Dataset d;
        settings_ = buildSettings(d, KelvinHelmholtzConstants(), settingsFile, reader);
    }

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cbrtNumPart, Dataset& simData,
                                                 IFileReader* reader) const override
    {
        using KeyType = typename Dataset::KeyType;
        using T       = typename Dataset::RealType;
        auto& d       = simData.hydro;
        auto  pbc     = cstone::BoundaryType::periodic;

        std::vector<T> xBlock, yBlock, zBlock;
        readTemplateBlock(glassBlock, reader, xBlock, yBlock, zBlock);
        sortBySfcKey<KeyType>(xBlock, yBlock, zBlock);

        cstone::Box<T> globalBox(0, 1, 0, 1, 0, 0.0625, pbc, pbc, pbc);
        auto [keyStart, keyEnd] = equiDistantSfcSegments<KeyType>(rank, numRanks, 100);

        int               multi1D    = std::lround(cbrtNumPart / std::cbrt(xBlock.size()));
        cstone::Vec3<int> innerMulti = {16 * multi1D, 8 * multi1D, multi1D};
        cstone::Vec3<int> outerMulti = {16 * multi1D, 4 * multi1D, multi1D};

        cstone::Box<T> layer1(0, 1, 0, 0.25, 0, 0.0625, pbc, pbc, pbc);
        cstone::Box<T> layer2(0, 1, 0.25, 0.75, 0, 0.0625, pbc, pbc, pbc);
        cstone::Box<T> layer3(0, 1, 0.75, 1, 0, 0.0625, pbc, pbc, pbc);

        std::vector<T> x, y, z;
        assembleCuboid<T>(keyStart, keyEnd, layer1, outerMulti, xBlock, yBlock, zBlock, x, y, z);

        T stretch = std::cbrt(settings_.at("rhoInt") / settings_.at("rhoExt"));
        T topEdge = layer3.ymax();

        auto inLayer1 = [b = layer1](T u, T v, T w)
        { return u >= b.xmin() && u < b.xmax() && v >= b.ymin() && v < b.ymax() && w >= b.zmin() && w < b.zmax(); };

        for (size_t i = 0; i < x.size(); ++i)
        {
            cstone::Vec3<T> X{x[i], y[i], z[i]};
            // double the volume of layer1 to halve the density
            X *= stretch;
            // crop layer1 back to original size
            if (inLayer1(X[0], X[1], X[2]))
            {
                d.x.push_back(X[0]);
                d.y.push_back(X[1]);
                d.z.push_back(X[2]);
                // layer3: reflect (to preserve the relaxed PBC surface in y direction) and translate
                T yLayer3 = -X[1] + topEdge;
                d.x.push_back(X[0]);
                d.y.push_back(yLayer3);
                d.z.push_back(X[2]);
            }
        }

        assembleCuboid<T>(keyStart, keyEnd, layer2, innerMulti, xBlock, yBlock, zBlock, d.x, d.y, d.z);

        size_t numParticlesGlobal = d.x.size();
        MPI_Allreduce(MPI_IN_PLACE, &numParticlesGlobal, 1, MpiType<size_t>{}, MPI_SUM, simData.comm);
        syncCoords<KeyType>(rank, numRanks, d.numParticlesGlobal, d.x, d.y, d.z, globalBox);

        size_t npartInner   = innerMulti[0] * innerMulti[1] * innerMulti[2] * xBlock.size();
        T      volumeHD     = 0.5 * globalBox.lx() * globalBox.ly() * globalBox.lz();
        T      particleMass = volumeHD * settings_.at("rhoInt") / npartInner;

        d.resize(d.x.size());

        settings_["numParticlesGlobal"] = double(numParticlesGlobal);
        BuiltinWriter attributeSetter(settings_);
        d.loadOrStoreAttributes(&attributeSetter);

        initKelvinHelmholtzFields(d, settings_, particleMass);

        return globalBox;
    }

    [[nodiscard]] const InitSettings& constants() const override { return settings_; }
};

} // namespace sphexa
