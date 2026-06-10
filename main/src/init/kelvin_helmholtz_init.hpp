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

#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/sfc/box.hpp"

#include "grid.hpp"
#include "utils.hpp"
#include "early_sync.hpp"
#include "isim_init.hpp"

namespace sphexa
{

InitSettings KelvinHelmholtzConstants()
{
    return {{"rhoInt", 2.},        {"rhoExt", 1.},           {"vxExt", 0.5},
            {"vxInt", -0.5},       {"gamma", 5. / 3.},       {"p", 2.5},
            {"omega0", 0.01},      {"Kcour", 0.2},           {"ng0", 100},
            {"ngmax", 150},        {"minDt", 1e-7},          {"minDt_m1", 1e-7},
            {"gravConstant", 0.0}, {"kelvin-helmholtz", 1.0}};
}

template<class T, class Dataset>
void initKelvinHelmholtzFields(Dataset& d, const InitSettings& constants, T massPart)
{
    constexpr auto stream = cstone::Stream<typename Dataset::AcceleratorType>::Default();
    using HydroType       = Dataset::HydroType;
    T rhoInt              = constants.at("rhoInt");
    T rhoExt              = constants.at("rhoExt");
    T omega0              = constants.at("omega0");
    T gamma               = constants.at("gamma");
    T p                   = constants.at("p");
    T vxInt               = constants.at("vxInt");
    T vxExt               = constants.at("vxExt");

    T uInt = p / ((gamma - 1.) * rhoInt);
    T uExt = p / ((gamma - 1.) * rhoExt);
    T vDif = 0.5 * (vxExt - vxInt);
    T ls   = 0.025;

    T hInt = 0.5 * std::cbrt(3. * d.ng0 * massPart / 4. / M_PI / rhoInt);
    T hExt = 0.5 * std::cbrt(3. * d.ng0 * massPart / 4. / M_PI / rhoExt);

    initFieldsAtRest(d, massPart, stream);
    cstone::fill(d.mue.begin(), d.mue.end(), 2.0, stream);
    cstone::fill(d.mui.begin(), d.mui.end(), 10.0, stream);

    auto   cv = sph::idealGasCv(d.muiConst, gamma);
    auto&& x  = toHost(d.x);
    auto&& y  = toHost(d.y);

    cstone::LocalIndex     numPartLocal = d.x.size();
    std::vector<HydroType> h(numPartLocal);
    std::vector<HydroType> vx(numPartLocal), vy(numPartLocal);
    std::vector<T>         u(numPartLocal);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < d.x.size(); i++)
    {
        vy[i] = omega0 * std::sin(4 * M_PI * x[i]);

        if (y[i] < 0.75 && y[i] > 0.25)
        {
            h[i] = hInt;
            u[i] = uInt;
            if (y[i] > 0.5) { vx[i] = vxInt + vDif * std::exp((y[i] - 0.75) / ls); }
            else
            {
                vx[i] = vxInt + vDif * std::exp((0.25 - y[i]) / ls);
            }
        }
        else
        {
            if (y[i] > 0.75 + 2 * hExt || y[i] < 0.25 - 2 * hExt)
            {
                // more than two smoothing lengths away from the high density band
                h[i] = hExt;
            }
            else
            {
                T dist = (y[i] > 0.75) ? y[i] - 0.75 : 0.25 - y[i];
                // linear interpolation from hInt to hExt for particles within 2 * hExt of the high density band
                h[i] = hInt * (1 - dist / (2 * hExt)) + hExt * dist / (2 * hExt);
            }

            u[i] = uExt;
            if (y[i] < 0.25) { vx[i] = vxExt - vDif * std::exp((y[i] - 0.25) / ls); }
            else
            {
                vx[i] = vxExt - vDif * std::exp((0.75 - y[i]) / ls);
            }
        }
    }
    d.h  = std::move(h);
    d.vx = std::move(vx);
    d.vy = std::move(vy);
    cstone::scaleGpuAcc(d.vx.data(), d.vx.data() + d.vx.size(), d.x_m1.data(), constants.at("minDt"), stream);
    cstone::scaleGpuAcc(d.vy.data(), d.vy.data() + d.vy.size(), d.y_m1.data(), constants.at("minDt"), stream);

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

        std::vector<T> xl1, yl1, zl1;
        assembleCuboid<T>(keyStart, keyEnd, layer1, outerMulti, xBlock, yBlock, zBlock, xl1, yl1, zl1);

        T stretch = std::cbrt(settings_.at("rhoInt") / settings_.at("rhoExt"));
        T topEdge = layer3.ymax();

        auto inLayer1 = [b = layer1](T u, T v, T w)
        { return u >= b.xmin() && u < b.xmax() && v >= b.ymin() && v < b.ymax() && w >= b.zmin() && w < b.zmax(); };

        std::vector<T> x, y, z;
        for (size_t i = 0; i < xl1.size(); ++i)
        {
            cstone::Vec3<T> X{xl1[i], yl1[i], zl1[i]};
            // double the volume of layer1 to halve the density
            X *= stretch;
            // crop layer1 back to original size
            if (inLayer1(X[0], X[1], X[2]))
            {
                x.push_back(X[0]);
                y.push_back(X[1]);
                z.push_back(X[2]);
                // layer3: reflect (to preserve the relaxed PBC surface in y direction) and translate
                T yLayer3 = -X[1] + topEdge;
                x.push_back(X[0]);
                y.push_back(yLayer3);
                z.push_back(X[2]);
            }
        }

        assembleCuboid<T>(keyStart, keyEnd, layer2, innerMulti, xBlock, yBlock, zBlock, x, y, z);
        d.x = x; // uploads to GPU if active
        d.y = y;
        d.z = z;

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
