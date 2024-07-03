/*
 * MIT License
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *               2024 University of Basel
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
 * @brief Initialization of the Triple-point Shock test
 *
 * @author Lukas Schmidt
 */

#pragma once

#include "cstone/sfc/box.hpp"
#include "cstone/tree/definitions.h"
#include "isim_init.hpp"
#include "grid.hpp"
#include "sph/eos.hpp"
#include "utils.hpp"
#include <cmath>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace sphexa
{

template<class T, class Dataset>
void initTriplePointFields(Dataset& d, const std::map<std::string, double>& constants, T particleMass)
{

    using Th = typename Dataset::HydroType;

    T  rho_I     = constants.at("rho_I");
    T  rho_II    = constants.at("rho_II");
    T  rho_III   = constants.at("rho_III");
    Th gamma_I   = constants.at("gamma_I");
    Th gamma_III = constants.at("gamma_III");
    T  p_I       = constants.at("p_I");
    T  p_II      = constants.at("p_II");
    T  p_III     = constants.at("p_III");

    T h_I  = 0.5 * std::cbrt(3. * d.ng0 * particleMass / 4. / M_PI / constants.at("rho_I"));
    T h_II = 0.5 * std::cbrt(3. * d.ng0 * particleMass / 4. / M_PI / constants.at("rho_II"));

    auto cv_I     = sph::idealGasCv(d.muiConst, gamma_I);
    auto cv_III   = sph::idealGasCv(d.muiConst, gamma_III);
    T    temp_I   = p_I / ((gamma_I - 1.) * rho_I) / cv_I;
    T    temp_II  = p_II / ((gamma_I - 1.) * rho_II) / cv_I;
    T    temp_III = p_III / ((gamma_III - 1.) * rho_III) / cv_III;

    std::fill(d.m.begin(), d.m.end(), particleMass);
    std::fill(d.du_m1.begin(), d.du_m1.end(), 0.0);
    std::fill(d.mui.begin(), d.mui.end(), d.muiConst);
    std::fill(d.alpha.begin(), d.alpha.end(), d.alphamin);

    std::fill(d.vx.begin(), d.vx.end(), 0.0);
    std::fill(d.vy.begin(), d.vy.end(), 0.0);
    std::fill(d.vz.begin(), d.vz.end(), 0.0);

    // general form: d.x_m1[i] = d.vx[i] * firstTimeStep;
    std::fill(d.x_m1.begin(), d.x_m1.end(), 0.0);
    std::fill(d.y_m1.begin(), d.y_m1.end(), 0.0);
    std::fill(d.z_m1.begin(), d.z_m1.end(), 0.0);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < d.x.size(); ++i)
    {
        if (d.x[i] <= 1.) // Region I
        {
            d.h[i]     = h_I;
            d.temp[i]  = temp_I;
            d.gamma[i] = gamma_I;
        }
        else if (d.y[i] <= 1.5) // Region III
        {
            d.h[i]     = h_I;
            d.temp[i]  = temp_III;
            d.gamma[i] = gamma_III;
        }
        else // Region II
        {
            d.h[i]     = h_II;
            d.temp[i]  = temp_II;
            d.gamma[i] = gamma_I;
        }
    }
}

InitSettings TriplePointConstants()
{
    return {{"rho_I", 1.},      {"rho_II", 0.125},
            {"rho_III", 1.},    {"p_I", 1.},
            {"p_II", 0.1},      {"p_III", 0.1},
            {"gamma_I", 1.5},   {"gamma_III", 1.4},
            {"ng0", 100},       {"ngmax", 150},
            {"Kcour", 0.4},     {"minDt", 1e-7},
            {"minDt_m1", 1e-7}, {"triple-point-shock", 1.0}};
}

template<class Dataset>
class TriplePointGlass : public ISimInitializer<Dataset>
{
    std::string          glassBlock;
    mutable InitSettings settings_;

public:
    TriplePointGlass(std::string initBlock, std::string settingsFile, IFileReader* reader)
        : glassBlock(initBlock)
    {
        Dataset d;
        settings_ = buildSettings(d, TriplePointConstants(), settingsFile, reader);
    }

    using T = typename Dataset::RealType;

    cstone::Box<T> init(int rank, int numRanks, size_t cbrtNumPart, Dataset& simData,
                        IFileReader* reader) const override
    {
        using KeyType = typename Dataset::KeyType;
        using T       = typename Dataset::RealType;
        auto& d       = simData.hydro;
        auto  pbc     = cstone::BoundaryType::periodic;

        std::vector<T> xBlock, yBlock, zBlock;
        readTemplateBlock(glassBlock, reader, xBlock, yBlock, zBlock);
        sortBySfcKey<KeyType>(xBlock, yBlock, zBlock);
        auto [keyStart, keyEnd] = equiDistantSfcSegments<KeyType>(rank, numRanks, 100);

        int               multi1D          = std::lround(cbrtNumPart / std::cbrt(xBlock.size()));
        cstone::Vec3<int> multiplicity_I   = {6 * multi1D, 18 * multi1D, multi1D};
        cstone::Vec3<int> multiplicity_III = {36 * multi1D, 9 * multi1D, multi1D};

        T              zCoord = 1 / 6.;
        auto           fbc    = cstone::BoundaryType::fixed;
        cstone::Box<T> globalBox(0., 7., 0., 3., 0., zCoord, fbc, fbc, fbc);

        cstone::Box<T> section_I(0., 1., 0., 3., 0., zCoord, pbc, pbc, pbc);
        cstone::Box<T> section_II(1., 7., 1.5, 3., 0., zCoord, pbc, pbc, pbc);
        cstone::Box<T> section_III(1., 7., 0., 1.5, 0., zCoord, pbc, pbc, pbc);

        assembleCuboid<T>(keyStart, keyEnd, section_III, multiplicity_III, xBlock, yBlock, zBlock, d.x, d.y, d.z);

        T    stretch      = std::cbrt(settings_.at("rho_III") / settings_.at("rho_II"));
        T    yOffset      = section_III.ymax();
        auto inSection_II = [box = section_II](T u, T v, T w) {
            return u >= box.xmin() && u < box.xmax() && v >= box.ymin() && v < box.ymax() && w >= box.zmin() &&
                   w < box.zmax();
        };

        for (size_t i = 0; i < d.x.size(); ++i)
        {
            cstone::Vec3<T> X{d.x[i], d.y[i], d.z[i]};
            X *= stretch;
            X[0] -= stretch;
            X[1] += yOffset;
            if (inSection_II(X[0], X[1], X[2]))
            {
                d.x.push_back(X[0]);
                d.y.push_back(X[1]);
                d.z.push_back(X[2]);
            }
        }

        assembleCuboid<T>(keyStart, keyEnd, section_I, multiplicity_I, xBlock, yBlock, zBlock, d.x, d.y, d.z);

        size_t numParticlesGlobal = d.x.size();
        MPI_Allreduce(MPI_IN_PLACE, &numParticlesGlobal, 1, MpiType<size_t>{}, MPI_SUM, simData.comm);
        syncCoords<KeyType>(rank, numRanks, d.numParticlesGlobal, d.x, d.y, d.z, globalBox);

        d.resize(d.x.size());

        settings_["numParticlesGlobal"] = double(numParticlesGlobal);
        BuiltinWriter attributeSetter(settings_);
        d.loadOrStoreAttributes(&attributeSetter);

        size_t npart_I      = multiplicity_I[0] * multiplicity_I[1] * multiplicity_I[2] * xBlock.size();
        T      volume_I     = section_I.lx() * section_I.ly() * section_I.lz();
        T      particleMass = volume_I * settings_.at("rho_I") / npart_I;

        initTriplePointFields(d, settings_, particleMass);

        return globalBox;
    }

    [[nodiscard]] const InitSettings& constants() const override { return settings_; }
};

} // namespace sphexa
