/*
 * SPH-EXA
 *
 * Copyright (c) 2026 CSCS, ETH Zurich, University of Zurich, University of Basel
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Evrard collapse initialization
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <map>

#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/tree/continuum.hpp"

#include "radial_profile.hpp"

namespace sphexa
{

InitSettings evrardConstants()
{
    return {{"gravConstant", 1.}, {"r", 1.},          {"mTotal", 1.}, {"gamma", 5. / 3.}, {"u0", 0.05},
            {"minDt", 1e-4},      {"minDt_m1", 1e-4}, {"mui", 10},    {"ng0", 100},       {"ngmax", 150}};
}

template<class Dataset>
void initEvrardFields(Dataset& d, const InitSettings& constants)
{
    using Exec = typename Dataset::Exec;
    auto exec  = cstone::execution::defaultExec<Exec>;

    initFieldsAtRest(d, constants.at("mTotal") / d.numParticlesGlobal, exec);

    auto cv    = sph::idealGasCv(d.muiConst, d.gamma);
    auto temp0 = constants.at("u0") / cv;
    cstone::fill(exec, d.temp.begin(), d.temp.end(), temp0);
    cstone::fill(exec, d.u.begin(), d.u.end(), constants.at("u0"));

    double totalVolume = 4 * M_PI / 3 * std::pow(constants.at("r"), 3);
    // before the contraction with sqrt(r), the sphere has a constant particle concentration of Ntot / Vtot
    // after shifting particles towards the center by factor sqrt(r), the local concentration becomes
    // c(r) = 2/3 * 1/r * Ntot / Vtot
    double c0 = 2. / 3. * d.numParticlesGlobal / totalVolume;

    auto&&                                   x = cstone::toHost(d.x);
    auto&&                                   y = cstone::toHost(d.y);
    auto&&                                   z = cstone::toHost(d.z);
    std::vector<typename Dataset::HydroType> h(d.x.size());
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < d.x.size(); i++)
    {
        auto radius        = std::sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
        auto concentration = c0 / radius;
        h[i]               = std::cbrt(3 / (4 * M_PI) * d.ng0 / concentration) * 0.5;
    }

    d.h = std::move(h);
}

//! @brief Estimate SFC partition of the Evrard sphere based on approximate continuum particle counts
template<class KeyType, class T>
std::tuple<KeyType, KeyType> estimateEvrardSfcPartition(size_t cbrtNumPart, const cstone::Box<T>& box, int rank,
                                                        int numRanks)
{
    size_t numParticlesGlobal = 0.523 * cbrtNumPart * cbrtNumPart * cbrtNumPart;
    T      r                  = box.xmax();

    double   eps        = 2.0 * r / (1u << cstone::maxTreeLevel<KeyType>{});
    unsigned bucketSize = numParticlesGlobal / (100 * numRanks);

    auto oneOverR = [numParticlesGlobal, r, eps](T x, T y, T z)
    {
        T radius = std::max(std::sqrt(norm2(cstone::Vec3<T>{x, y, z})), eps);
        if (radius > r) { return 0.0; }
        else { return T(numParticlesGlobal) / (2 * M_PI * radius); }
    };

    auto [tree, counts] = cstone::computeContinuumCsarray<KeyType>(oneOverR, box, bucketSize);
    auto a              = cstone::makeSfcAssignment(numRanks, counts, tree.data());

    return {a[rank], a[rank + 1]};
}

template<class Dataset>
class EvrardGlassSphere : public RadialProfile<Dataset>
{
    using Base = RadialProfile<Dataset>;
    using Base::settings_;

public:
    explicit EvrardGlassSphere(std::string initBlock, std::string settingsFile, IFileReader* reader)
        : Base(std::move(initBlock), evrardConstants(), std::move(settingsFile), reader)
    {
    }

    cstone::Box<typename Dataset::RealType> initImpl(int rank, int numRanks, size_t cbrtNumPart, Dataset& simData,
                                                     IFileReader* reader) const override
    {
        auto radialTransform = [](auto r) { return std::sqrt(r); };
        auto globalBox = Base::init(rank, numRanks, cbrtNumPart, simData, reader, settings_.at("r"), radialTransform);
        initEvrardFields(simData.hydro, settings_);
        return globalBox;
    }
};

} // namespace sphexa
