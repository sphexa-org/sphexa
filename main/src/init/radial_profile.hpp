/*
 * SPH-EXA
 *
 * Copyright (c) 2026 CSCS, ETH Zurich, University of Zurich, University of Basel
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Generic radially symmetric initial particles distributions
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <string>
#include <vector>

#include "cstone/sfc/box.hpp"
#include "io/ifile_io.hpp"
#include "grid.hpp"
#include "utils.hpp"
#include "early_sync.hpp"
#include "isim_init.hpp"

namespace sphexa
{
template<class T, class KeyType>
auto createUniformSphere(int rank, int numRanks, size_t cbrtNumPart, IFileReader* reader, double r_total,
                         const std::string& glassBlock)
{
    std::vector<T> xBlock, yBlock, zBlock;
    readTemplateBlock(glassBlock, reader, xBlock, yBlock, zBlock);
    size_t blockSize = xBlock.size();

    int               multi1D      = rint(cbrtNumPart / std::cbrt(blockSize));
    cstone::Vec3<int> multiplicity = {multi1D, multi1D, multi1D};

    cstone::Box<T> globalBox(-r_total, r_total, cstone::BoundaryType::open);

    auto [keyStart, keyEnd] = equiDistantSfcSegments<KeyType>(rank, numRanks, 100);

    std::vector<T> x, y, z;
    auto           t0 = std::chrono::high_resolution_clock::now();
    assembleCuboid<T>(keyStart, keyEnd, globalBox, multiplicity, xBlock, yBlock, zBlock, x, y, z);
    cutSphere(r_total, x, y, z);
    auto t1 = std::chrono::high_resolution_clock::now();
    if (rank == 0) std::cout << "assembly " << std::chrono::duration<float>(t1 - t0).count() << std::endl;

    return std::make_tuple(globalBox, x, y, z);
}

template<class Dataset, class T, class Vector>
size_t syncAndLoadAttributes(int rank, int numRanks, Dataset& d, MPI_Comm comm, const cstone::Box<T>& globalBox,
                             const Vector& x, const Vector& y, const Vector& z)
{
    using KeyType = Dataset::KeyType;

    size_t numParticlesGlobal = x.size();
    MPI_Allreduce(MPI_IN_PLACE, &numParticlesGlobal, 1, MpiType<size_t>{}, MPI_SUM, comm);

    auto t0 = std::chrono::high_resolution_clock::now();
    d.x     = x; // uploads to GPU if active
    d.y     = y;
    d.z     = z;
    syncCoords<KeyType>(rank, numRanks, numParticlesGlobal, d.x, d.y, d.z, globalBox);
    // 2nd call needed to reduce imbalance, 1st call might not able to fully balance number of particles per rank
    syncCoords<KeyType>(rank, numRanks, numParticlesGlobal, d.x, d.y, d.z, globalBox);
    auto t1 = std::chrono::high_resolution_clock::now();
    if (rank == 0) std::cout << "earlySync " << std::chrono::duration<float>(t1 - t0).count() << std::endl;

    d.resize(d.x.size());
    return numParticlesGlobal;
}

template<class Vector, class F>
void radialTransformation(Vector& x, Vector& y, Vector& z, F&& f)
{
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x.size(); i++)
    {
        auto radius0 = std::sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);

        auto contraction = f(radius0);
        x[i] *= contraction;
        y[i] *= contraction;
        z[i] *= contraction;
    }
}

template<class Dataset>
class RadialProfile : public ISimInitializer<Dataset>
{
    using T       = Dataset::RealType;
    using KeyType = Dataset::KeyType;

    std::string glassBlock_;

protected:
    mutable InitSettings settings_;

public:
    explicit RadialProfile(std::string initBlock, const InitSettings& testCaseSettings, std::string settingsFile,
                           IFileReader* reader)
        : glassBlock_(std::move(initBlock))
        , ISimInitializer<Dataset>(settingsFile)
    {
        Dataset d;
        settings_ = buildSettings(d, testCaseSettings, settingsFile, reader);
    }

    void initAttributes(Dataset& simData) const
    {
        BuiltinWriter attributeSetter(settings_);
        simData.hydro.loadOrStoreAttributes(&attributeSetter);
    }

    //! @brief Create local x,y,z particles. Note: radialTransform is NOT the radial profile, profile and transform
    //!        are related to each other via a differential equation, e.g. profile=1/r -> transform=r->sqrt(r)
    template<class F>
    cstone::Box<T> init(int rank, int nRanks, size_t cbrtNumPart, Dataset& simData, IFileReader* reader, T radius,
                        F&& radialTransform) const
    {
        auto [globalBox, x, y, z] =
            createUniformSphere<T, KeyType>(rank, nRanks, cbrtNumPart, reader, radius, glassBlock_);

        radialTransformation(x, y, z, radialTransform);

        const auto numParticlesGlobal =
            syncAndLoadAttributes(rank, nRanks, simData.hydro, simData.comm, globalBox, x, y, z);

        settings_["numParticlesGlobal"] = double(numParticlesGlobal);
        initAttributes(simData);

        return globalBox;
    }

    [[nodiscard]] const InitSettings& constants() const override { return settings_; }

    using ISimInitializer<Dataset>::init; // just to avoid compiler warnings about hidden init
};

} // namespace sphexa
