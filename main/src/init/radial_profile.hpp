//
// Created by Noah Kubli on 23.03.2026.
//

#pragma once

#include "isim_init.hpp"

namespace sphexa
{
template<class Dataset>
class RadialProfile : public ISimInitializer<Dataset>
{
    std::string glassBlock;

public:
    explicit RadialProfile(std::string initBlock, IFileReader* reader)
        : glassBlock(std::move(initBlock))
    {
    }
    auto createUniformSphere(int rank, int numRanks, size_t cbrtNumPart, IFileReader* reader,
                             const double r_total) const
    {
        using T       = typename Dataset::RealType;
        using KeyType = typename Dataset::KeyType;

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

    size_t syncAndLoadAttributes(int rank, int numRanks, Dataset& simData, const auto& globalBox, const auto& x,
                                 const auto& y, const auto& z) const
    {
        auto& d       = simData.hydro;
        using KeyType = typename Dataset::KeyType;

        size_t numParticlesGlobal = x.size();
        MPI_Allreduce(MPI_IN_PLACE, &numParticlesGlobal, 1, MpiType<size_t>{}, MPI_SUM, simData.comm);

        auto t0 = std::chrono::high_resolution_clock::now();
        d.x     = x; // uploads to GPU if active
        d.y     = y;
        d.z     = z;
        syncCoords<KeyType>(rank, numRanks, numParticlesGlobal, d.x, d.y, d.z, globalBox);
        // 2nd call needed to reduce imbalance, 1st call is not able to fully balance number of particles per rank
        syncCoords<KeyType>(rank, numRanks, numParticlesGlobal, d.x, d.y, d.z, globalBox);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (rank == 0) std::cout << "earlySync " << std::chrono::duration<float>(t1 - t0).count() << std::endl;

        d.resize(d.x.size());
        printf("number of particles: %zu\n", numParticlesGlobal);
        return numParticlesGlobal;
    }

    template<class Vector>
    void radialTransformation(Vector& x, Vector& y, Vector& z, auto&& f) const
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
};

} // namespace sphexa