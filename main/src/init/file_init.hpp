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
 * @brief Simulation data initialization from an HDF5 file
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/sfc/box.hpp"

#include "isim_init.hpp"

namespace sphexa
{

template<class Dataset>
void restoreDataset(IFileReader* reader, Dataset& d)
{
    d.loadOrStoreAttributes(reader);
    d.resize(reader->localNumParticles());

    auto fieldPointers = d.data();
    for (size_t i = 0; i < fieldPointers.size(); ++i)
    {
        if (d.isConserved(i))
        {
            if (reader->rank() == 0) { std::cout << "restoring " << d.fieldNames[i]; }
            auto t0 = std::chrono::high_resolution_clock::now();
            std::visit(
                [reader, key = d.fieldNames[i]](auto field)
                {
                    using T = std::remove_reference<decltype(*field->data())>::type;
                    std::vector<T> tmp(field->size());
                    reader->readField(Dataset::prefix + key, tmp.data());
                    *field = std::move(tmp);
                },
                fieldPointers[i]);
            MPI_Barrier(MPI_COMM_WORLD);
            auto  t1       = std::chrono::high_resolution_clock::now();
            int   typeSize = std::visit([](auto field) { return sizeof(*field->data()); }, fieldPointers[i]);
            float readTime = std::chrono::duration<float>(t1 - t0).count();
            if (reader->rank() == 0)
            {
                float sizeGB = float(typeSize) * reader->globalNumParticles() / 1024 / 1024 / 1024;
                std::cout << ", " << sizeGB << " GB in " << readTime << " s, " << sizeGB / readTime << " GB/s"
                          << std::endl;
            }
        }
    }
}

template<class SimulationData>
auto restoreData(IFileReader* reader, SimulationData& simData)
{
    using T = typename SimulationData::RealType;

    cstone::Box<T> box(0, 1);
    box.loadOrStore(reader);

    restoreDataset(reader, simData.hydro);
    restoreDataset(reader, simData.chem);

    return box;
}

template<class Dataset>
class FileInit : public ISimInitializer<Dataset>
{
    InitSettings settings_;
    std::string  h5_fname;
    int          initStep = -1;

public:
    explicit FileInit(const std::string& fname, int initStep_, IFileReader* reader)
        : h5_fname(fname)
        , initStep(initStep_)
    {
        // Read file attributes and put them in settings_ such that they propagate to the new output after a restart
        readFileAttributes(settings_, h5_fname, reader, false);
    }

    cstone::Box<typename Dataset::RealType> init(int /* rank */, int /* numRanks */, size_t /* n */, Dataset& simData,
                                                 IFileReader* reader) const override
    {
        reader->setStep(h5_fname, initStep, FileMode::collective);
        auto box = restoreData(reader, simData);
        reader->closeStep();
        return box;
    }

    [[nodiscard]] const InitSettings& constants() const override { return settings_; }
};

template<class Dataset>
class FileSplitInit : public ISimInitializer<Dataset>
{
    InitSettings settings_;
    std::string  h5_fname;
    int          numSplits;

public:
    explicit FileSplitInit(const std::string& fname, int numSplits_, IFileReader* reader)
        : h5_fname(fname)
        , numSplits(numSplits_)
    {
        if (numSplits < 1)
        {
            throw std::runtime_error("Number of particle splits must be a positive integer. Provided value: " +
                                     std::to_string(numSplits));
        }
        // Read file attributes and put them in constants_ such that they propagate to the new output after a restart
        readFileAttributes(settings_, h5_fname, reader, false);
    }

    cstone::Box<typename Dataset::RealType> init(int /* rank */, int, size_t, Dataset& simData,
                                                 IFileReader* reader) const override
    {
        constexpr auto stream = cstone::Execution<typename Dataset::AcceleratorType>::Default();
        reader->setStep(h5_fname, -1, FileMode::collective);

        size_t numParticlesInFile = reader->localNumParticles();
        size_t numParticlesSplit  = numParticlesInFile * numSplits;

        using KeyType = typename Dataset::KeyType;
        using T       = typename Dataset::RealType;
        cstone::Box<T> box(0, 1);
        box.loadOrStore(reader);

        auto& d = simData.hydro;
        d.loadOrStoreAttributes(reader);

        d.numParticlesGlobal = reader->globalNumParticles() * numSplits;
        d.iteration          = 1;
        d.ttot               = 0.0;
        d.minDt /= (100 * numSplits);
        d.minDt_m1 /= (100 * numSplits);

        d.x.resize(numParticlesSplit);
        d.y.resize(numParticlesSplit);
        d.z.resize(numParticlesSplit);
        d.h.resize(numParticlesSplit);

        std::vector<cstone::LocalIndex> sfcOrder(numParticlesInFile);
        {
            std::vector<T> x0(numParticlesInFile), y0(numParticlesInFile), z0(numParticlesInFile),
                tmp(numParticlesInFile);
            reader->readField("x", x0.data());
            reader->readField("y", y0.data());
            reader->readField("z", z0.data());

            std::vector<KeyType> keys(numParticlesInFile);
            cstone::computeSfcKeys(x0.data(), y0.data(), z0.data(), cstone::sfcKindPointer(keys.data()),
                                   numParticlesInFile, box);
            std::iota(sfcOrder.begin(), sfcOrder.end(), 0);
            cstone::sort_by_key(keys.begin(), keys.end(), sfcOrder.begin());

            auto gatherSwap = [&tmp](auto& v, auto& order)
            {
                cstone::gather<cstone::LocalIndex>(order, v.data(), tmp.data());
                swap(v, tmp);
            };
            gatherSwap(x0, sfcOrder);
            gatherSwap(y0, sfcOrder);
            gatherSwap(z0, sfcOrder);

            std::vector<T> x(numParticlesSplit);
            std::vector<T> y(numParticlesSplit);
            std::vector<T> z(numParticlesSplit);
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < numParticlesInFile; ++i)
            {
                size_t sIdx = numSplits * i;

                x[sIdx] = x0[i];
                y[sIdx] = y0[i];
                z[sIdx] = z0[i];

                bool isLast   = (i == numParticlesInFile - 1);
                long keyDelta = (isLast ? -(keys[i] - keys[i - 1]) : keys[i + 1] - keys[i]) / (numSplits + isLast);

                for (int j = 1; j < numSplits; ++j)
                {
                    auto [ixj, iyj, izj] = cstone::decodeSfc(cstone::sfcKey(keys[i] + j * keyDelta));

                    x[sIdx + j] = box.xmin() + (ixj * box.lx()) / cstone::maxCoord<KeyType>{};
                    y[sIdx + j] = box.ymin() + (iyj * box.ly()) / cstone::maxCoord<KeyType>{};
                    z[sIdx + j] = box.zmin() + (izj * box.lz()) / cstone::maxCoord<KeyType>{};
                }
            }
            d.x = std::move(x);
            d.y = std::move(y);
            d.z = std::move(z);
        }

        auto replicateField = [&sfcOrder, numParticlesInFile, numParticlesSplit,
                               this](IFileReader* reader, const std::string& key, auto& dest, T scale)
        {
            std::vector<T> src(numParticlesInFile), tmp(numParticlesInFile);
            reader->readField(key, src.data());
            cstone::gather<cstone::LocalIndex>(sfcOrder, src.data(), tmp.data());
            swap(src, tmp);
            tmp.clear();

            using DestVectorType = std::decay_t<decltype(dest)>::value_type;
            std::vector<DestVectorType> outTmp(numParticlesSplit);
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < numParticlesInFile; ++i)
            {
                size_t sIdx = numSplits * i;
                std::fill(outTmp.data() + sIdx, outTmp.data() + sIdx + numSplits, src[i] * scale);
            }
            dest = std::move(outTmp);
        };

        d.resize(numParticlesSplit);
        replicateField(reader, "m", d.m, T(1) / numSplits);
        replicateField(reader, "h", d.h, T(1) / std::cbrt(numSplits));
        replicateField(reader, "vx", d.vx, T(1));
        replicateField(reader, "vy", d.vy, T(1));
        replicateField(reader, "vz", d.vz, T(1));
        if (d.isAllocated("temp")) { replicateField(reader, "temp", d.temp, T(1)); }
        else if (d.isAllocated("u")) { replicateField(reader, "u", d.u, T(1)); }
        cstone::fill(d.du_m1.begin(), d.du_m1.end(), 0, stream);
        cstone::fill(d.rung.begin(), d.rung.end(), 0, stream);
        cstone::scaleGpuAcc(d.vx.data(), d.vx.data() + d.vx.size(), d.x_m1.data(), d.minDt, stream);
        cstone::scaleGpuAcc(d.vy.data(), d.vy.data() + d.vy.size(), d.y_m1.data(), d.minDt, stream);
        cstone::scaleGpuAcc(d.vz.data(), d.vz.data() + d.vz.size(), d.z_m1.data(), d.minDt, stream);

        generateParticleIDs(d.id, stream);

        if (d.isAllocated("alpha"))
        {
            try
            {
                replicateField(reader, "alpha", d.alpha, T(1));
            }
            catch (std::runtime_error&)
            {
                cstone::fill(d.alpha.begin(), d.alpha.end(), d.alphamin, stream);
            }
        }

        reader->closeStep();

        return box;
    }

    [[nodiscard]] const InitSettings& constants() const override { return settings_; }
};

} // namespace sphexa
