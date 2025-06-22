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

/*!@file
 * @brief utilities for initial condition generation
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <numeric>
#include <span>
#include <string>
#include <vector>

#include "cstone/primitives/gather.hpp"
#include "cstone/sfc/sfc.hpp"
#include "io/ifile_io.hpp"
#include "init/settings.hpp"
#include "cstone/tree/definitions.h"

namespace sphexa
{

//! @brief sort x,y,z coordinates in the unit cube by SFC keys
template<class KeyType, class T>
void sortBySfcKey(std::vector<T>& x, std::vector<T>& y, std::vector<T>& z)
{
    assert(x.size() == y.size());
    assert(y.size() == z.size());
    size_t blockSize = x.size();

    cstone::Box<T> box(0, 1);

    std::vector<KeyType> keys(blockSize);
    computeSfcKeys(x.data(), y.data(), z.data(), cstone::sfcKindPointer(keys.data()), blockSize, box);

    std::vector<cstone::LocalIndex> sfcOrder(blockSize);
    std::iota(begin(sfcOrder), end(sfcOrder), cstone::LocalIndex(0));
    cstone::sort_by_key(begin(keys), end(keys), begin(sfcOrder));

    std::vector<T> buffer(blockSize);
    cstone::gather<cstone::LocalIndex>(sfcOrder, x.data(), buffer.data());
    std::swap(x, buffer);
    cstone::gather<cstone::LocalIndex>(sfcOrder, y.data(), buffer.data());
    std::swap(y, buffer);
    cstone::gather<cstone::LocalIndex>(sfcOrder, z.data(), buffer.data());
    std::swap(z, buffer);
}

//! @brief read x,y,z coordinates from an H5Part file (at step 0)
template<class Vector>
void readTemplateBlock(const std::string& block, IFileReader* reader, Vector& x, Vector& y, Vector& z)
{
    reader->setStep(block, -1, FileMode::independent);
    size_t blockSize = reader->numParticles();
    x.resize(blockSize);
    y.resize(blockSize);
    z.resize(blockSize);

    reader->readField("x", x.data());
    reader->readField("y", y.data());
    reader->readField("z", z.data());

    reader->closeStep();
}

/*!@brief apply fixed boundary conditions to one axis
 *
 * @tparam T    field type
 * @param pos   position data of axis to be applied
 * @param axisMax max coordinate of the box in the axis of pos
 * @param axisMin min coordinate of the box in the axis of pos
 * @param size  number of particles
 */
template<class T, class Th>
void initFixedBoundaries(T* pos, Th* vx, Th* vy, Th* vz, Th* h, T axisMax, T axisMin, size_t size, T thickness)
{

#pragma omp parallel for
    for (size_t i = 0; i < size; i++)
    {
        T distMax = std::abs(axisMax - pos[i]);
        T distMin = std::abs(axisMin - pos[i]);

        if (distMax < 2.0 * h[i] * thickness || distMin < 2.0 * h[i] * thickness)
        {
            vx[i] = 0.0;
            vy[i] = 0.0;
            vz[i] = 0.0;
        }
    }
}

struct
{
    cstone::Vec3<int> x = {1, 0, 0};
    cstone::Vec3<int> y = {0, 1, 0};
    cstone::Vec3<int> z = {0, 0, 1};
} Axis;

/*! @brief add additional particles as a fixed boundary layer
 *
 * @tparam T coordinate floating point type
 * @tparam Th
 * @tparam Vector coordinate vector type
 * @param axis  axis to add particles to, for example Axis::x
 * @param x     x-coordinates of particles
 * @param y     y-coordinates of particles
 * @param z     z-coordinates of particles
 * @param h     smoothing lengths
 * @param box   global bounding box
 * @param thickness thickness of particle layer to be added, in  2 * h per SPH convention
 */
template<class T, class Vector>
void addFixedBoundaryLayer(cstone::Vec3<int> axis, Vector& x, Vector& y, Vector& z, std::vector<T> h, size_t size,
                           cstone::Box<T>& box, T thickness)
{
    int             axisIndex = axis[0] * 1 + axis[1] * 2 + axis[2] * 3;
    cstone::Vec3<T> boxMax    = {box.xmax(), box.ymax(), box.zmax()};
    cstone::Vec3<T> boxMin    = {box.xmin(), box.ymin(), box.zmin()};

    for (int i = 0; i < size; ++i)
    {
        cstone::Vec3<T> X           = {x[i], y[i], z[i]};
        T               distanceMax = std::abs(boxMax[axisIndex - 1] - X[axisIndex - 1]);
        T               distanceMin = std::abs(boxMin[axisIndex - 1] - X[axisIndex - 1]);
        if (distanceMax < 2 * h[i] * thickness)
        {
            X[axisIndex - 1] += 2.0 * distanceMax;
            x.push_back(X[0]);
            y.push_back(X[1]);
            z.push_back(X[2]);
        }
        if (distanceMin < 2 * h[i] * thickness)
        {
            X[axisIndex - 1] -= 2.0 * distanceMin;
            x.push_back(X[0]);
            y.push_back(X[1]);
            z.push_back(X[2]);
        }
    }
}

//! @brief read file attributes into an associative container
void readFileAttributes(InitSettings& settings, const std::string& settingsFile, IFileReader* reader, bool verbose)
{
    if (not settingsFile.empty())
    {
        reader->setStep(settingsFile, -1, FileMode::independent);

        auto fileAttributes = reader->fileAttributes();
        for (const auto& attr : fileAttributes)
        {
            int64_t sz = reader->fileAttributeSize(attr);
            if (sz == 1)
            {
                bool settingRecognized = settings.count(attr);
                settings[attr]         = {};
                reader->fileAttribute(attr, &settings[attr], sz);
                if (reader->rank() == 0 && verbose)
                {
                    if (settingRecognized)
                    {
                        std::cout << "Override setting from " << settingsFile << ": " << attr << " = " << settings[attr]
                                  << std::endl;
                    }
                    else
                    {
                        std::cout << "Setting from " << settingsFile << ": " << attr << " = " << settings[attr]
                                  << " not recognized " << std::endl;
                    }
                }
            }
        }
        reader->closeStep();
    }
}

//! @brief generate particle IDs at the beginning of the simulation initialization
void generateParticleIDs(std::span<uint64_t> id)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    std::vector<uint64_t> ranksLocalParticles(numRanks);
    size_t                localNumRanks = id.size();

    // fill ranksLocalParticles with the number of particles per rank
    MPI_Allgather(&localNumRanks, 1, MpiType<uint64_t>{}, ranksLocalParticles.data(), 1, MpiType<uint64_t>{},
                  MPI_COMM_WORLD);

    std::exclusive_scan(ranksLocalParticles.begin(), ranksLocalParticles.end(), ranksLocalParticles.begin(),
                        uint64_t(0));
    std::iota(id.begin(), id.end(), ranksLocalParticles[rank]);
}

//! @brief Used to read the default values of dataset attributes
class BuiltinReader
{
public:
    using FieldType = util::Reduce<std::variant, util::Map<std::add_pointer_t, IO::Types>>;

    explicit BuiltinReader(InitSettings& attrs)
        : attributes_(attrs)
    {
    }

    [[nodiscard]] static int rank() { return -1; }

    void stepAttribute(const std::string& key, FieldType val, int64_t /*size*/)
    {
        std::visit([this, &key](auto arg) { attributes_[key] = *arg; }, val);
    };

private:
    //! @brief reference to attributes
    InitSettings& attributes_;
};

//! @brief build up an associative container with test case settings
template<class Dataset>
[[nodiscard]] InitSettings buildSettings(Dataset&& d, const InitSettings& testCaseSettings,
                                         const std::string& settingsFile, IFileReader* reader)
{
    InitSettings settings;
    // first layer: class member defaults in code
    BuiltinReader extractor(settings);
    d.hydro.loadOrStoreAttributes(&extractor);

    // second layer: test-case specific settings
    for (const auto& kv : testCaseSettings)
    {
        settings[kv.first] = kv.second;
    }

    // third layer: settings override by file given on commandline (highest precedence)
    readFileAttributes(settings, settingsFile, reader, true);

    return settings;
}

/*!@brief remove every n-th element of a sequence
 * @param   n  factor to reduce density
 * @param   x  a vector
 * @return     vector with 1/n-th the elements of @p v
 */
template<class T>
auto skipElements(size_t n, const std::vector<T>& x)
{
    size_t newSize = x.size() / n;

    std::vector<T> xSmall(newSize);
    for (size_t i = 0; i < newSize; i++)
    {
        xSmall[i] = x[2 * i];
    }

    return xSmall;
}

//! @brief reduce density of @a SFC sorted template glass block (x,y,z) coordinates by factor of @p n
template<class T>
auto makeLessDenseTemplate(size_t n, const std::vector<T>& x, const std::vector<T>& y, const std::vector<T>& z)
{
    assert(x.size() == y.size());
    assert(y.size() == z.size());
    return std::make_tuple(skipElements(n, x), skipElements(n, y), skipElements(n, z));
}


} // namespace sphexa
