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

#include <filesystem>
#include <numeric>
#include <span>
#include <string>
#include <vector>

#include "cstone/primitives/gather.hpp"
#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/sfc/sfc.hpp"
#include "io/id_tag_utils.hpp"
#include "io/ifile_io.hpp"
#include "init/settings.hpp"

namespace sphexa
{

//! @brief sort x,y,z coordinates in the unit cube by SFC keys
template<class KeyType, class T>
void sortBySfcKey(std::vector<T>& x, std::vector<T>& y, std::vector<T>& z)
{
    assert(x.size() == y.size() && y.size() == z.size());
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

//! @brief read file attributes into an associative container
inline void readFileAttributes(InitSettings& settings, const std::string& settingsFile, IFileReader* reader, bool verbose)
{
    if (not settingsFile.empty())
    {
        reader->setStep(settingsFile, -1, FileMode::independent);

        auto fileAttributes = reader->fileAttributes();
        for (const auto& attr : fileAttributes)
        {
            // skip tagging related attributes
            if (std::ranges::find(taggingAttributes, attr) != taggingAttributes.end()) continue;

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

//! @brief read tagging related file attributes
inline void readFileTaggingAttributes(const std::string& settingsFile, IFileReader* reader,
                                      std::vector<IdSelectionSphere>& selSpheres, std::vector<unsigned int>& sphereGroupIds,
                                      std::vector<uint64_t>& selList, std::vector<unsigned int>& selListGroupIds)
{
    selSpheres.clear();
    sphereGroupIds.clear();
    selList.clear();
    selListGroupIds.clear();
    if (std::filesystem::exists(settingsFile) && not settingsFile.empty())
    {
        reader->setStep(settingsFile, -1, FileMode::independent);

        auto fileAttributes = reader->fileAttributes();
        // Read sphere selection data
        if(std::ranges::find(fileAttributes, std::string("id_selection_spheres")) != fileAttributes.end())
        {
            auto attr_size = reader->fileAttributeSize("id_selection_spheres");
            selSpheres.resize(attr_size/4);// TODO: add safety extra element
            // TODO: this is potentially dangerous, is there a way to check if utils::array has the assumed memory layout?
            // TODO: I'd like to keep all setup consistency check in a separate function but I think this is the right place to
            // check if the attribute size is multiple of 4. I could assign a default value to selSpheres here to identify
            // uninitialized data in the check function but I'm not sure if this is better.
            std::cout<<"WARNING: reading id_selection_spheres attribute, make sure that IdSelectionSphere has the expected memory layout!"<<std::endl;
            reader->fileAttribute("id_selection_spheres", selSpheres.data()->data(), attr_size);
        }
        if(std::ranges::find(fileAttributes, std::string("id_selection_spheres_group_ids")) != fileAttributes.end())
        {
            auto attr_size = reader->fileAttributeSize("id_selection_spheres_group_ids");
            sphereGroupIds.resize(attr_size);
            reader->fileAttribute(std::string("id_selection_spheres_group_ids"), sphereGroupIds.data(), attr_size);
        }

        // Read list selection data
        if(std::ranges::find(fileAttributes, std::string("id_selection_list")) != fileAttributes.end())
        {
            auto attr_size = reader->fileAttributeSize("id_selection_list");
            selList.resize(attr_size);
            reader->fileAttribute(std::string("id_selection_list"), selList.data(), attr_size);
        }
        if(std::ranges::find(fileAttributes, std::string("id_selection_list_group_ids")) != fileAttributes.end())
        {
            auto attr_size = reader->fileAttributeSize("id_selection_list_group_ids");
            selListGroupIds.resize(attr_size);
            reader->fileAttribute(std::string("id_selection_list_group_ids"), selListGroupIds.data(), attr_size);
        }

        reader->closeStep();
    }
}


//! @brief generate particle IDs at the beginning of the simulation initialization
template<bool gpu>
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
    cstone::sequenceAcc<gpu>(id.data(), id.data() + id.size(), ranksLocalParticles[rank]);
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

} // namespace sphexa
