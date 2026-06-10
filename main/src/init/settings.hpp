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
 * @brief Translation unit for the simulation initializer library
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <filesystem>
#include <string>

#include "io/arg_parser.hpp"
#include "io/id_tag_setup.hpp"
#include "io/ifile_io.hpp"

namespace sphexa
{

using InitSettings = std::map<std::string, double>;

inline void saveTaggingSetup(const IdTaggingSetup& taggingSetup, IFileWriter* writer)
{
    if(taggingSetup.selSpheres.size() > 0)
    {
        writer->fileAttribute("id_selection_spheres", taggingSetup.selSpheres.data()->data(), 4*taggingSetup.selSpheres.size());
        writer->fileAttribute("id_selection_spheres_group_ids", taggingSetup.sphereGroupIds.data(), taggingSetup.sphereGroupIds.size());
    }
    if(taggingSetup.selList.size() > 0)
    {
        writer->fileAttribute("id_selection_list", taggingSetup.selList.data(), taggingSetup.selList.size());
        writer->fileAttribute("id_selection_list_group_ids", taggingSetup.selListGroupIds.data(), taggingSetup.selListGroupIds.size());
    }
}

inline void saveTaggingOutputSetup(const IdTaggingOutputSetup& idTaggingOutputSetup, IFileWriter* writer)
{
    if (!idTaggingOutputSetup.outFile.empty())
    {
        writer->fileAttribute("o_subset", strBeforeSign(idTaggingOutputSetup.outFile, "."));
    }
    if (!idTaggingOutputSetup.writeFreqStr.empty())
    {
        auto writeFreqValue = std::stod(idTaggingOutputSetup.writeFreqStr);
        writer->fileAttribute("w_subset", &writeFreqValue, 1);
    }
    if (!idTaggingOutputSetup.outputFields.empty())
    {
        // TODO: does this code already exist somewhere else?
        std::string mergeString;
        for(unsigned int i = 0; i < idTaggingOutputSetup.outputFields.size(); ++i)
        {
            mergeString += idTaggingOutputSetup.outputFields[i];
            if(i < idTaggingOutputSetup.outputFields.size() - 1)
            {
                mergeString += ",";
            }
        }
        writer->fileAttribute("f_subset", mergeString);
    }
    if (!idTaggingOutputSetup.writeExtra.empty())
    {
        // TODO: does this code already exist somewhere else?
        std::vector<double> writeExtraValues(idTaggingOutputSetup.writeExtra.size());
        for(unsigned int i = 0; i < idTaggingOutputSetup.writeExtra.size(); ++i)
        {
            writeExtraValues[i] = std::stod(idTaggingOutputSetup.writeExtra[i]);
        }
        writer->fileAttribute("wextra_subset", writeExtraValues.data(), writeExtraValues.size());
    }
}

//! @brief write @p InitSettings and @p IdTaggingSetup as file attributes of a new file @p path
inline void writeSettings(const InitSettings& settings, const IdTaggingSetup& taggingSetup, const IdTaggingOutputSetup& idTaggingOutputSetup,
    const std::string& path, IFileWriter* writer)
{
    if (std::filesystem::exists(path))
    {
        throw std::runtime_error("Cannot write settings: file " + path + " already exists\n");
    }

    writer->addStep(0, 0, path);
    for (auto it = settings.cbegin(); it != settings.cend(); ++it)
    {
        writer->fileAttribute(it->first, &(it->second), 1);
    }

    // write id tagging settings
    saveTaggingSetup(taggingSetup, writer);

    // write id tagging output settings
    saveTaggingOutputSetup(idTaggingOutputSetup, writer);

    writer->closeStep();
}

//! @brief Used to initialize particle dataset attributes from builtin named test-cases
class BuiltinWriter
{
public:
    using FieldType = util::Reduce<std::variant, util::Map<std::add_pointer_t, IO::Types>>;

    explicit BuiltinWriter(InitSettings attrs)
        : attributes_(std::move(attrs))
    {
    }

    [[nodiscard]] static int rank() { return -1; }

    void stepAttribute(const std::string& key, FieldType val, int64_t /*size*/)
    {
        std::visit([this, &key](auto arg) { *arg = attributes_.at(key); }, val);
    };

private:
    InitSettings attributes_;
};

} // namespace sphexa
