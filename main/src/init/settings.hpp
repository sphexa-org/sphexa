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

#include "io/ifile_io.hpp"

namespace sphexa
{

using InitSettings = std::map<std::string, double>;

//! @brief write @p InitSettings as file attributes of a new file @p path
inline void writeSettings(const InitSettings& settings, const std::string& path, IFileWriter* writer)
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
