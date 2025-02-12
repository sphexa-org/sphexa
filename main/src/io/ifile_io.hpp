/*
 * MIT License
 *
 * Copyright (c) 2023 CSCS, ETH Zurich, University of Basel, University of Zurich
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
 * @brief file I/O interface
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <map>
#include <string>
#include <variant>
#include <vector>

#include "cstone/util/type_list.hpp"

namespace sphexa
{

struct IO
{
    template<class T>
    using ConstPtr = const T*;

    using Types = util::TypeList<double, float, char, uint8_t, int, int64_t, unsigned, uint64_t>;
};

class IFileWriter
{
    template<class T>
    using ToVec = const std::vector<T>;

public:
    using FieldType   = util::Reduce<std::variant, util::Map<IO::ConstPtr, IO::Types>>;
    using FieldVector = util::Reduce<std::variant, util::Map<ToVec, IO::Types>>;

    virtual int         rank() const     = 0;
    virtual int         numRanks() const = 0;
    virtual std::string suffix() const   = 0;

    virtual ~IFileWriter() = default;

    virtual void    addStep(size_t firstIndex, size_t lastIndex, std::string path, bool settingsWriting = false) = 0;
    virtual int64_t stepAttributeSize(const std::string& /*key*/) { return 0; }
    virtual void    stepAttribute(const std::string& key, FieldType val, int64_t size) = 0;
    virtual void    fileAttribute(const std::string& key, FieldType val, int64_t size) = 0;
    virtual void    writeField(const std::string& key, FieldType field, int col)       = 0;
    virtual void    closeStep()                                                        = 0;
};

enum class FileMode
{
    collective  = 0,
    independent = 1,
};

class IFileReader
{
public:
    using FieldType = util::Reduce<std::variant, util::Map<std::add_pointer_t, IO::Types>>;

    virtual ~IFileReader() = default;

    virtual int                      rank() const                                                       = 0;
    virtual int64_t                  numParticles() const                                               = 0;
    virtual void                     setStep(std::string path, int step, FileMode mode)                 = 0;
    virtual std::vector<std::string> fileAttributes()                                                   = 0;
    virtual std::vector<std::string> stepAttributes()                                                   = 0;
    virtual int64_t                  fileAttributeSize(const std::string& key)                          = 0;
    virtual int64_t                  stepAttributeSize(const std::string& key)                          = 0;
    virtual void                     fileAttribute(const std::string& key, FieldType val, int64_t size) = 0;
    virtual void                     stepAttribute(const std::string& key, FieldType val, int64_t size) = 0;
    virtual void                     readField(const std::string& key, FieldType field)                 = 0;
    virtual uint64_t                 localNumParticles()                                                = 0;
    virtual uint64_t                 globalNumParticles()                                               = 0;
    virtual void                     closeStep()                                                        = 0;
};

class UnimplementedReader : public IFileReader
{
public:
    int                      rank() const override { return 0; };
    int64_t                  numParticles() const override { return 0; };
    void                     setStep(std::string, int, FileMode) override { throwError(); }
    std::vector<std::string> fileAttributes() override
    {
        throwError();
        return {};
    }
    std::vector<std::string> stepAttributes() override
    {
        throwError();
        return {};
    }
    int64_t fileAttributeSize(const std::string&) override
    {
        throwError();
        return {};
    }
    int64_t stepAttributeSize(const std::string&) override
    {
        throwError();
        return {};
    }
    void     fileAttribute(const std::string&, FieldType, int64_t) override { throwError(); }
    void     stepAttribute(const std::string&, FieldType, int64_t) override { throwError(); }
    void     readField(const std::string&, FieldType) override { throwError(); }
    uint64_t localNumParticles() override
    {
        throwError();
        return {};
    }
    uint64_t globalNumParticles() override
    {
        throwError();
        return {};
    }
    void closeStep() override { throwError(); }

private:
    static void throwError() { throw std::runtime_error("File reader not implemented\n"); }
};

} // namespace sphexa
