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
 * @brief A C++ layer over H5Part
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "cstone/util/type_list.hpp"
#include "cstone/util/tuple_util.hpp"

#include "H5Part.h"

namespace sphexa
{
namespace fileutils
{

using H5PartTypes = util::TypeList<double, float, char, int, int64_t, unsigned, uint64_t>;

std::string H5PartTypeToString(h5part_int64_t type)
{
    if (type == H5PART_FLOAT64) { return "C++: double / python: np.float64"; }
    if (type == H5PART_FLOAT32) { return "C++: float / python: np.float32"; }
    if (type == H5PART_INT32) { return "C++: int / python: np.int32"; }
    if (type == H5PART_INT64) { return "C++: int64_t / python: np.int64"; }
    if (type == H5PART_CHAR) { return "C++: char / python: np.int8"; }

    return "H5PART_UNKNOWN";
}

template<class T>
struct H5PartType
{
};

template<>
struct H5PartType<double>
{
    operator h5part_int64_t() const noexcept { return H5PART_FLOAT64; } // NOLINT
};

template<>
struct H5PartType<float>
{
    operator h5part_int64_t() const noexcept { return H5PART_FLOAT32; } // NOLINT
};

template<>
struct H5PartType<char>
{
    operator h5part_int64_t() const noexcept { return H5PART_CHAR; } // NOLINT
};

template<>
struct H5PartType<unsigned char>
{
    operator h5part_int64_t() const noexcept { return H5PART_CHAR; } // NOLINT
};

template<>
struct H5PartType<int>
{
    operator h5part_int64_t() const noexcept { return H5PART_INT32; } // NOLINT
};

template<>
struct H5PartType<unsigned>
{
    operator h5part_int64_t() const noexcept { return H5PART_INT32; } // NOLINT
};

template<>
struct H5PartType<int64_t>
{
    operator h5part_int64_t() const noexcept { return H5PART_INT64; } // NOLINT
};

template<>
struct H5PartType<uint64_t>
{
    operator h5part_int64_t() const noexcept { return H5PART_INT64; } // NOLINT
};

//! @brief return the names of all datasets in @p h5_file
std::vector<std::string> datasetNames(H5PartFile* h5_file)
{
    auto numSets = H5PartGetNumDatasets(h5_file);

    std::vector<std::string> setNames(numSets);
    for (int64_t fi = 0; fi < numSets; ++fi)
    {
        int  maxlen = 256;
        char fieldName[maxlen];
        H5PartGetDatasetName(h5_file, fi, fieldName, maxlen);
        setNames[fi] = std::string(fieldName);
    }

    return setNames;
}

//! @brief return the names of all file attributes in @p h5_file
std::vector<std::string> fileAttributeNames(H5PartFile* h5_file)
{
    auto numAttributes = H5PartGetNumFileAttribs(h5_file);

    std::vector<std::string> setNames(numAttributes);
    for (int64_t fi = 0; fi < numAttributes; ++fi)
    {
        int            maxlen = 256;
        char           attrName[maxlen];
        h5part_int64_t typeId, attrSize;

        H5PartGetFileAttribInfo(h5_file, fi, attrName, maxlen, &typeId, &attrSize);
        setNames[fi] = std::string(attrName);
    }

    return setNames;
}

//! @brief return the names of all step attributes in @p h5_file
std::vector<std::string> stepAttributeNames(H5PartFile* h5_file)
{
    auto numAttributes = H5PartGetNumStepAttribs(h5_file);

    std::vector<std::string> setNames(numAttributes);
    for (int64_t fi = 0; fi < numAttributes; ++fi)
    {
        int            maxlen = 256;
        char           attrName[maxlen];
        h5part_int64_t typeId, attrSize;

        H5PartGetStepAttribInfo(h5_file, fi, attrName, maxlen, &typeId, &attrSize);
        setNames[fi] = std::string(attrName);
    }

    return setNames;
}

template<class T>
auto writeH5PartStepAttrib(H5PartFile* h5_file, const std::string& name, const T* value, size_t numElements)
{
    return H5PartWriteStepAttrib(h5_file, name.c_str(), H5PartType<T>{}, value, numElements);
}

template<class T>
auto writeH5PartFileAttrib(H5PartFile* h5_file, const std::string& name, const T* value, size_t numElements)
{
    return H5PartWriteFileAttrib(h5_file, name.c_str(), H5PartType<T>{}, value, numElements);
}

//! @brief Read an HDF5 attribute into the provided buffer, doing type-conversions when it is safe to do so
template<class ExtractType, class Reader, class Info>
void readAttribute(ExtractType* attr, int attrSizeBuf, int attrIndex, Reader&& readAttrib, Info&& readAttribInfo)
{
    using IoTuple = util::Reduce<std::tuple, H5PartTypes>;

    h5part_int64_t typeId, attrSizeFile;
    char           attrName[256];
    readAttribInfo(attrIndex, attrName, 256, &typeId, &attrSizeFile);
    bool breakLoop = false;

    if (attrSizeBuf != attrSizeFile)
    {
        throw std::runtime_error("Attribute " + std::string(attrName) + " size mismatch: in file " +
                                 std::to_string(attrSizeFile) + ", but provided buffer has size " +
                                 std::to_string(attrSizeBuf) + "\n");
    }

    auto readTypesafe = [&](auto dummyValue)
    {
        using TypeInFile = std::decay_t<decltype(dummyValue)>;
        if (fileutils::H5PartType<TypeInFile>{} == typeId && not breakLoop)
        {
            std::vector<TypeInFile> attrBuf(attrSizeFile);
            if (readAttrib(attrName, attrBuf.data()) != H5PART_SUCCESS)
            {
                throw std::runtime_error("Could not read attribute " + std::string(attrName) + "\n");
            }

            bool bothFloating        = std::is_floating_point_v<TypeInFile> && std::is_floating_point_v<ExtractType>;
            bool extractToCommonType = std::is_same_v<std::common_type_t<TypeInFile, ExtractType>, ExtractType>;
            if (bothFloating || extractToCommonType) { std::copy(attrBuf.begin(), attrBuf.end(), attr); }
            else
            {
                int64_t memTypeId = fileutils::H5PartType<ExtractType>{};
                throw std::runtime_error("Reading attribute " + std::string(attrName) +
                                         " failed: " + "type in file is " + fileutils::H5PartTypeToString(typeId) +
                                         ", but supplied buffer type is " + fileutils::H5PartTypeToString(memTypeId) +
                                         "\n");
            }
            breakLoop = true;
        }
    };
    util::for_each_tuple(readTypesafe, IoTuple{});
}

template<class ExtractType>
auto readH5PartStepAttribute(ExtractType* attr, size_t size, int attrIndex, H5PartFile* h5File)
{
    auto read = [h5File](const char* key, void* attr) { return H5PartReadStepAttrib(h5File, key, attr); };

    auto info = [h5File](int idx, char* buf, int sz, h5part_int64_t* typeId, h5part_int64_t* attrSize)
    { return H5PartGetStepAttribInfo(h5File, idx, buf, sz, typeId, attrSize); };

    return readAttribute(attr, int(size), attrIndex, read, info);
}

template<class ExtractType>
auto readH5PartFileAttribute(ExtractType* attr, size_t size, int attrIndex, H5PartFile* h5File)
{
    auto read = [h5File](const char* key, void* attr) { return H5PartReadFileAttrib(h5File, key, attr); };

    auto info = [h5File](int idx, char* buf, int sz, h5part_int64_t* typeId, h5part_int64_t* attrSize)
    { return H5PartGetFileAttribInfo(h5File, idx, buf, sz, typeId, attrSize); };

    return readAttribute(attr, int(size), attrIndex, read, info);
}

/* read fields */

inline h5part_int64_t readH5PartField(H5PartFile* h5_file, const std::string& fieldName, double* field)
{
    static_assert(std::is_same_v<double, h5part_float64_t>);
    return H5PartReadDataFloat64(h5_file, fieldName.c_str(), field);
}

inline h5part_int64_t readH5PartField(H5PartFile* h5_file, const std::string& fieldName, float* field)
{
    static_assert(std::is_same_v<float, h5part_float32_t>);
    return H5PartReadDataFloat32(h5_file, fieldName.c_str(), field);
}

inline h5part_int64_t readH5PartField(H5PartFile* h5_file, const std::string& fieldName, uint8_t* field)
{
    return H5PartReadDataInt8(h5_file, fieldName.c_str(), field);
}

inline h5part_int64_t readH5PartField(H5PartFile* h5_file, const std::string& fieldName, char* field)
{
    return H5PartReadDataInt8(h5_file, fieldName.c_str(), (uint8_t*)field);
}

inline h5part_int64_t readH5PartField(H5PartFile* h5_file, const std::string& fieldName, int* field)
{
    static_assert(std::is_same_v<int, h5part_int32_t>);
    return H5PartReadDataInt32(h5_file, fieldName.c_str(), field);
}

inline h5part_int64_t readH5PartField(H5PartFile* h5_file, const std::string& fieldName, int64_t* field)
{
    return H5PartReadDataInt64(h5_file, fieldName.c_str(), (h5part_int64_t*)field);
}

inline h5part_int64_t readH5PartField(H5PartFile* h5_file, const std::string& fieldName, unsigned* field)
{
    return H5PartReadDataInt32(h5_file, fieldName.c_str(), (h5part_int32_t*)field);
}

inline h5part_int64_t readH5PartField(H5PartFile* h5_file, const std::string& fieldName, uint64_t* field)
{
    return H5PartReadDataInt64(h5_file, fieldName.c_str(), (h5part_int64_t*)field);
}

/* write fields */

inline h5part_int64_t writeH5PartField(H5PartFile* h5_file, const std::string& fieldName, const double* field)
{
    static_assert(std::is_same_v<double, h5part_float64_t>);
    return H5PartWriteDataFloat64(h5_file, fieldName.c_str(), field);
}

inline h5part_int64_t writeH5PartField(H5PartFile* h5_file, const std::string& fieldName, const float* field)
{
    static_assert(std::is_same_v<float, h5part_float32_t>);
    return H5PartWriteDataFloat32(h5_file, fieldName.c_str(), field);
}

inline h5part_int64_t writeH5PartField(H5PartFile* h5_file, const std::string& fieldName, const uint8_t* field)
{
    return H5PartWriteDataInt8(h5_file, fieldName.c_str(), field);
}

inline h5part_int64_t writeH5PartField(H5PartFile* h5_file, const std::string& fieldName, const char* field)
{
    return H5PartWriteDataInt8(h5_file, fieldName.c_str(), (const uint8_t*)field);
}

inline h5part_int64_t writeH5PartField(H5PartFile* h5_file, const std::string& fieldName, const int* field)
{
    static_assert(std::is_same_v<int, h5part_int32_t>);
    return H5PartWriteDataInt32(h5_file, fieldName.c_str(), field);
}

inline h5part_int64_t writeH5PartField(H5PartFile* h5_file, const std::string& fieldName, const int64_t* field)
{
    return H5PartWriteDataInt64(h5_file, fieldName.c_str(), (const h5part_int64_t*)field);
}

inline h5part_int64_t writeH5PartField(H5PartFile* h5_file, const std::string& fieldName, const unsigned* field)
{
    return H5PartWriteDataInt32(h5_file, fieldName.c_str(), (const h5part_int32_t*)field);
}

inline h5part_int64_t writeH5PartField(H5PartFile* h5_file, const std::string& fieldName, const uint64_t* field)
{
    return H5PartWriteDataInt64(h5_file, fieldName.c_str(), (const h5part_int64_t*)field);
}

//! @brief Open in parallel mode if supported, otherwise serial if numRanks == 1
H5PartFile* openH5Part(const std::string& path, h5part_int64_t mode, MPI_Comm comm)
{
    const char* h5_fname = path.c_str();
    H5PartFile* h5_file  = nullptr;

#ifdef H5PART_PARALLEL_IO
    h5_file = H5PartOpenFileParallel(h5_fname, mode, comm);
#else
    int numRanks;
    MPI_Comm_size(comm, &numRanks);
    if (numRanks > 1)
    {
        throw std::runtime_error("Cannot open HDF5 file on multiple ranks without parallel HDF5 support\n");
    }
    h5_file = H5PartOpenFile(h5_fname, mode);
#endif

    return h5_file;
}

} // namespace fileutils
} // namespace sphexa
