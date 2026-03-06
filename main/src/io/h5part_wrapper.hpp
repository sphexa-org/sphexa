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

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "H5hut.h"

#include "cstone/util/type_list.hpp"
#include "cstone/util/tuple_util.hpp"
#include "type_config.h"

namespace sphexa::fileutils
{

using H5Types = util::TypeList<double, float, std::int8_t, std::uint8_t, std::int16_t, std::uint16_t, std::int32_t,
                               std::uint32_t, std::int64_t, std::uint64_t>;

enum attribute_type
{
    file,
    step,
};

template<typename I>
struct H5hutType;

template<>
struct H5hutType<double>
{
    inline static constexpr auto value    = H5_FLOAT64_T;
    inline static constexpr char string[] = "python: np.float64";
};

template<>
struct H5hutType<float>
{
    inline static constexpr auto value    = H5_FLOAT32_T;
    inline static constexpr char string[] = "python: np.float32";
};

template<std::signed_integral I>
requires(sizeof(I) == 1) struct H5hutType<I>
{
    inline static constexpr auto value    = H5_INT8_T;
    inline static constexpr char string[] = "python: np.int8";
};

template<std::signed_integral I>
requires(sizeof(I) == 2) struct H5hutType<I>
{
    inline static constexpr auto value    = H5_INT16_T;
    inline static constexpr char string[] = "python: np.int16";
};

template<std::signed_integral I>
requires(sizeof(I) == 4) struct H5hutType<I>
{
    inline static constexpr auto value    = H5_INT32_T;
    inline static constexpr char string[] = "python: np.int32";
};

template<std::signed_integral I>
requires(sizeof(I) == 8) struct H5hutType<I>
{
    inline static constexpr auto value    = H5_INT64_T;
    inline static constexpr char string[] = "python: np.int64";
};

template<std::unsigned_integral I>
requires(sizeof(I) == 1) struct H5hutType<I>
{
    inline static constexpr auto value    = H5_UINT8_T;
    inline static constexpr char string[] = "python: np.uint8";
};

template<std::unsigned_integral I>
requires(sizeof(I) == 2) struct H5hutType<I>
{
    inline static constexpr auto value    = H5_UINT16_T;
    inline static constexpr char string[] = "python: np.uint16";
};

template<std::unsigned_integral I>
requires(sizeof(I) == 4) struct H5hutType<I>
{
    inline static constexpr auto value    = H5_UINT32_T;
    inline static constexpr char string[] = "python: np.uint32";
};

template<std::unsigned_integral I>
requires(sizeof(I) == 8) struct H5hutType<I>
{
    inline static constexpr auto value    = H5_UINT64_T;
    inline static constexpr char string[] = "python: np.uint64";
};

template<typename I>
inline constexpr auto H5hutType_v = H5hutType<I>::value;


//! @brief return the names of all datasets in @p h5_file
std::vector<std::string> datasetNames(h5_file_t h5_file)
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
std::vector<std::string> fileAttributeNames(h5_file_t h5_file)
{
    auto numAttributes = H5GetNumFileAttribs(h5_file);

    std::vector<std::string> setNames(numAttributes);
    for (int64_t fi = 0; fi < numAttributes; ++fi)
    {
        int        maxlen = 256;
        char       attrName[maxlen];
        h5_int64_t typeId;
        h5_size_t  attrSize;

        H5GetFileAttribInfo(h5_file, fi, attrName, maxlen, &typeId, &attrSize);
        setNames[fi] = std::string(attrName);
    }

    return setNames;
}

//! @brief return the names of all step attributes in @p h5_file
std::vector<std::string> stepAttributeNames(h5_file_t h5_file)
{
    auto numAttributes = H5GetNumStepAttribs(h5_file);

    std::vector<std::string> setNames(numAttributes);
    for (int64_t fi = 0; fi < numAttributes; ++fi)
    {
        int        maxlen = 256;
        char       attrName[maxlen];
        h5_int64_t typeId;
        h5_size_t  attrSize;

        H5GetStepAttribInfo(h5_file, fi, attrName, maxlen, &typeId, &attrSize);
        setNames[fi] = std::string(attrName);
    }

    return setNames;
}

// ---------------------------------------------------------------------------
// templated access to Step Attributes
template<typename T>
inline h5_err_t H5WriteStepAttribT(const h5_file_t f, const char* const attrib_name, const T* const buffer,
                                   const h5_size_t nelems)
{
    // copied from h5hut source, but templated to simplify calling
    return h5_write_iteration_attrib(f, attrib_name, H5hutType_v<T>, buffer, nelems);
}

template<typename T>
inline h5_err_t H5ReadStepAttribT(const h5_file_t f, const char* const attrib_name, void* const buffer)
{
    // copied from h5hut source, but templated to simplify calling
    return h5_read_iteration_attrib(f, attrib_name, H5hutType_v<T>, (void*)buffer);
}

template<class ExtractType>
auto readH5PartStepAttribute(ExtractType* attr, size_t size, int attrIndex, h5_file_t h5File)
{
    return readAttribute_typesafe(h5File, attribute_type::step, attr, int(size), attrIndex);
}

// ---------------------------------------------------------------------------
// templated access to File Attributes
template<typename T>
static inline h5_err_t H5WriteFileAttribT(const h5_file_t f, const char* const attrib_name, const T* const buffers,
                                          const h5_size_t nelems)
{
    // copied from h5hut source, but templated to simplify calling
    return h5_write_file_attrib(f, attrib_name, H5hutType_v<T>, buffers, nelems);
}

template<typename T>
static inline h5_err_t H5ReadFileAttribT(const h5_file_t f, const char* const attrib_name, void* const buffer)
{
    // copied from h5hut source, but templated to simplify calling
    return h5_read_file_attrib(f, attrib_name, H5hutType_v<T>, (void*)buffer);
}

template<class ExtractType>
auto readH5PartFileAttribute(ExtractType* attr, size_t size, int attrIndex, h5_file_t h5File)
{
    return readAttribute_typesafe(h5File, attribute_type::file, attr, int(size), attrIndex);
}

// ---------------------------------------------------------------------------
// templated access to Fields
template<typename T>
static inline h5_err_t H5PartWriteDataT(const h5_file_t f, const char* name, const T* data)
{
    // copied from h5hut source, but templated to simplify calling
    return h5u_write_dataset(f, name, (void*)data, H5hutType_v<T>);
}

template<typename T>
static inline h5_err_t H5PartReadDataT(const h5_file_t f, const char* name, T* data)
{
    // copied from h5hut source, but templated to simplify calling
    return h5u_read_dataset(f, name, data, H5hutType_v<T>);
}

// ---------------------------------------------------------------------------
//! @brief Read an HDF5 attribute into the provided buffer, doing type-conversions when it is safe to do so
template<class ExtractType>
void readAttribute_typesafe(h5_file_t h5_file, attribute_type domain, ExtractType* attr, std::size_t attrSizeBuf,
                            int attrIndex)
{
    using IoTuple = util::Reduce<std::tuple, IO::Types>;

    h5_int64_t typeId;
    h5_size_t  attrSizeFile;
    char       attrName[256];
    bool       breakLoop = false;

    if (std::invoke(
            [&]()
            {
                if (domain == attribute_type::step)
                    return H5GetStepAttribInfo(h5_file, attrIndex, attrName, 256, &typeId, &attrSizeFile);
                else
                    return H5GetFileAttribInfo(h5_file, attrIndex, attrName, 256, &typeId, &attrSizeFile);
            }) != H5_SUCCESS)
    {
        throw std::runtime_error("Could not read attribute info " + std::string(attrName) + "\n");
    }

    if (attrSizeBuf != attrSizeFile)
    {
        throw std::runtime_error("Attribute " + std::string(attrName) + " size mismatch: in file " +
                                 std::to_string(attrSizeFile) + ", but provided buffer has size " +
                                 std::to_string(attrSizeBuf) + "\n");
    }

    auto readTypesafe = [&](auto dummyValue)
    {
        using TrialType = std::decay_t<decltype(dummyValue)>;
        if (H5hutType_v<TrialType> == typeId && not breakLoop)
        {
            bool h5IsSame            = H5hutType_v<TrialType> == H5hutType_v<ExtractType>;
            bool bothFloating        = std::is_floating_point_v<TrialType> && std::is_floating_point_v<ExtractType>;
            bool extractToCommonType = std::is_same_v<std::common_type_t<TrialType, ExtractType>, ExtractType>;
            if (h5IsSame ||  bothFloating || extractToCommonType)
            {
                std::vector<TrialType> attrBuf(attrSizeFile);
                if (std::invoke(
                        [&]()
                        {
                            if (domain == attribute_type::step)
                                return H5ReadStepAttribT<TrialType>(h5_file, attrName, attrBuf.data());
                            else
                                return H5ReadFileAttribT<TrialType>(h5_file, attrName, attrBuf.data());
                        }) != H5_SUCCESS)
                {
                    throw std::runtime_error("Could not read attribute " + std::string(attrName) + "\n");
                }
                std::copy(attrBuf.begin(), attrBuf.end(), attr);
            }
            else
            {
                throw std::runtime_error("Reading attribute " + std::string(attrName) +
                                         " failed: " + "type in file is " + H5hutType<TrialType>::string +
                                         ", but supplied buffer type is " + H5hutType<ExtractType>::string + "\n");
            }
            breakLoop = true;
        }
    };
    util::for_each_tuple(readTypesafe, IoTuple{});
}

// ---------------------------------------------------------------------------
//! @brief Open in parallel mode if supported, otherwise serial if numRanks == 1
h5_file_t openH5Part(const std::string& path, h5_int64_t mode, MPI_Comm comm)
{
    const char* h5_fname = path.c_str();
    h5_file_t   h5_file;

#ifdef H5_HAVE_PARALLEL
    h5_prop_t prop = H5CreateFileProp();
#if defined(SPH_EXA_HDF5_PARALLEL_MODE_INDEPENDENT)
    H5SetPropFileMPIOIndependent(prop, &comm);
#elif defined(SPH_EXA_HDF5_PARALLEL_MODE_COLLECTIVE)
    H5SetPropFileMPIOCollective(prop, &comm);
#else
    static_assert(false, "HDF5 Parallel IO mode unset");
#endif
    h5_file = H5OpenFile(h5_fname, mode, prop);
    H5CloseProp(prop);
#else
    int numRanks;
    MPI_Comm_size(comm, &numRanks);
    if (numRanks > 1)
    {
        throw std::runtime_error("Cannot open HDF5 file on multiple ranks without parallel HDF5 support\n");
    }
    h5_file = H5OpenFile(h5_fname, mode, H5_PROP_DEFAULT);
#endif

    return h5_file;
}

} // namespace sphexa::fileutils
