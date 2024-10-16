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

#include "cstone/util/type_list.hpp"
#include "cstone/util/tuple_util.hpp"

#include "H5hut.h"

namespace sphexa::fileutils
{

// ---------------------------------------------------------------------------
// types used by sph-exa throughout
using H5PartTypes = util::TypeList<double, float, char, int, int64_t, unsigned, uint64_t>;

// ---------------------------------------------------------------------------
enum attribute_type
{
    file,
    step,
};

// ---------------------------------------------------------------------------
// helper to define traits that we can use for brevity
template<typename H5_Type>
struct h5_traits
{
};

#define make_h5_trait(T, NATIVE_TYPE, H5HUT_TYPE, STRING)                                                              \
    template<>                                                                                                         \
    struct h5_traits<T>                                                                                                \
    {                                                                                                                  \
                                    operator h5_int64_t() const noexcept { return NATIVE_TYPE; }                       \
        static constexpr h5_types_t h5hut_type = H5HUT_TYPE;                                                           \
        static const std::string    stringval() { return STRING; }                                                     \
    }

make_h5_trait(double, /*        */ H5T_NATIVE_DOUBLE, /* */ H5_FLOAT64_T, /**/ "C++: double / python: np.float64");
make_h5_trait(float, /*         */ H5T_NATIVE_FLOAT, /*  */ H5_FLOAT32_T, /**/ "C++: float / python: np.float32");
make_h5_trait(std::int64_t, /*  */ H5T_NATIVE_LONG, /*   */ H5_INT64_T, /*  */ "C++:  int64 / python: np.int64");
make_h5_trait(std::uint64_t, /* */ H5T_NATIVE_ULONG, /*  */ H5_UINT64_T, /* */ "C++: uint64 / python: np.uint64");
make_h5_trait(int, /*           */ H5T_NATIVE_INT, /*    */ H5_INT32_T, /*  */ "C++:  int / python: np.int32");
make_h5_trait(unsigned, /*      */ H5T_NATIVE_UINT, /*   */ H5_UINT32_T, /* */ "C++: uint / python: np.uint32");
make_h5_trait(std::int16_t, /*  */ H5T_NATIVE_SHORT, /*  */ H5_INT16_T, /*  */ "C++:  int16 / python: np.int16");
make_h5_trait(std::uint16_t, /* */ H5T_NATIVE_USHORT, /* */ H5_UINT16_T, /* */ "C++: uint16 / python: np.uint16");
make_h5_trait(char, /*          */ H5T_NATIVE_CHAR, /*   */ H5_INT8_T, /*   */ "C++:  char / python: np.int8");
make_h5_trait(unsigned char, /* */ H5T_NATIVE_UCHAR, /*  */ H5_UINT8_T, /*  */ "C++: uchar / python: np.uint8");

// ---------------------------------------------------------------------------
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
    return h5_write_iteration_attrib(f, attrib_name, h5_traits<T>::h5hut_type, buffer, nelems);
}

template<typename T>
inline h5_err_t H5ReadStepAttribT(const h5_file_t f, const char* const attrib_name, void* const buffer)
{
    // copied from h5hut source, but templated to simplify calling
    return h5_read_iteration_attrib(f, attrib_name, h5_traits<T>::h5hut_type, (void*)buffer);
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
    return h5_write_file_attrib(f, attrib_name, h5_traits<T>::h5hut_type, buffers, nelems);
}

template<typename T>
static inline h5_err_t H5ReadFileAttribT(const h5_file_t f, const char* const attrib_name, void* const buffer)
{
    // copied from h5hut source, but templated to simplify calling
    return h5_read_file_attrib(f, attrib_name, h5_traits<T>::h5hut_type, (void*)buffer);
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
    return h5u_write_dataset(f, name, (void*)data, h5_traits<T>::h5hut_type);
}

template<typename T>
static inline h5_err_t H5PartReadDataT(const h5_file_t f, const char* name, T* data)
{
    // copied from h5hut source, but templated to simplify calling
    return h5u_read_dataset(f, name, data, h5_traits<T>::h5hut_type);
}

// ---------------------------------------------------------------------------
//! @brief Read an HDF5 attribute into the provided buffer, doing type-conversions when it is safe to do so
template<class ExtractType>
void readAttribute_typesafe(h5_file_t h5_file, attribute_type domain, ExtractType* attr, std::size_t attrSizeBuf,
                            int attrIndex)
{
    using IoTuple = util::Reduce<std::tuple, H5PartTypes>;

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
        using TypeInFile = std::decay_t<decltype(dummyValue)>;
        if (h5_traits<TypeInFile>::h5hut_type == typeId && not breakLoop)
        {
            bool bothFloating        = std::is_floating_point_v<TypeInFile> && std::is_floating_point_v<ExtractType>;
            bool extractToCommonType = std::is_same_v<std::common_type_t<TypeInFile, ExtractType>, ExtractType>;
            if (bothFloating || extractToCommonType)
            {
                std::vector<TypeInFile> attrBuf(attrSizeFile);
                if (std::invoke(
                        [&]()
                        {
                            if (domain == attribute_type::step)
                                return H5ReadStepAttribT<TypeInFile>(h5_file, attrName, attrBuf.data());
                            else
                                return H5ReadFileAttribT<TypeInFile>(h5_file, attrName, attrBuf.data());
                        }) != H5_SUCCESS)
                {
                    throw std::runtime_error("Could not read attribute " + std::string(attrName) + "\n");
                }
                std::copy(attrBuf.begin(), attrBuf.end(), attr);
            }
            else
            {
                throw std::runtime_error("Reading attribute " + std::string(attrName) +
                                         " failed: " + "type in file is " + h5_traits<TypeInFile>::stringval() +
                                         ", but supplied buffer type is " + h5_traits<ExtractType>::stringval() + "\n");
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
    H5SetPropFileMPIOCollective(prop, &comm);
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
