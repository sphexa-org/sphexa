/*
 * MIT License
 *
 * Copyright (c) 2024 CSCS, ETH Zurich, University of Basel, University of Zurich
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
 * @brief Contains the object holding all particle data required for Magneto-Hydrodynamics
 *
 * @author Lukas Schmidt
 *
 */

#pragma once

#include <array>
#include <iostream>
#include <vector>
#include <variant>

#include "cstone/cuda/cuda_utils.hpp"
#include "cstone/fields/data_util.hpp"
#include "cstone/fields/field_states.hpp"
#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/tree/definitions.h"
#include "cstone/tree/octree.hpp"
#include "cstone/util/reallocate.hpp"
#include "sph/types.hpp"

#include "sph/particles_data_stubs.hpp"

#if defined(USE_CUDA)
#include "magneto_data_gpu.cuh"
#endif

namespace sphexa::magneto
{
template<class AccType>
class MagnetoData : public cstone::FieldStates<MagnetoData<AccType>>
{
public:
    using AcceleratorType = AccType;

    using RealType  = sph::SphTypes::CoordinateType;
    using HydroType = sph::SphTypes::HydroType;
    using XM1Type   = sph::SphTypes::XM1Type;
    using Tmass     = sph::SphTypes::Tmass;

    template<class ValueType>
    using PinnedVec = std::vector<ValueType, PinnedAlloc_t<AcceleratorType, ValueType>>;

    template<class ValueType>
    using FieldVector = std::vector<ValueType, std::allocator<ValueType>>;

    using FieldVariant = std::variant<FieldVector<float>*, FieldVector<double>*, FieldVector<unsigned>*,
                                      FieldVector<uint64_t>*, FieldVector<uint8_t>*>;

    MagnetoData() {}
    MagnetoData(const MagnetoData&) = delete;

    RealType mu_0{1.0}, eMag{0.0};

    //!@brief particle fields used for magneto-hydrodynamics
    FieldVector<RealType> Bx, By, Bz;             // Magnetic field components
    FieldVector<RealType> dBx, dBy, dBz;          // Magnetic field rate of change (dB_i/dt)
    FieldVector<XM1Type>  dBx_m1, dBy_m1, dBz_m1; // previous Magnetic field rate of change (dB_i/dt)

    // fields for divergence cleaning
    FieldVector<HydroType> psi_ch;      // scalar field used for divergence cleaning divided by the cleaning speed
    FieldVector<HydroType> d_psi_ch;    // rate of change of the scalar field (d/dt(psi/ch))
    FieldVector<XM1Type>   d_psi_ch_m1; // previous rate of change

    // Velocity Jacobian
    FieldVector<HydroType> dvxdx, dvxdy, dvxdz;
    FieldVector<HydroType> dvydx, dvydy, dvydz;
    FieldVector<HydroType> dvzdx, dvzdy, dvzdz;

    // Magnetic field spatial derivatives
    FieldVector<HydroType> divB;
    FieldVector<HydroType> curlB_x, curlB_y, curlB_z;

    DeviceMagneto_t<AccType> devData;

    /* Is this a good idea?
     *
     * //! @brief returns external magnetic field contribtion at @p pos and @p time
     *
    HOST_DEVICE_INLINE cstone::Vec3<RealType> externalMagneticField(const cstone::Vec3<RealType>& pos, const RealType
    time)
    {

    }
    */

    /*! @brief
     * Name of each field as string for use e.g in HDF5 output. Order has to correspond to what's returned by data().
     */
    inline static constexpr std::array fieldNames{
        "Bx",     "By",       "Bz",          "dBx",   "dBy",     "dBz",     "dBx_m1", "dBy_m1", "dBz_m1",
        "psi_ch", "d_psi_ch", "d_psi_ch_m1", "dvxdx", "dvxdy",   " dvxdz",  "dvydx",  "dvydy",  "dvydz",
        "dvzdx",  "dvzdy",    "dvzdz",       "divB",  "curlB_x", "curlB_y", "curlB_z"};

    static const inline std::string prefix{"magneto::"};

    static_assert(!cstone::HaveGpu<AcceleratorType>{} || fieldNames.size() == DeviceMagneto_t<AccType>::fieldNames.size(),
                  "MagnetoData on CPU and GPU must have the same fields");

    /*! @brief return a tuple of field references
     *
     * Note: this needs to be in the same order as listed in fieldNames
     */
    auto dataTuple()
    {
        auto ret = std::tie(Bx, By, Bz, dBx, dBy, dBz, dBx_m1, dBy_m1, dBz_m1, psi_ch, d_psi_ch, d_psi_ch_m1, dvxdx,
                            dvxdy, dvxdz, dvydx, dvydy, dvydz, dvzdx, dvzdy, dvzdz, divB, curlB_x, curlB_y, curlB_z);

#if defined(__clang__) || __GNUC__ > 11
        static_assert(std::tuple_size_v<decltype(ret)> == fieldNames.size());
#endif
        return ret;
    }
    /*! @brief return a vector of pointers to field vectors
     *
     * We implement this by returning an rvalue to prevent having to store pointers and avoid
     * non-trivial copy/move constructors.
     */
    auto data()
    {
        return std::apply([](auto&... fields) { return std::array<FieldVariant, sizeof...(fields)>{&fields...}; },
                          dataTuple());
    }

    /*! @brief mark fields file output
     *
     * @param outFields  list of field names
     *
     * Selected fields that match existing names contained in @a fieldNames will be removed from the argument
     * @p field names.
     */
    void setOutputFields(std::vector<std::string>& outFields)
    {
        auto hasField = [](const std::string& field)
        { return cstone::getFieldIndex(field, fieldNames) < fieldNames.size(); };

        std::copy_if(outFields.begin(), outFields.end(), std::back_inserter(outputFieldNames), hasField);
        outputFieldIndices = cstone::fieldStringsToInt(outputFieldNames, fieldNames);
        std::for_each(outputFieldNames.begin(), outputFieldNames.end(), [](auto& f) { f = prefix + f; });

        outFields.erase(std::remove_if(outFields.begin(), outFields.end(), hasField), outFields.end());
    }

    void resize(size_t size)
    {
        auto data_ = data();

        auto deallocateVector = [size](auto* devVectorPtr)
        {
            using DevVector = std::decay_t<decltype(*devVectorPtr)>;
            if (devVectorPtr->capacity() < size) { *devVectorPtr = DevVector{}; }
        };

        for (size_t i = 0; i < data_.size(); ++i)
        {
            if (this->isAllocated(i) && not this->isConserved(i)) { std::visit(deallocateVector, data_[i]); }
        }

        for (size_t i = 0; i < data_.size(); ++i)
        {
            if (this->isAllocated(i))
            {
                std::visit([size, gr = allocGrowthRate_](auto* arg) { reallocate(*arg, size, gr); }, data_[i]);
            }
        }

        devData.resize(size, allocGrowthRate_);
    }

    size_t size()
    {
        auto data_ = data();
        for (size_t i = 0; i < data_.size(); ++i)
        {
            if (this->isAllocated(i))
            {
                return std::visit([](auto* arg) { return arg->size(); }, data_[i]);
            }
        }
        return 0;
    }

    //! @brief resize GPU arrays if in use, CPU arrays otherwise
    void resizeAcc(size_t size)
    {
        if (cstone::HaveGpu<AccType>{}) { devData.resize(size, allocGrowthRate_); }
        else { resize(size); }
    }

    //! @brief return the size of GPU arrays if in use, CPU arrays otherwise
    size_t accSize()
    {
        if (cstone::HaveGpu<AccType>{}) { return devData.size(); }
        else { return size(); }
    }

    //! @brief particle fields selected for file output
    std::vector<int>         outputFieldIndices;
    std::vector<std::string> outputFieldNames;

    //! @brief buffer growth factor when reallocating
    float allocGrowthRate_{1.05};

    float getAllocGrowthRate() const { return allocGrowthRate_; }

    //! @brief Unified interface to attribute initialization, reading and writing
    template<class Archive>
    void loadOrStoreAttributes(Archive* ar)
    {
        //! @brief load or store an attribute, skips non-existing attributes on load.
        auto optionalIO = [ar](const std::string& attribute, auto* location, size_t attrSize)
        {
            try
            {
                if constexpr (std::is_enum_v<std::decay_t<decltype(*location)>>)
                {
                    // handle pointers to enum by casting to the underlying type
                    using EType = std::decay_t<decltype(*location)>;
                    using UType = std::underlying_type_t<EType>;
                    auto tmp    = static_cast<UType>(*location);
                    ar->stepAttribute(attribute, &tmp, attrSize);
                    *location = static_cast<EType>(tmp);
                }
                else { ar->stepAttribute(attribute, location, attrSize); }
            }
            catch (std::out_of_range&)
            {
                if (ar->rank() == 0)
                {
                    std::cout << "Attribute " << attribute << " not set in file, setting to default value " << *location
                              << std::endl;
                }
            }
        };
        optionalIO("mu_0", &mu_0, 1);
    }
};
} // namespace sphexa::magneto
