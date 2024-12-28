/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Extension for particle field containers. Provides acquire/release semantics.
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <algorithm>
#include <vector>
#include <variant>

namespace cstone
{

/*! @brief Helper class to keep track of field states
 *
 * @tparam DataType  the array with the particles fields with a static array "fieldNames" and a data()
 *                   function returning variant pointers to the fields
 *
 * -Conserved fields always stay allocated and their state cannot be modified by release/acquire
 *
 * -Dependent fields are also allocated, but the list of dependent fields can be changed at any time
 *  by release/acquire
 *
 * -release and acquire do NOT deallocate or allocate memory, they just pass on existing memory from
 *  one field to another.
 *
 *  This class guarantees that:
 *      -conserved fields are not modified
 *      -only dependent fields can be released
 *      -a field can be acquired only if it is unused and a suitable released field is available
 *
 * It remains the programmers responsibility to not access unused fields or fields outside their allocated bounds.
 */
template<class DataType>
class FieldStates
{
public:
    template<class... Fields>
    void setConserved(const Fields&... fields)
    {
        [[maybe_unused]] std::initializer_list<int> list{(setState(fields, State::conserved), 0)...};
    }

    template<class... Fields>
    void setDependent(const Fields&... fields)
    {
        [[maybe_unused]] std::initializer_list<int> list{(setState(fields, State::dependent), 0)...};
    }

    bool isAllocated(size_t fieldIdx) const
    {
        return fieldIdx < fieldStates_.size() && fieldStates_[fieldIdx] != State::unused;
    }
    bool isConserved(size_t fieldIdx) const
    {
        return fieldIdx < fieldStates_.size() && fieldStates_[fieldIdx] == State::conserved;
    }

    bool isAllocated(const std::string& field) const
    {
        size_t fieldIdx =
            std::find(DataType::fieldNames.begin(), DataType::fieldNames.end(), field) - DataType::fieldNames.begin();
        return isAllocated(fieldIdx);
    }

    //! @brief indicate that @p fields are currently not required
    template<class... Fields>
    void release(const Fields&... fields)
    {
        [[maybe_unused]] std::initializer_list<int> list{(releaseOne(fields), 0)...};
    }

    //! @brief try to acquire memory for @p fields from previously released fields
    template<class... Fields>
    void acquire(const Fields&... fields)
    {
        auto data_ = static_cast<DataType*>(this)->data();

        [[maybe_unused]] std::initializer_list<int> list{(acquireOne(data_, fields), 0)...};
    }

private:
    /*! @brief private constructor to ensure that only class X that is derived from FieldStates<X> can instantiate
     *
     * This prohibits the following:
     *
     * class Y : public FieldStates<X>
     * {};
     */
    FieldStates()
        : fieldStates_(DataType::fieldNames.size(), State::unused)
    {
    }

    friend DataType;

    enum class State
    {
        unused,
        conserved,
        dependent,
        released
    };

    void releaseOne(const std::string& field)
    {
        int fieldIdx =
            std::find(DataType::fieldNames.begin(), DataType::fieldNames.end(), field) - DataType::fieldNames.begin();
        releaseOne(fieldIdx);
    }

    void releaseOne(int idx)
    {
        if (fieldStates_[idx] != State::dependent)
        {
            throw std::runtime_error("The following field could not be released due to wrong state: " +
                                     std::string(DataType::fieldNames[idx]));
        }

        fieldStates_[idx] = State::released;
    }

    template<class FieldPointers>
    void acquireOne(FieldPointers& data_, const std::string& field)
    {
        int fieldIdx =
            std::find(DataType::fieldNames.begin(), DataType::fieldNames.end(), field) - DataType::fieldNames.begin();
        acquireOne(data_, fieldIdx);
    }

    template<class FieldPointers>
    void acquireOne(FieldPointers& data_, int fieldIdx)
    {
        if (fieldStates_[fieldIdx] != State::unused)
        {
            throw std::runtime_error("The following field could not be acquired because already in use: " +
                                     std::string(DataType::fieldNames[fieldIdx]));
        }

        auto checkTypesMatch = [](const auto* var1, const auto* var2)
        {
            using Type1 = std::decay_t<decltype(*var1)>;
            using Type2 = std::decay_t<decltype(*var2)>;
            return std::is_same_v<Type1, Type2>;
        };

        auto swapFields = [](auto* varPtr1, auto* varPtr2)
        {
            using Type1 = std::decay_t<decltype(*varPtr1)>;
            using Type2 = std::decay_t<decltype(*varPtr2)>;
            if constexpr (std::is_same_v<Type1, Type2>) { swap(*varPtr1, *varPtr2); }
        };

        for (size_t i = 0; i < fieldStates_.size(); ++i)
        {
            if (fieldStates_[i] == State::released)
            {
                bool typesMatch = std::visit(checkTypesMatch, data_[i], data_[fieldIdx]);
                if (typesMatch)
                {
                    std::visit(swapFields, data_[i], data_[fieldIdx]);
                    fieldStates_[i]        = State::unused;
                    fieldStates_[fieldIdx] = State::dependent;
                    return;
                }
            }
        }
        throw std::runtime_error("Could not acquire field " + std::string(DataType::fieldNames[fieldIdx]) +
                                 ". No suitable field available");
    }

    void setState(size_t idx, State state) { fieldStates_[idx] = state; }

    void setState(const std::string& field, State state)
    {
        size_t idx =
            std::find(DataType::fieldNames.begin(), DataType::fieldNames.end(), field) - DataType::fieldNames.begin();
        if (idx == fieldStates_.size())
        {
            throw std::runtime_error("Cannot set state of " + field + ": unknown field\n");
        }
        setState(idx, state);
    }

    //! @brief current state of each field
    std::vector<State> fieldStates_;
};

} // namespace cstone
