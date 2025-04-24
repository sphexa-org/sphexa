/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Encapsulation around a std::vector like GPU-resident device vector, suitable for use in .cpp files
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <memory>
#include <vector>

#include "cstone/util/array.hpp"

namespace cstone
{

/*! @brief Encapsulation for a device vector with a minimal interface that hides any GPU/CUDA headers from CPU code
 *
 * This class is used in .cpp host code where thrust::device_vector cannot be used and in interfaces to device code
 * callable from the host. Within .cu translation units, thrust::device_vector is still used as it has the full set of
 * features.
 */
template<class T>
class DeviceVector
{
public:
    using value_type = T;

    DeviceVector();
    DeviceVector(std::size_t);
    DeviceVector(std::size_t, T init);
    DeviceVector(const DeviceVector&);
    //! @brief upload from host vecotr
    DeviceVector(const std::vector<T>&);
    //! @brief upload from host pointers
    DeviceVector(const T* first, const T* last);

    ~DeviceVector();

    T* data();
    const T* data() const;

    void resize(std::size_t size);

    void reserve(std::size_t size);
    void shrink_to_fit();

    std::size_t size() const;
    bool empty() const;
    std::size_t capacity() const;

    DeviceVector& swap(DeviceVector<T>& rhs);
    DeviceVector& operator=(const std::vector<T>& rhs);
    DeviceVector& operator=(DeviceVector<T> rhs);

private:
    friend void swap(DeviceVector& lhs, DeviceVector& rhs) { lhs.swap(rhs); }
    template<class S>
    friend bool operator==(const DeviceVector<S>& lhs, const DeviceVector<S>& rhs);

    class Impl;
    std::unique_ptr<Impl> impl_;
};

extern template class DeviceVector<char>;
extern template class DeviceVector<uint8_t>;
extern template class DeviceVector<int>;
extern template class DeviceVector<unsigned>;
extern template class DeviceVector<uint64_t>;
extern template class DeviceVector<float>;
extern template class DeviceVector<double>;
extern template class DeviceVector<util::array<int, 2>>;
extern template class DeviceVector<util::array<int, 3>>;
extern template class DeviceVector<util::array<unsigned, 1>>;
extern template class DeviceVector<util::array<unsigned, 2>>;
extern template class DeviceVector<util::array<float, 3>>;
extern template class DeviceVector<util::array<double, 3>>;
extern template class DeviceVector<util::array<float, 4>>;
extern template class DeviceVector<util::array<double, 4>>;

} // namespace cstone

template<class T>
T* rawPtr(cstone::DeviceVector<T>& p)
{
    return p.data();
}

template<class T>
const T* rawPtr(const cstone::DeviceVector<T>& p)
{
    return p.data();
}
