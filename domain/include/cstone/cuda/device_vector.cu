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

#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>

#include "cstone/cuda/cuda_runtime.hpp"
#include "cstone/cuda/errorcheck.cuh"
#include "cstone/util/noinit_thrust.cuh"

#include "device_vector.h"

namespace cstone
{

template<class T>
class DeviceVector<T>::Impl
{
public:
    Impl() {}

    T* data() { return thrust::raw_pointer_cast(data_.data()); }
    const T* data() const { return thrust::raw_pointer_cast(data_.data()); }

    void resize(std::size_t size) { data_.resize(size); }

    void reserve(std::size_t size) { data_.reserve(size); };
    void shrink_to_fit() { data_.shrink_to_fit(); };

    std::size_t size() const { return data_.size(); };
    std::size_t capacity() const { return data_.capacity(); };

    Impl& operator=(const std::vector<T>& rhs)
    {
        data_ = rhs;
        return *this;
    }

private:
    friend bool operator==(const Impl& lhs, const Impl& rhs) { return lhs.data_ == rhs.data_; }
    thrust::device_vector<T, util::uninitialized_allocator<T>> data_;
};

template<class T>
DeviceVector<T>::DeviceVector()
    : impl_(new Impl())
{
}

template<class T>
DeviceVector<T>::DeviceVector(std::size_t size)
    : impl_(new Impl())
{
    impl_->resize(size);
}

template<class T>
DeviceVector<T>::DeviceVector(std::size_t size, T init)
    : impl_(new Impl())
{
    impl_->resize(size);
    thrust::fill(thrust::device, impl_->data(), impl_->data() + impl_->size(), init);
}

template<class T>
DeviceVector<T>::DeviceVector(const DeviceVector<T>& other)
    : impl_(new Impl())
{
    *impl_ = *other.impl_;
}

template<class T>
DeviceVector<T>::DeviceVector(const std::vector<T>& rhs)
    : impl_(new Impl())
{
    *impl_ = rhs;
}

template<class T>
DeviceVector<T>::DeviceVector(const T* first, const T* last)
    : impl_(new Impl())
{
    auto size = last - first;
    impl_->resize(size);
    checkGpuErrors(cudaMemcpy(impl_->data(), first, size * sizeof(T), cudaMemcpyHostToDevice));
}

template<class T>
DeviceVector<T>::~DeviceVector() = default;

template<class T>
T* DeviceVector<T>::data()
{
    return impl_->data();
}

template<class T>
const T* DeviceVector<T>::data() const
{
    return impl_->data();
}

template<class T>
void DeviceVector<T>::resize(std::size_t size)
{
    impl_->resize(size);
}

template<class T>
void DeviceVector<T>::reserve(std::size_t size)
{
    impl_->reserve(size);
}

template<class T>
void DeviceVector<T>::shrink_to_fit()
{
    impl_->shrink_to_fit();
}

template<class T>
std::size_t DeviceVector<T>::size() const
{
    return impl_->size();
}

template<class T>
bool DeviceVector<T>::empty() const
{
    return impl_->size() == 0;
}

template<class T>
std::size_t DeviceVector<T>::capacity() const
{
    return impl_->capacity();
}

template<class T>
DeviceVector<T>& DeviceVector<T>::swap(DeviceVector<T>& rhs)
{
    std::swap(impl_, rhs.impl_);
    return *this;
}

template<class T>
DeviceVector<T>& DeviceVector<T>::operator=(DeviceVector<T> rhs)
{
    this->swap(rhs);
    return *this;
}

template<class T>
DeviceVector<T>& DeviceVector<T>::operator=(const std::vector<T>& rhs)
{
    *impl_ = rhs;
    return *this;
}

template<class T>
bool operator==(const DeviceVector<T>& lhs, const DeviceVector<T>& rhs)
{
    return *lhs.impl_ == *rhs.impl_;
}

#define DEVICE_VECTOR(T)                                                                                               \
    template class DeviceVector<T>;                                                                                    \
    template bool operator==(const DeviceVector<T>&, const DeviceVector<T>&);

DEVICE_VECTOR(char);
DEVICE_VECTOR(uint8_t);
DEVICE_VECTOR(int);
DEVICE_VECTOR(unsigned);
DEVICE_VECTOR(uint64_t);
DEVICE_VECTOR(float);
DEVICE_VECTOR(double);

template class DeviceVector<util::array<int, 2>>;
template class DeviceVector<util::array<int, 3>>;
template class DeviceVector<util::array<unsigned, 1>>;
template class DeviceVector<util::array<uint64_t, 1>>;
template class DeviceVector<util::array<uint64_t, 2>>;
template class DeviceVector<util::array<unsigned, 2>>;
template class DeviceVector<util::array<float, 3>>;
template class DeviceVector<util::array<float, 4>>;
template class DeviceVector<util::array<float, 8>>;
template class DeviceVector<util::array<float, 12>>;
template class DeviceVector<util::array<double, 3>>;
template class DeviceVector<util::array<double, 4>>;
template class DeviceVector<util::array<double, 8>>;
template class DeviceVector<util::array<double, 12>>;

} // namespace cstone
