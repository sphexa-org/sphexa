#include "hip/hip_runtime.h"
/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
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
 * @brief  Exposes gather functionality to reorder arrays by a map
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <vector>

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "errorcheck.cuh"
#include "gather.cuh"

namespace cstone
{

template<class LocalIndex>
class DeviceMemory
{
public:
    static constexpr size_t ElementSize = 8;

    DeviceMemory() = default;

    ~DeviceMemory()
    {
        if (allocatedSize_ > 0)
        {
            checkGpuErrors(hipFree(d_ordering_));
            checkGpuErrors(hipFree(d_buffer_));
        }
    }

    void reallocate(std::size_t newSize)
    {
        if (newSize > allocatedSize_)
        {
            // allocate 5% extra to avoid reallocation on small increase
            newSize = double(newSize) * 1.05;
            // round up newSize to next 4K boundary
            newSize += newSize % alignment;

            if (allocatedSize_ > 0)
            {
                checkGpuErrors(hipFree(d_ordering_));
                checkGpuErrors(hipFree(d_buffer_));
            }

            checkGpuErrors(hipMalloc((void**)&d_ordering_, newSize * sizeof(LocalIndex)));
            checkGpuErrors(hipMalloc((void**)&(d_buffer_), 2 * newSize * ElementSize));

            allocatedSize_ = newSize;
        }
    }

    LocalIndex* ordering() { return d_ordering_; }

    char* deviceBuffer(int i)
    {
        if (i > 1) throw std::runtime_error("buffer index out of bounds\n");
        return d_buffer_ + i * allocatedSize_ * ElementSize;
    }

private:
    static constexpr int alignment = 4096 / ElementSize;

    std::size_t allocatedSize_{0};

    //! @brief reorder map
    LocalIndex* d_ordering_;

    //! @brief device buffers
    char* d_buffer_;
};

template<class KeyType, class IndexType>
DeviceGather<KeyType, IndexType>::DeviceGather()
    : deviceMemory_(std::make_unique<DeviceMemory<IndexType>>())
{
}

template<class KeyType, class IndexType>
void DeviceGather<KeyType, IndexType>::setReorderMap(const IndexType* map_first, const IndexType* map_last)
{
    mapSize_ = map_last - map_first;
    deviceMemory_->reallocate(mapSize_);
    // upload new ordering to the device
    hipMemcpy(deviceMemory_->ordering(), map_first, mapSize_ * sizeof(IndexType), hipMemcpyHostToDevice);
    checkGpuErrors(hipGetLastError());
}

template<class KeyType, class IndexType>
void DeviceGather<KeyType, IndexType>::getReorderMap(IndexType* map_first, IndexType first, IndexType last)
{
    hipMemcpy(map_first, deviceMemory_->ordering() + first, (last - first) * sizeof(IndexType),
               hipMemcpyDeviceToHost);
}

template<class I>
__global__ void iotaKernel(I* buffer, size_t n, size_t offset)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) { buffer[tid] = offset + tid; }
}

template<class KeyType, class IndexType>
void DeviceGather<KeyType, IndexType>::setMapFromCodes(KeyType* codes_first, KeyType* codes_last)
{
    offset_     = 0;
    mapSize_    = codes_last - codes_first;
    numExtract_ = mapSize_;
    deviceMemory_->reallocate(mapSize_);

    // the deviceBuffer is allocated as a single chunk of size 2 * mapSize_ * sizeof(T)
    // so we can reuse it for mapSize_ elements of KeyType, as long as the static assert holds
    static_assert(sizeof(KeyType) <= 2 * DeviceMemory<IndexType>::ElementSize,
                  "buffer size not big enough for codes device array\n");
    KeyType* d_codes = reinterpret_cast<KeyType*>(deviceMemory_->deviceBuffer(0));

    // send Morton codes to the device
    hipMemcpy(d_codes, codes_first, mapSize_ * sizeof(KeyType), hipMemcpyHostToDevice);
    checkGpuErrors(hipGetLastError());

    constexpr int nThreads = 256;
    int nBlocks            = (mapSize_ + nThreads - 1) / nThreads;
    hipLaunchKernelGGL(iotaKernel, nBlocks, nThreads, 0, 0, deviceMemory_->ordering(), mapSize_, 0);
    checkGpuErrors(hipGetLastError());

    // sort SFC keys on device, track new ordering on the device
    thrust::sort_by_key(thrust::device, thrust::device_pointer_cast(d_codes),
                        thrust::device_pointer_cast(d_codes + mapSize_),
                        thrust::device_pointer_cast(deviceMemory_->ordering()));
    checkGpuErrors(hipGetLastError());

    // send sorted codes back to host
    hipMemcpy(codes_first, d_codes, mapSize_ * sizeof(KeyType), hipMemcpyDeviceToHost);
    checkGpuErrors(hipGetLastError());
}

template<class KeyType, class IndexType>
DeviceGather<KeyType, IndexType>::~DeviceGather() = default;

template<class T, class I>
__global__ void reorder(I* map, T* source, T* destination, size_t n)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) { destination[tid] = source[map[tid]]; }
}

template<class KeyType, class IndexType>
template<class T>
void DeviceGather<KeyType, IndexType>::operator()(const T* values,
                                                  T* destination,
                                                  IndexType offset,
                                                  IndexType numExtract) const
{
    static_assert(sizeof(T) <= DeviceMemory<IndexType>::ElementSize);

    constexpr int nThreads = 256;
    int nBlocks            = (numExtract + nThreads - 1) / nThreads;

    // upload to device
    hipMemcpy(deviceMemory_->deviceBuffer(0), values, mapSize_ * sizeof(T), hipMemcpyHostToDevice);
    checkGpuErrors(hipGetLastError());

    // reorder on device
    hipLaunchKernelGGL(reorder, nBlocks, nThreads, 0, 0, deviceMemory_->ordering() + offset,
                                   reinterpret_cast<T*>(deviceMemory_->deviceBuffer(0)),
                                   reinterpret_cast<T*>(deviceMemory_->deviceBuffer(1)), numExtract);
    checkGpuErrors(hipGetLastError());

    // download to host
    hipMemcpy(destination, deviceMemory_->deviceBuffer(1), numExtract * sizeof(T), hipMemcpyDeviceToHost);
    checkGpuErrors(hipGetLastError());
}

template<class KeyType, class IndexType>
template<class T>
void DeviceGather<KeyType, IndexType>::operator()(const T* values, T* destination) const
{
    this->operator()(values, destination, offset_, numExtract_);
}

template<class KeyType, class IndexType>
void DeviceGather<KeyType, IndexType>::restrictRange(std::size_t offset, std::size_t numExtract)
{
    assert(offset + numExtract <= mapSize_);

    offset_     = offset;
    numExtract_ = numExtract;
}

template class DeviceGather<unsigned, unsigned>;
template class DeviceGather<uint64_t, unsigned>;

template void DeviceGather<unsigned, unsigned>::operator()(const double*, double*) const;
template void DeviceGather<unsigned, unsigned>::operator()(const float*, float*) const;

template void DeviceGather<uint64_t, unsigned>::operator()(const double*, double*) const;
template void DeviceGather<uint64_t, unsigned>::operator()(const float*, float*) const;

} // namespace cstone