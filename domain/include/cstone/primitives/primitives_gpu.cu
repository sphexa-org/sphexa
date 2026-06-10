/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Basic algorithms on the GPU
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <stdexcept>

#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include "cstone/cuda/cub.hpp"
#include "cstone/cuda/errorcheck.cuh"
#include "cstone/cuda/thrust_util.cuh"
#include "cstone/primitives/math.hpp"
#include "cstone/util/array.hpp"
#include "primitives_gpu.h"

namespace cstone
{

template<class T>
void fillGpu(T* first, T* last, T value, execution::Gpu exec)
{
    if (last <= first) { return; }
    thrust::fill(devicePar(exec), first, last, value);
}

template void fillGpu(double*, double*, double, execution::Gpu);
template void fillGpu(float*, float*, float, execution::Gpu);
template void fillGpu(int*, int*, int, execution::Gpu);
template void fillGpu(uint8_t*, uint8_t*, uint8_t, execution::Gpu);
template void fillGpu(char*, char*, char, execution::Gpu);
template void fillGpu(unsigned*, unsigned*, unsigned, execution::Gpu);
template void fillGpu(uint64_t*, uint64_t*, uint64_t, execution::Gpu);

template<class T>
struct ScaleFunctor
{
    const T s;

    ScaleFunctor(T s_)
        : s(s_)
    {
    }

    __host__ __device__ T operator()(const T& x) const { return s * x; }
};

template<class T1, class T2, class T3>
void scaleGpu(const T1* in1, const T1* in2, T2* out, T3 value, execution::Gpu exec)
{
    thrust::transform(devicePar(exec), in1, in2, out, ScaleFunctor<T3>(value));
}

template void scaleGpu(const double*, const double*, double*, double, execution::Gpu);
template void scaleGpu(const float*, const float*, float*, double, execution::Gpu);
template void scaleGpu(const float*, const float*, float*, float, execution::Gpu);

template<class TS, class TD, class IndexType>
__global__ void gatherGpuKernel(const IndexType* map, size_t n, const TS* source, TD* destination)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) { destination[tid] = source[map[tid]]; }
}

template<class TS, class TD, class IndexType>
void gatherGpu(const IndexType* map, size_t n, const TS* source, TD* destination, execution::Gpu exec)
{
    int numThreads = 256;
    int numBlocks  = iceil(n, numThreads);

    if (numBlocks == 0) { return; }
    gatherGpuKernel<<<numBlocks, numThreads, 0, exec>>>(map, n, source, destination);
}

template void gatherGpu(const int*, size_t, const uint8_t*, uint32_t*, execution::Gpu);
template void gatherGpu(const int*, size_t, const int*, int*, execution::Gpu);
template void gatherGpu(const int*, size_t, const uint32_t*, uint32_t*, execution::Gpu);
template void gatherGpu(const int*, size_t, const uint64_t*, uint64_t*, execution::Gpu);
template void gatherGpu(const int*, size_t, const util::array<float, 3>*, util::array<float, 3>*, execution::Gpu);
template void gatherGpu(const int*, size_t, const util::array<float, 4>*, util::array<float, 4>*, execution::Gpu);
template void gatherGpu(const int*, size_t, const util::array<float, 8>*, util::array<float, 8>*, execution::Gpu);
template void gatherGpu(const int*, size_t, const util::array<float, 12>*, util::array<float, 12>*, execution::Gpu);
template void gatherGpu(const int*, size_t, const util::array<double, 3>*, util::array<double, 3>*, execution::Gpu);
template void gatherGpu(const int*, size_t, const util::array<double, 4>*, util::array<double, 4>*, execution::Gpu);
template void gatherGpu(const int*, size_t, const util::array<double, 8>*, util::array<double, 8>*, execution::Gpu);
template void gatherGpu(const int*, size_t, const util::array<double, 12>*, util::array<double, 12>*, execution::Gpu);

template void gatherGpu(const unsigned*, size_t, const uint8_t*, uint8_t*, execution::Gpu);
template void gatherGpu(const unsigned*, size_t, const double*, double*, execution::Gpu);
template void gatherGpu(const unsigned*, size_t, const float*, float*, execution::Gpu);
template void gatherGpu(const unsigned*, size_t, const char*, char*, execution::Gpu);
template void gatherGpu(const unsigned*, size_t, const int*, int*, execution::Gpu);
template void gatherGpu(const unsigned*, size_t, const long*, long*, execution::Gpu);
template void gatherGpu(const unsigned*, size_t, const unsigned*, unsigned*, execution::Gpu);
template void gatherGpu(const unsigned*, size_t, const unsigned long*, unsigned long*, execution::Gpu);
template void gatherGpu(const unsigned*, size_t, const unsigned long long*, unsigned long long*, execution::Gpu);
template void gatherGpu(const unsigned*, size_t, const util::array<float, 1>*, util::array<float, 1>*, execution::Gpu);
template void gatherGpu(const unsigned*, size_t, const util::array<float, 2>*, util::array<float, 2>*, execution::Gpu);
template void gatherGpu(const unsigned*, size_t, const util::array<float, 3>*, util::array<float, 3>*, execution::Gpu);
template void gatherGpu(const unsigned*, size_t, const util::array<float, 4>*, util::array<float, 4>*, execution::Gpu);

template<class T, class IndexType>
__global__ void scatterGpuKernel(const IndexType* map, size_t n, const T* source, T* destination)
{
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) { destination[map[tid]] = source[tid]; }
}

template<class T, class IndexType>
void scatterGpu(const IndexType* map, size_t n, const T* source, T* destination, execution::Gpu exec)
{
    int numThreads = 256;
    int numBlocks  = iceil(n, numThreads);

    if (numBlocks == 0) { return; }
    scatterGpuKernel<<<numBlocks, numThreads, 0, exec>>>(map, n, source, destination);
}

template void scatterGpu(const int*, size_t, const int*, int*, execution::Gpu);
template void scatterGpu(const int*, size_t, const uint32_t*, uint32_t*, execution::Gpu);
template void scatterGpu(const int*, size_t, const uint64_t*, uint64_t*, execution::Gpu);
template void scatterGpu(const int*, size_t, const util::array<float, 4>*, util::array<float, 4>*, execution::Gpu);
template void scatterGpu(const int*, size_t, const util::array<float, 8>*, util::array<float, 8>*, execution::Gpu);
template void scatterGpu(const int*, size_t, const util::array<float, 12>*, util::array<float, 12>*, execution::Gpu);
template void scatterGpu(const int*, size_t, const util::array<double, 4>*, util::array<double, 4>*, execution::Gpu);
template void scatterGpu(const int*, size_t, const util::array<double, 8>*, util::array<double, 8>*, execution::Gpu);
template void scatterGpu(const int*, size_t, const util::array<double, 12>*, util::array<double, 12>*, execution::Gpu);

template<class T, class IndexType>
__global__ void
gatherScatterGpuKernel(const IndexType* gmap, const IndexType* smap, size_t n, const T* source, T* destination)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) { destination[smap[tid]] = source[gmap[tid]]; }
}

template<class T, class IndexType>
void gatherScatterGpu(
    const IndexType* gmap, const IndexType* smap, size_t n, const T* source, T* destination, execution::Gpu exec)
{
    int numThreads = 256;
    int numBlocks  = iceil(n, numThreads);

    if (numBlocks == 0) { return; }
    gatherScatterGpuKernel<<<numBlocks, numThreads, 0, exec>>>(gmap, smap, n, source, destination);
}

template void gatherScatterGpu(const int*, const int*, size_t, const int*, int*, execution::Gpu);
template void gatherScatterGpu(const int*, const int*, size_t, const uint32_t*, uint32_t*, execution::Gpu);
template void gatherScatterGpu(const int*, const int*, size_t, const uint64_t*, uint64_t*, execution::Gpu);
template void
gatherScatterGpu(const int*, const int*, size_t, const util::array<float, 4>*, util::array<float, 4>*, execution::Gpu);
template void
gatherScatterGpu(const int*, const int*, size_t, const util::array<float, 8>*, util::array<float, 8>*, execution::Gpu);
template void gatherScatterGpu(
    const int*, const int*, size_t, const util::array<float, 12>*, util::array<float, 12>*, execution::Gpu);
template void gatherScatterGpu(
    const int*, const int*, size_t, const util::array<double, 4>*, util::array<double, 4>*, execution::Gpu);
template void gatherScatterGpu(
    const int*, const int*, size_t, const util::array<double, 8>*, util::array<double, 8>*, execution::Gpu);
template void gatherScatterGpu(
    const int*, const int*, size_t, const util::array<double, 12>*, util::array<double, 12>*, execution::Gpu);

template<class T>
std::tuple<T, T> MinMax<execution::Gpu, T>::operator()(const T* first, const T* last)
{
    auto minMax = thrust::minmax_element(devicePar(exec), first, last);

    T theMinimum, theMaximum;
    checkGpuErrors(cudaMemcpyAsync(&theMinimum, minMax.first, sizeof(T), cudaMemcpyDeviceToHost, exec));
    checkGpuErrors(cudaMemcpyAsync(&theMaximum, minMax.second, sizeof(T), cudaMemcpyDeviceToHost, exec));
    checkGpuErrors(cudaStreamSynchronize(exec));

    return std::make_tuple(theMinimum, theMaximum);
}

template struct MinMax<execution::Gpu, double>;
template struct MinMax<execution::Gpu, float>;
template struct MinMax<execution::Gpu, unsigned>;

using thrust::get;

template<class T>
struct NormSquare3D
{
    HOST_DEVICE_FUN T operator()(const thrust::tuple<T, T, T>& X)
    {
        return get<0>(X) * get<0>(X) + get<1>(X) * get<1>(X) + get<2>(X) * get<2>(X);
    }
};

template<class T>
T maxNormSquareGpu(const T* x, const T* y, const T* z, size_t numElements, execution::Gpu exec)
{
    auto it1 = thrust::make_zip_iterator(x, y, z);
    auto it2 = thrust::make_zip_iterator(x + numElements, y + numElements, z + numElements);

    T init = 0;

    return thrust::transform_reduce(devicePar(exec), it1, it2, NormSquare3D<T>{}, init, thrust::maximum<T>{});
}

template float maxNormSquareGpu(const float*, const float*, const float*, size_t, execution::Gpu);
template double maxNormSquareGpu(const double*, const double*, const double*, size_t, execution::Gpu);

template<class T>
size_t lowerBoundGpu(const T* first, const T* last, T value, execution::Gpu exec)
{
    return thrust::lower_bound(devicePar(exec), first, last, value) - first;
}

template size_t lowerBoundGpu(const unsigned*, const unsigned*, unsigned, execution::Gpu);
template size_t lowerBoundGpu(const uint64_t*, const uint64_t*, uint64_t, execution::Gpu);
template size_t lowerBoundGpu(const int*, const int*, int, execution::Gpu);
template size_t lowerBoundGpu(const int64_t*, const int64_t*, int64_t, execution::Gpu);
template size_t lowerBoundGpu(const float*, const float*, float, execution::Gpu);

template<class T, class IndexType>
void lowerBoundGpu(
    const T* first, const T* last, const T* valueFirst, const T* valueLast, IndexType* result, execution::Gpu exec)
{
    thrust::lower_bound(devicePar(exec), first, last, valueFirst, valueLast, result);
}

template void
lowerBoundGpu(const unsigned*, const unsigned*, const unsigned*, const unsigned*, unsigned*, execution::Gpu);
template void
lowerBoundGpu(const uint64_t*, const uint64_t*, const uint64_t*, const uint64_t*, unsigned*, execution::Gpu);
template void
lowerBoundGpu(const unsigned*, const unsigned*, const unsigned*, const unsigned*, uint64_t*, execution::Gpu);
template void
lowerBoundGpu(const uint64_t*, const uint64_t*, const uint64_t*, const uint64_t*, uint64_t*, execution::Gpu);

template<class T1, class T2, class Tout>
void sequenceMax(const T1* i1_begin, const T1* i1_end, const T2* i2, Tout* output, execution::Gpu exec)
{
    thrust::transform(devicePar(exec), i1_begin, i1_end, i2, output, thrust::maximum<unsigned>{});
}

template void sequenceMax(const unsigned*, const unsigned*, const unsigned*, unsigned*, execution::Gpu);

template<class Tin, class Tout>
Tout reduceGpu(const Tin* input, size_t numElements, Tout init, execution::Gpu exec)
{
    return thrust::reduce(devicePar(exec), input, input + numElements, init);
}

template size_t reduceGpu(const unsigned*, size_t, size_t, execution::Gpu);

template<class IndexType>
void sequenceGpu(IndexType* input, size_t numElements, IndexType init, execution::Gpu exec)
{
    thrust::sequence(devicePar(exec), input, input + numElements, init);
}

template void sequenceGpu(int*, size_t, int, execution::Gpu);
template void sequenceGpu(unsigned*, size_t, unsigned, execution::Gpu);
template void sequenceGpu(uint64_t*, uint64_t, uint64_t, execution::Gpu);

template<class KeyType>
void sortGpu(KeyType* first, KeyType* last, KeyType* keyBuf, execution::Gpu exec)
{
    size_t numElements = last - first;

    cub::DoubleBuffer<KeyType> d_keys(first, keyBuf);

    // Determine temporary device storage requirements
    void* d_tempStorage     = nullptr;
    size_t tempStorageBytes = 0;
    checkGpuErrors(cub::DeviceRadixSort::SortKeys(d_tempStorage, tempStorageBytes, d_keys, numElements, 0,
                                                  sizeof(KeyType) * 8, exec));

    // Allocate temporary storage
    checkGpuErrors(cudaMalloc(&d_tempStorage, tempStorageBytes));

    // Run sorting operation
    checkGpuErrors(cub::DeviceRadixSort::SortKeys(d_tempStorage, tempStorageBytes, d_keys, numElements, 0,
                                                  sizeof(KeyType) * 8, exec));

    auto* curValues = d_keys.Current();
    if (curValues != first)
    {
        checkGpuErrors(
            cudaMemcpyAsync(first, curValues, numElements * sizeof(KeyType), cudaMemcpyDeviceToDevice, exec));
    }

    checkGpuErrors(cudaFree(d_tempStorage));
}

template void sortGpu(uint32_t*, uint32_t*, uint32_t*, execution::Gpu);
template void sortGpu(uint64_t*, uint64_t*, uint64_t*, execution::Gpu);
template void sortGpu(float*, float*, float*, execution::Gpu);

// Determine temporary device storage requirements
template<class KeyType, class ValueType>
uint64_t sortByKeyTempStorage(uint64_t numElements)
{
    cub::DoubleBuffer<KeyType> d_keys(nullptr, nullptr);
    cub::DoubleBuffer<ValueType> d_values(nullptr, nullptr);

    uint64_t tempStorageBytes = 0;
    checkGpuErrors(cub::DeviceRadixSort::SortPairs(nullptr, tempStorageBytes, d_keys, d_values, numElements, 0,
                                                   sizeof(KeyType) * 8));
    return tempStorageBytes;
}

template<class KeyType, class ValueType>
void sortByKeyGpu(KeyType* first,
                  KeyType* last,
                  ValueType* values,
                  KeyType* keyBuf,
                  ValueType* valueBuf,
                  void* d_tempStorage,
                  uint64_t tempStorageBytes,
                  execution::Gpu exec)
{
    size_t numElements = last - first;

    cub::DoubleBuffer<KeyType> d_keys(first, keyBuf);
    cub::DoubleBuffer<ValueType> d_values(values, valueBuf);

    auto tempBytesCheck = sortByKeyTempStorage<KeyType, ValueType>(numElements);
    if (tempStorageBytes < tempBytesCheck) { throw std::runtime_error("temp storage too small\n"); };

    // Run sorting operation
    checkGpuErrors(cub::DeviceRadixSort::SortPairs(d_tempStorage, tempStorageBytes, d_keys, d_values, numElements, 0,
                                                   sizeof(KeyType) * 8, exec));

    auto* curKeys = d_keys.Current();
    if (curKeys != first)
    {
        checkGpuErrors(cudaMemcpyAsync(first, curKeys, numElements * sizeof(KeyType), cudaMemcpyDeviceToDevice, exec));
    }

    auto* curValues = d_values.Current();
    if (curValues != values)
    {
        checkGpuErrors(
            cudaMemcpyAsync(values, curValues, numElements * sizeof(ValueType), cudaMemcpyDeviceToDevice, exec));
    }
}

#define SORT_BY_KEY_GPU_DB(KeyType, ValueType)                                                                         \
    template void sortByKeyGpu(KeyType*, KeyType*, ValueType*, KeyType*, ValueType*, void*, uint64_t, execution::Gpu); \
    template uint64_t sortByKeyTempStorage<KeyType, ValueType>(uint64_t)

SORT_BY_KEY_GPU_DB(unsigned, unsigned);
SORT_BY_KEY_GPU_DB(unsigned, int);
SORT_BY_KEY_GPU_DB(uint64_t, unsigned);
SORT_BY_KEY_GPU_DB(uint64_t, int);
SORT_BY_KEY_GPU_DB(uint64_t, uint64_t);
SORT_BY_KEY_GPU_DB(float, unsigned);

template<class KeyType, class ValueType>
void sortByKeyGpu(KeyType* first, KeyType* last, ValueType* values, execution::Gpu exec)
{
    thrust::sort_by_key(devicePar(exec), first, last, values);
}

template void sortByKeyGpu(unsigned*, unsigned*, unsigned*, execution::Gpu);
template void sortByKeyGpu(unsigned*, unsigned*, int*, execution::Gpu);
template void sortByKeyGpu(uint64_t*, uint64_t*, unsigned*, execution::Gpu);
template void sortByKeyGpu(uint64_t*, uint64_t*, int*, execution::Gpu);
template void sortByKeyGpu(uint64_t*, uint64_t*, uint64_t*, execution::Gpu);

template<class IndexType, class SumType>
void exclusiveScanGpu(const IndexType* first, const IndexType* last, SumType* output, SumType init, execution::Gpu exec)
{
    thrust::exclusive_scan(devicePar(exec), first, last, output, init);
}

template void exclusiveScanGpu(const int*, const int*, int*, int, execution::Gpu);
template void exclusiveScanGpu(const int*, const int*, unsigned*, unsigned, execution::Gpu);
template void exclusiveScanGpu(const int*, const int*, uint64_t*, uint64_t, execution::Gpu);
template void exclusiveScanGpu(const unsigned*, const unsigned*, unsigned*, unsigned, execution::Gpu);
template void exclusiveScanGpu(const unsigned*, const unsigned*, uint64_t*, uint64_t, execution::Gpu);

template<class IndexType, class SumType>
void inclusiveScanGpu(const IndexType* first, const IndexType* last, SumType* output, execution::Gpu exec)
{
    thrust::inclusive_scan(devicePar(exec), first, last, output);

    /*! Accumulation in 64-bit from 32-bit inputs only works by explicitly setting the type of the initial
     *  value, which is only supported in Thrust/CUB version shipped with CUDA 12.7 and later
     */
    // thrust::inclusive_scan(devicePar(exec), first, last, output, SumType(0), thrust::plus<>{});
    /*
    SumType init = 0;
    size_t temp_storage_bytes{};
    size_t num_elements = last - first;
    cub::DeviceScan::InclusiveScanInit(nullptr, temp_storage_bytes, first, output, thrust::plus<>{}, init,
                                       num_elements);

    // Allocate temporary storage for inclusive scan
    uint8_t* temp_storage;
    checkGpuErrors(cudaMalloc(&temp_storage, temp_storage_bytes));

    // Run inclusive prefix sum
    cub::DeviceScan::InclusiveScanInit(temp_storage, temp_storage_bytes, first, output, thrust::plus<>{}, init,
                                       num_elements);

    checkGpuErrors(cudaFree(temp_storage));
    */
}

template void inclusiveScanGpu(const int*, const int*, int*, execution::Gpu);
template void inclusiveScanGpu(const int*, const int*, unsigned*, execution::Gpu);
// template void inclusiveScanGpu(const int*, const int*, uint64_t*, execution::Gpu);
template void inclusiveScanGpu(const unsigned*, const unsigned*, unsigned*, execution::Gpu);
// template void inclusiveScanGpu(const unsigned*, const unsigned*, uint64_t*, execution::Gpu);

template<class ValueType>
size_t countGpu(const ValueType* first, const ValueType* last, ValueType v, execution::Gpu exec)
{
    return thrust::count(devicePar(exec), first, last, v);
}

template size_t countGpu(const int* first, const int* last, int v, execution::Gpu);
template size_t countGpu(const unsigned* first, const unsigned* last, unsigned v, execution::Gpu);
template size_t countGpu(const uint64_t* first, const uint64_t* last, uint64_t v, execution::Gpu);

template<class TS, class TD, class S>
__global__ void selectCopyKernel(const TS* src, LocalIndex n, const S* selectFlags, TD* dest)
{
    LocalIndex tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n && selectFlags[tid]) { dest[tid] = src[tid]; }
}

template<class TS, class TD, class S>
void selectCopyGpu(const TS* src, LocalIndex n, const S* selectFlags, TD* dest, execution::Gpu exec)
{
    int numThreads = 256;
    int numBlocks  = iceil(n, numThreads);
    if (numBlocks == 0) { return; }
    selectCopyKernel<<<numBlocks, numThreads, 0, exec>>>(src, n, selectFlags, dest);
}

template void selectCopyGpu(const int*, LocalIndex, const unsigned*, unsigned*, execution::Gpu);
template void selectCopyGpu(const unsigned*, LocalIndex, const unsigned*, unsigned*, execution::Gpu);

} // namespace cstone
