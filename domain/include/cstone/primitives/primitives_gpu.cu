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
void fill(execution::Gpu exec, T* first, T* last, T value)
{
    if (last <= first) { return; }
    thrust::fill(thrustExecPolicy(exec), first, last, value);
}

template void fill(execution::Gpu, double*, double*, double);
template void fill(execution::Gpu, float*, float*, float);
template void fill(execution::Gpu, int*, int*, int);
template void fill(execution::Gpu, uint8_t*, uint8_t*, uint8_t);
template void fill(execution::Gpu, char*, char*, char);
template void fill(execution::Gpu, unsigned*, unsigned*, unsigned);
template void fill(execution::Gpu, uint64_t*, uint64_t*, uint64_t);

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
void scale(execution::Gpu exec, const T1* in1, const T1* in2, T2* out, T3 value)
{
    thrust::transform(thrustExecPolicy(exec), in1, in2, out, ScaleFunctor<T3>(value));
}

template void scale(execution::Gpu, const double*, const double*, double*, double);
template void scale(execution::Gpu, const float*, const float*, float*, double);
template void scale(execution::Gpu, const float*, const float*, float*, float);

template<class TS, class TD, class IndexType>
__global__ void gatherGpuKernel(const IndexType* map, size_t n, const TS* source, TD* destination)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) { destination[tid] = source[map[tid]]; }
}

template<class TS, class TD, class IndexType>
void gather(execution::Gpu exec, const IndexType* map, size_t n, const TS* source, TD* destination)
{
    int numThreads = 256;
    int numBlocks  = iceil(n, numThreads);

    if (numBlocks == 0) { return; }
    gatherGpuKernel<<<numBlocks, numThreads, 0, exec>>>(map, n, source, destination);
}

template void gather(execution::Gpu, const int*, size_t, const uint8_t*, uint32_t*);
template void gather(execution::Gpu, const int*, size_t, const int*, int*);
template void gather(execution::Gpu, const int*, size_t, const uint32_t*, uint32_t*);
template void gather(execution::Gpu, const int*, size_t, const uint64_t*, uint64_t*);
template void gather(execution::Gpu, const int*, size_t, const util::array<float, 3>*, util::array<float, 3>*);
template void gather(execution::Gpu, const int*, size_t, const util::array<float, 4>*, util::array<float, 4>*);
template void gather(execution::Gpu, const int*, size_t, const util::array<float, 8>*, util::array<float, 8>*);
template void gather(execution::Gpu, const int*, size_t, const util::array<float, 12>*, util::array<float, 12>*);
template void gather(execution::Gpu, const int*, size_t, const util::array<double, 3>*, util::array<double, 3>*);
template void gather(execution::Gpu, const int*, size_t, const util::array<double, 4>*, util::array<double, 4>*);
template void gather(execution::Gpu, const int*, size_t, const util::array<double, 8>*, util::array<double, 8>*);
template void gather(execution::Gpu, const int*, size_t, const util::array<double, 12>*, util::array<double, 12>*);

template void gather(execution::Gpu, const unsigned*, size_t, const uint8_t*, uint8_t*);
template void gather(execution::Gpu, const unsigned*, size_t, const double*, double*);
template void gather(execution::Gpu, const unsigned*, size_t, const float*, float*);
template void gather(execution::Gpu, const unsigned*, size_t, const char*, char*);
template void gather(execution::Gpu, const unsigned*, size_t, const int*, int*);
template void gather(execution::Gpu, const unsigned*, size_t, const long*, long*);
template void gather(execution::Gpu, const unsigned*, size_t, const unsigned*, unsigned*);
template void gather(execution::Gpu, const unsigned*, size_t, const unsigned long*, unsigned long*);
template void gather(execution::Gpu, const unsigned*, size_t, const unsigned long long*, unsigned long long*);
template void gather(execution::Gpu, const unsigned*, size_t, const util::array<float, 1>*, util::array<float, 1>*);
template void gather(execution::Gpu, const unsigned*, size_t, const util::array<float, 2>*, util::array<float, 2>*);
template void gather(execution::Gpu, const unsigned*, size_t, const util::array<float, 3>*, util::array<float, 3>*);
template void gather(execution::Gpu, const unsigned*, size_t, const util::array<float, 4>*, util::array<float, 4>*);

template<class T, class IndexType>
__global__ void scatterGpuKernel(const IndexType* map, size_t n, const T* source, T* destination)
{
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) { destination[map[tid]] = source[tid]; }
}

template<class T, class IndexType>
void scatter(execution::Gpu exec, const IndexType* map, size_t n, const T* source, T* destination)
{
    int numThreads = 256;
    int numBlocks  = iceil(n, numThreads);

    if (numBlocks == 0) { return; }
    scatterGpuKernel<<<numBlocks, numThreads, 0, exec>>>(map, n, source, destination);
}

template void scatter(execution::Gpu, const int*, size_t, const int*, int*);
template void scatter(execution::Gpu, const int*, size_t, const uint32_t*, uint32_t*);
template void scatter(execution::Gpu, const int*, size_t, const uint64_t*, uint64_t*);
template void scatter(execution::Gpu, const int*, size_t, const util::array<float, 4>*, util::array<float, 4>*);
template void scatter(execution::Gpu, const int*, size_t, const util::array<float, 8>*, util::array<float, 8>*);
template void scatter(execution::Gpu, const int*, size_t, const util::array<float, 12>*, util::array<float, 12>*);
template void scatter(execution::Gpu, const int*, size_t, const util::array<double, 4>*, util::array<double, 4>*);
template void scatter(execution::Gpu, const int*, size_t, const util::array<double, 8>*, util::array<double, 8>*);
template void scatter(execution::Gpu, const int*, size_t, const util::array<double, 12>*, util::array<double, 12>*);

template<class T, class IndexType>
__global__ void
gatherScatterGpuKernel(const IndexType* gmap, const IndexType* smap, size_t n, const T* source, T* destination)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) { destination[smap[tid]] = source[gmap[tid]]; }
}

template<class T, class IndexType>
void gatherScatter(
    execution::Gpu exec, const IndexType* gmap, const IndexType* smap, size_t n, const T* source, T* destination)
{
    int numThreads = 256;
    int numBlocks  = iceil(n, numThreads);

    if (numBlocks == 0) { return; }
    gatherScatterGpuKernel<<<numBlocks, numThreads, 0, exec>>>(gmap, smap, n, source, destination);
}

template void gatherScatter(execution::Gpu, const int*, const int*, size_t, const int*, int*);
template void gatherScatter(execution::Gpu, const int*, const int*, size_t, const uint32_t*, uint32_t*);
template void gatherScatter(execution::Gpu, const int*, const int*, size_t, const uint64_t*, uint64_t*);
template void
gatherScatter(execution::Gpu, const int*, const int*, size_t, const util::array<float, 4>*, util::array<float, 4>*);
template void
gatherScatter(execution::Gpu, const int*, const int*, size_t, const util::array<float, 8>*, util::array<float, 8>*);
template void
gatherScatter(execution::Gpu, const int*, const int*, size_t, const util::array<float, 12>*, util::array<float, 12>*);
template void
gatherScatter(execution::Gpu, const int*, const int*, size_t, const util::array<double, 4>*, util::array<double, 4>*);
template void
gatherScatter(execution::Gpu, const int*, const int*, size_t, const util::array<double, 8>*, util::array<double, 8>*);
template void
gatherScatter(execution::Gpu, const int*, const int*, size_t, const util::array<double, 12>*, util::array<double, 12>*);

template<class T>
std::tuple<T, T> minMax(execution::Gpu exec, const T* first, const T* last)
{
    auto minMaxElements = thrust::minmax_element(thrustExecPolicy(exec), first, last);

    T theMinimum, theMaximum;
    checkGpuErrors(cudaMemcpyAsync(&theMinimum, minMaxElements.first, sizeof(T), cudaMemcpyDeviceToHost, exec));
    checkGpuErrors(cudaMemcpyAsync(&theMaximum, minMaxElements.second, sizeof(T), cudaMemcpyDeviceToHost, exec));
    checkGpuErrors(cudaStreamSynchronize(exec));

    return std::make_tuple(theMinimum, theMaximum);
}

template std::tuple<double, double> minMax(execution::Gpu, const double*, const double*);
template std::tuple<float, float> minMax(execution::Gpu, const float*, const float*);
template std::tuple<unsigned, unsigned> minMax(execution::Gpu, const unsigned*, const unsigned*);

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
T maxNormSquare(execution::Gpu exec, const T* x, const T* y, const T* z, size_t numElements)
{
    auto it1 = thrust::make_zip_iterator(x, y, z);
    auto it2 = thrust::make_zip_iterator(x + numElements, y + numElements, z + numElements);

    T init = 0;

    return thrust::transform_reduce(thrustExecPolicy(exec), it1, it2, NormSquare3D<T>{}, init, thrust::maximum<T>{});
}

template float maxNormSquare(execution::Gpu, const float*, const float*, const float*, size_t);
template double maxNormSquare(execution::Gpu, const double*, const double*, const double*, size_t);

template<class T>
size_t lowerBound(execution::Gpu exec, const T* first, const T* last, T value)
{
    return thrust::lower_bound(thrustExecPolicy(exec), first, last, value) - first;
}

template size_t lowerBound(execution::Gpu, const unsigned*, const unsigned*, unsigned);
template size_t lowerBound(execution::Gpu, const uint64_t*, const uint64_t*, uint64_t);
template size_t lowerBound(execution::Gpu, const int*, const int*, int);
template size_t lowerBound(execution::Gpu, const int64_t*, const int64_t*, int64_t);
template size_t lowerBound(execution::Gpu, const float*, const float*, float);

template<class T, class IndexType>
void lowerBound(
    execution::Gpu exec, const T* first, const T* last, const T* valueFirst, const T* valueLast, IndexType* result)
{
    thrust::lower_bound(thrustExecPolicy(exec), first, last, valueFirst, valueLast, result);
}

template void lowerBound(execution::Gpu, const unsigned*, const unsigned*, const unsigned*, const unsigned*, unsigned*);
template void lowerBound(execution::Gpu, const uint64_t*, const uint64_t*, const uint64_t*, const uint64_t*, unsigned*);
template void lowerBound(execution::Gpu, const unsigned*, const unsigned*, const unsigned*, const unsigned*, uint64_t*);
template void lowerBound(execution::Gpu, const uint64_t*, const uint64_t*, const uint64_t*, const uint64_t*, uint64_t*);

template<class T1, class T2, class Tout>
void sequenceMax(execution::Gpu exec, const T1* i1_begin, const T1* i1_end, const T2* i2, Tout* output)
{
    thrust::transform(thrustExecPolicy(exec), i1_begin, i1_end, i2, output, thrust::maximum<unsigned>{});
}

template void sequenceMax(execution::Gpu, const unsigned*, const unsigned*, const unsigned*, unsigned*);

template<class Tin, class Tout>
Tout reduce(execution::Gpu exec, const Tin* input, size_t numElements, Tout init)
{
    return thrust::reduce(thrustExecPolicy(exec), input, input + numElements, init);
}

template size_t reduce(execution::Gpu, const unsigned*, size_t, size_t);

template<class IndexType>
void sequence(execution::Gpu exec, IndexType* input, size_t numElements, IndexType init)
{
    thrust::sequence(thrustExecPolicy(exec), input, input + numElements, init);
}

template void sequence(execution::Gpu, int*, size_t, int);
template void sequence(execution::Gpu, unsigned*, size_t, unsigned);
template void sequence(execution::Gpu, uint64_t*, uint64_t, uint64_t);

template<class KeyType>
void sort(execution::Gpu exec, KeyType* first, KeyType* last, KeyType* keyBuf)
{
    size_t numElements = last - first;

    cub::DoubleBuffer<KeyType> d_keys(first, keyBuf);

    // Determine temporary device storage requirements
    void* d_tempStorage     = nullptr;
    size_t tempStorageBytes = 0;
    checkGpuErrors(cub::DeviceRadixSort::SortKeys(d_tempStorage, tempStorageBytes, d_keys, numElements, 0,
                                                  sizeof(KeyType) * 8, exec));

    // Allocate temporary storage
    checkGpuErrors(cudaMallocAsync(&d_tempStorage, tempStorageBytes, exec));

    // Run sorting operation
    checkGpuErrors(cub::DeviceRadixSort::SortKeys(d_tempStorage, tempStorageBytes, d_keys, numElements, 0,
                                                  sizeof(KeyType) * 8, exec));

    auto* curValues = d_keys.Current();
    if (curValues != first)
    {
        checkGpuErrors(
            cudaMemcpyAsync(first, curValues, numElements * sizeof(KeyType), cudaMemcpyDeviceToDevice, exec));
    }

    checkGpuErrors(cudaFreeAsync(d_tempStorage, exec));
}

template void sort(execution::Gpu, uint32_t*, uint32_t*, uint32_t*);
template void sort(execution::Gpu, uint64_t*, uint64_t*, uint64_t*);
template void sort(execution::Gpu, float*, float*, float*);

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
void sortByKey(execution::Gpu exec,
               KeyType* first,
               KeyType* last,
               ValueType* values,
               KeyType* keyBuf,
               ValueType* valueBuf,
               void* d_tempStorage,
               uint64_t tempStorageBytes)
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
    template void sortByKey(execution::Gpu, KeyType*, KeyType*, ValueType*, KeyType*, ValueType*, void*, uint64_t);    \
    template uint64_t sortByKeyTempStorage<KeyType, ValueType>(uint64_t)

SORT_BY_KEY_GPU_DB(unsigned, unsigned);
SORT_BY_KEY_GPU_DB(unsigned, int);
SORT_BY_KEY_GPU_DB(uint64_t, unsigned);
SORT_BY_KEY_GPU_DB(uint64_t, int);
SORT_BY_KEY_GPU_DB(uint64_t, uint64_t);
SORT_BY_KEY_GPU_DB(float, unsigned);

template<class IndexType, class SumType>
void exclusiveScan(execution::Gpu exec, const IndexType* first, const IndexType* last, SumType* output, SumType init)
{
    thrust::exclusive_scan(thrustExecPolicy(exec), first, last, output, init);
}

template void exclusiveScan(execution::Gpu, const int*, const int*, int*, int);
template void exclusiveScan(execution::Gpu, const int*, const int*, unsigned*, unsigned);
template void exclusiveScan(execution::Gpu, const int*, const int*, uint64_t*, uint64_t);
template void exclusiveScan(execution::Gpu, const unsigned*, const unsigned*, unsigned*, unsigned);
template void exclusiveScan(execution::Gpu, const unsigned*, const unsigned*, uint64_t*, uint64_t);

template<class IndexType, class SumType>
void inclusiveScan(execution::Gpu exec, const IndexType* first, const IndexType* last, SumType* output)
{
    thrust::inclusive_scan(thrustExecPolicy(exec), first, last, output);

    /*! Accumulation in 64-bit from 32-bit inputs only works by explicitly setting the type of the initial
     *  value, which is only supported in Thrust/CUB version shipped with CUDA 12.7 and later
     */
    // thrust::inclusive_scan(thrustExecPolicy(exec), first, last, output, SumType(0), thrust::plus<>{});
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

template void inclusiveScan(execution::Gpu, const int*, const int*, int*);
template void inclusiveScan(execution::Gpu, const int*, const int*, unsigned*);
// template void inclusiveScan(execution::Gpu, const int*, const int*, uint64_t*);
template void inclusiveScan(execution::Gpu, const unsigned*, const unsigned*, unsigned*);
// template void inclusiveScan(execution::Gpu, const unsigned*, const unsigned*, uint64_t*);

template<class ValueType>
size_t count(execution::Gpu exec, const ValueType* first, const ValueType* last, ValueType v)
{
    return thrust::count(thrustExecPolicy(exec), first, last, v);
}

template size_t count(execution::Gpu, const int* first, const int* last, int v);
template size_t count(execution::Gpu, const unsigned* first, const unsigned* last, unsigned v);
template size_t count(execution::Gpu, const uint64_t* first, const uint64_t* last, uint64_t v);

template<class TS, class TD, class S>
__global__ void selectCopyKernel(const TS* src, LocalIndex n, const S* selectFlags, TD* dest)
{
    LocalIndex tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n && selectFlags[tid]) { dest[tid] = src[tid]; }
}

template<class TS, class TD, class S>
void selectCopy(execution::Gpu exec, const TS* src, LocalIndex n, const S* selectFlags, TD* dest)
{
    int numThreads = 256;
    int numBlocks  = iceil(n, numThreads);
    if (numBlocks == 0) { return; }
    selectCopyKernel<<<numBlocks, numThreads, 0, exec>>>(src, n, selectFlags, dest);
}

template void selectCopy(execution::Gpu, const int*, LocalIndex, const unsigned*, unsigned*);
template void selectCopy(execution::Gpu, const unsigned*, LocalIndex, const unsigned*, unsigned*);

} // namespace cstone
