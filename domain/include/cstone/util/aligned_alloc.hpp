/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Aligned allocator
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <cstdlib>
#include <new>
#include <memory>

namespace util
{

// Alignment must be a power of 2 !
template<typename T, unsigned int Alignment>
class AlignedAllocator
{
public:
    typedef T* pointer;
    typedef T const* const_pointer;
    typedef T& reference;
    typedef T const& const_reference;
    typedef T value_type;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    template<typename U>
    struct rebind
    {
        typedef AlignedAllocator<U, Alignment> other;
    };

    AlignedAllocator() noexcept {}

    AlignedAllocator(AlignedAllocator const&) noexcept {}

    template<typename U>
    AlignedAllocator(AlignedAllocator<U, Alignment> const&) noexcept
    {
    }

    pointer allocate(size_type n)
    {
        pointer p;
        if (posix_memalign(reinterpret_cast<void**>(&p), Alignment, n * sizeof(T))) throw std::bad_alloc();
        return p;
    }

    void deallocate(pointer p, size_type /*n*/) noexcept { std::free(p); }

    template<typename C, class... Args>
    void construct(C* c, Args&&... args)
    {
        new ((void*)c) C(std::forward<Args>(args)...);
    }

    template<typename C>
    void destroy(C* c)
    {
        c->~C();
    }

    bool operator==(AlignedAllocator const&) const noexcept { return true; }

    bool operator!=(AlignedAllocator const&) const noexcept { return false; }

    template<typename U, unsigned int UAlignment>
    bool operator==(AlignedAllocator<U, UAlignment> const&) const noexcept
    {
        return false;
    }

    template<typename U, unsigned int UAlignment>
    bool operator!=(AlignedAllocator<U, UAlignment> const&) const noexcept
    {
        return true;
    }
};

} // namespace util
