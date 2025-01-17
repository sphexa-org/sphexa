/*
 * Ryoanji N-body solver
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Kahan summation
 *
 * @author Rio Yokota <rioyokota@gsic.titech.ac.jp>
 */

#pragma once

#include <iostream>

#include "cstone/cuda/annotation.hpp"

//! Operator overloading for Kahan summation
template<typename T>
struct kahan
{
    T                               s;
    T                               c;
    HOST_DEVICE_FUN __forceinline__ kahan() {} // Default constructor
    HOST_DEVICE_FUN __forceinline__ kahan(const T& v)
    { // Copy constructor (scalar)
        s = v;
        c = 0;
    }
    HOST_DEVICE_FUN kahan(const kahan& v)
    { // Copy constructor (structure)
        s = v.s;
        c = v.c;
    }
    HOST_DEVICE_FUN ~kahan() {} // Destructor
    HOST_DEVICE_FUN const kahan& operator=(const T v)
    { // Scalar assignment
        s = v;
        c = 0;
        return *this;
    }
    HOST_DEVICE_FUN const kahan& operator+=(const T v)
    { // Scalar compound assignment (add)
        T y = v - c;
        T t = s + y;
        c   = (t - s) - y;
        s   = t;
        return *this;
    }
    HOST_DEVICE_FUN const kahan& operator-=(const T v)
    { // Scalar compound assignment (subtract)
        T y = -v - c;
        T t = s + y;
        c   = (t - s) - y;
        s   = t;
        return *this;
    }
    HOST_DEVICE_FUN const kahan& operator*=(const T v)
    { // Scalar compound assignment (multiply)
        c *= v;
        s *= v;
        return *this;
    }
    HOST_DEVICE_FUN const kahan& operator/=(const T v)
    { // Scalar compound assignment (divide)
        c /= v;
        s /= v;
        return *this;
    }
    HOST_DEVICE_FUN const kahan& operator=(const kahan& v)
    { // Vector assignment
        s = v.s;
        c = v.c;
        return *this;
    }
    HOST_DEVICE_FUN const kahan& operator+=(const kahan& v)
    { // Vector compound assignment (add)
        T y = v.s - c;
        T t = s + y;
        c   = (t - s) - y;
        s   = t;
        y   = v.c - c;
        t   = s + y;
        c   = (t - s) - y;
        s   = t;
        return *this;
    }
    HOST_DEVICE_FUN const kahan& operator-=(const kahan& v)
    { // Vector compound assignment (subtract)
        T y = -v.s - c;
        T t = s + y;
        c   = (t - s) - y;
        s   = t;
        y   = -v.c - c;
        t   = s + y;
        c   = (t - s) - y;
        s   = t;
        return *this;
    }
    HOST_DEVICE_FUN const kahan& operator*=(const kahan& v)
    { // Vector compound assignment (multiply)
        c *= (v.c + v.s);
        s *= (v.c + v.s);
        return *this;
    }
    HOST_DEVICE_FUN const kahan& operator/=(const kahan& v)
    { // Vector compound assignment (divide)
        c /= (v.c + v.s);
        s /= (v.c + v.s);
        return *this;
    }
    HOST_DEVICE_FUN kahan operator-() const
    { // Vector arithmetic (negation)
        kahan temp;
        temp.s = -s;
        temp.c = -c;
        return temp;
    }
    HOST_DEVICE_FUN      operator T() { return s + c; }             // Type-casting (lvalue)
    HOST_DEVICE_FUN      operator const T() const { return s + c; } // Type-casting (rvalue)
    friend std::ostream& operator<<(std::ostream& s, const kahan& v)
    { // Output stream
        s << (v.s + v.c);
        return s;
    }
    friend std::istream& operator>>(std::istream& s, kahan& v)
    { // Input stream
        s >> v.s;
        v.c = 0;
        return s;
    }
};
