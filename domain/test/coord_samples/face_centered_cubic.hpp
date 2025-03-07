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
 * @brief Random coordinates generation for testing
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <algorithm>
#include <random>
#include <vector>

#include "cstone/primitives/gather.hpp"
#include "cstone/sfc/sfc.hpp"
#include "cstone/tree/definitions.h"

namespace cstone
{

template<class T, class KeyType_>
class RandomCoordinates
{
public:
    using KeyType = KeyType_;
    using Integer = typename KeyType::ValueType;

    RandomCoordinates(size_t n, Box<T> box, int seed = 42)
        : box_(std::move(box))
        , x_(n)
        , y_(n)
        , z_(n)
        , codes_(n)
    {
        // std::random_device rd;
        std::mt19937 gen(seed);
        std::uniform_real_distribution<T> disX(box_.xmin(), box_.xmax());
        std::uniform_real_distribution<T> disY(box_.ymin(), box_.ymax());
        std::uniform_real_distribution<T> disZ(box_.zmin(), box_.zmax());

        auto randX = [&disX, &gen]() { return disX(gen); };
        auto randY = [&disY, &gen]() { return disY(gen); };
        auto randZ = [&disZ, &gen]() { return disZ(gen); };

        std::generate(begin(x_), end(x_), randX);
        std::generate(begin(y_), end(y_), randY);
        std::generate(begin(z_), end(z_), randZ);

        auto keyData = (KeyType*)(codes_.data());
        computeSfcKeys(x_.data(), y_.data(), z_.data(), keyData, n, box);

        std::vector<LocalIndex> sfcOrder(n);
        std::iota(begin(sfcOrder), end(sfcOrder), LocalIndex(0));
        sort_by_key(begin(codes_), end(codes_), begin(sfcOrder));

        std::vector<T> temp(x_.size());
        gather<LocalIndex>(sfcOrder, x_.data(), temp.data());
        swap(x_, temp);
        gather<LocalIndex>(sfcOrder, y_.data(), temp.data());
        swap(y_, temp);
        gather<LocalIndex>(sfcOrder, z_.data(), temp.data());
        swap(z_, temp);
    }

    const std::vector<T>& x() const { return x_; }
    const std::vector<T>& y() const { return y_; }
    const std::vector<T>& z() const { return z_; }
    const std::vector<Integer>& particleKeys() const { return codes_; }

private:
    Box<T> box_;
    std::vector<T> x_, y_, z_;
    std::vector<Integer> codes_;
};

template<class T, class KeyType_>
class FaceCenteredCubicCoordinates
{
public:
    using KeyType = KeyType_;
    using Integer = typename KeyType::ValueType;

    FaceCenteredCubicCoordinates(unsigned nx, unsigned ny, unsigned nz, Box<T> box, int seed = 42)
        : box_(std::move(box))
        , x_(nx * ny * nz * 4)
        , y_(nx * ny * nz * 4)
        , z_(nx * ny * nz * 4)
        , codes_(nx * ny * nz * 4)
    {
        std::mt19937 gen(seed);

        const T dx       = box.lx() / nx;
        const T dy       = box.ly() / ny;
        const T dz       = box.lz() / nz;
        const unsigned n = nx * ny * nz * 4;
#pragma omp parallel for collapse(3)
        for (unsigned i = 0; i < nx; ++i)
        {
            for (unsigned j = 0; j < ny; ++j)
            {
                for (unsigned k = 0; k < nz; ++k)
                {
                    const unsigned idx = i * ny * nz * 4 + j * nz * 4 + k * 4;
                    x_[idx + 0]        = i * dx;
                    y_[idx + 0]        = j * dy;
                    z_[idx + 0]        = k * dz;
                    x_[idx + 1]        = i * dx;
                    y_[idx + 1]        = (j + 0.5) * dy;
                    z_[idx + 1]        = (k + 0.5) * dz;
                    x_[idx + 2]        = (i + 0.5) * dx;
                    y_[idx + 2]        = j * dy;
                    z_[idx + 2]        = (k + 0.5) * dz;
                    x_[idx + 3]        = (i + 0.5) * dx;
                    y_[idx + 3]        = (j + 0.5) * dy;
                    z_[idx + 3]        = k * dz;
                }
            }
        }

        auto keyData = (KeyType*)(codes_.data());
        computeSfcKeys(x_.data(), y_.data(), z_.data(), keyData, n, box);

        std::vector<LocalIndex> sfcOrder(n);
        std::iota(begin(sfcOrder), end(sfcOrder), LocalIndex(0));
        sort_by_key(begin(codes_), end(codes_), begin(sfcOrder));

        std::vector<T> temp(x_.size());
        gather<LocalIndex>(sfcOrder, x_.data(), temp.data());
        swap(x_, temp);
        gather<LocalIndex>(sfcOrder, y_.data(), temp.data());
        swap(y_, temp);
        gather<LocalIndex>(sfcOrder, z_.data(), temp.data());
        swap(z_, temp);
    }

    const Box<T>& box() const { return box_; }
    const std::vector<T>& x() const { return x_; }
    const std::vector<T>& y() const { return y_; }
    const std::vector<T>& z() const { return z_; }
    const std::vector<Integer>& particleKeys() const { return codes_; }

private:
    Box<T> box_;
    std::vector<T> x_, y_, z_;
    std::vector<Integer> codes_;
};

} // namespace cstone
