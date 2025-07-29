//
// Created by Noah Kubli on 30.11.2023.
//

#pragma once

#include <algorithm>

#include "cooler.hpp"
#include "cstone/util/tuple_util.hpp"
#include "cstone/fields/field_get.hpp"

namespace cooling
{

//! @brief Class to launch OMP task by dividing cooling data into arrays of a certain size
template<size_t N = 100>
struct Partition
{
    const size_t first;
    size_t       n_bins;
    size_t       N_last;
    Partition(size_t first, size_t last)
        : first(first)
    {
        const size_t n_particles = last - first;
        const size_t mod         = n_particles % N;
        const size_t ceil_term   = (mod == 0) ? 0 : 1;
        n_bins                   = n_particles / N + ceil_term;
        N_last                   = mod == 0 ? N : mod;
    }
};

struct Task
{
    const size_t first;
    const size_t len;
    const size_t last;

    template<size_t N>
    Task(size_t i, const Partition<N>& part)
        : first(part.first + N * i)
        , len((i == part.n_bins - 1) ? part.N_last : N)
        , last(first + len)
    {
    }

    Task(size_t first, size_t len)
        : first(first)
        , len(len)
        , last(first + len)
    {
    }
};

} // namespace cooling
