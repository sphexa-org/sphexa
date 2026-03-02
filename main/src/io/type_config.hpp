//
// Created by Noah Kubli on 02.03.2026.
//

#pragma once

#include "cstone/util/type_list.hpp"

namespace sphexa
{

struct IO
{
    template<class T>
    using ConstPtr = const T*;

    using Types = util::TypeList<double, float, char, unsigned char, signed char, short, int, long, long long,
                                 unsigned short, unsigned, unsigned long, unsigned long long>;
};

} // namespace sphexa
