//
// Created by Noah Kubli on 12.02.2025.
//

#pragma once

#include "cstone/cuda/device_vector.h"
#include "cstone/fields/field_get.hpp"
#include "cstone/util/constexpr_string.hpp"

namespace disk
{
template<util::StructuralString field, typename Dataset>
auto* getPtr(Dataset& d)
{
    return rawPtr(get<field>(d));
}
} // namespace disk
