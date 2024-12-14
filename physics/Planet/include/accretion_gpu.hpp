//
// Created by Noah Kubli on 12.03.2024.
//

#pragma once

#include "star_data.hpp"

namespace disk
{
template<typename Dataset>
void computeAccretionConditionGPU(size_t first, size_t last, Dataset& d, StarData& star);
}
