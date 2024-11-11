//
// Created by Noah Kubli on 12.03.2024.
//

#pragma once

template<typename Dataset, typename StarData>
void computeAccretionConditionGPU(size_t first, size_t last, Dataset& d, StarData& star);
