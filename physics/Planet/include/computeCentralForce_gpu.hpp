//
// Created by Noah Kubli on 11.03.2024.
//

#pragma once

namespace disk
{
template<typename Dataset, typename StarData>
void computeCentralForceGPU(size_t first, size_t last, Dataset& d, StarData& star);
}
