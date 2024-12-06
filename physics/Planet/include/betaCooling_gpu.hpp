//
// Created by Noah Kubli on 17.04.2024.
//

#pragma once

namespace disk
{
template<typename Dataset, typename StarData>
void betaCoolingGPU(size_t first, size_t last, Dataset& d, StarData& star);

template<typename Dataset, typename StarData>
double duTimestepGPU(size_t first, size_t last, const Dataset& d, const StarData& star);

} // namespace planet
