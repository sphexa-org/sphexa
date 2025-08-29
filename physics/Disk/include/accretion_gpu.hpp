//
// Created by Noah Kubli on 12.03.2024.
//

#pragma once

#include "star_data.hpp"

namespace disk
{
template<typename Treal, typename Thydro, typename Tkeys, typename Tmass, typename Tn>
void computeAccretionConditionGPU(size_t first, size_t last, const Treal* x, const Treal* y, const Treal* z,
                                  const Thydro* h, Tkeys* keys, const Tmass* m, const Thydro* vx, const Thydro* vy,
                                  const Thydro* vz, const Tn* nc, StarData& star, unsigned ng0, unsigned ngmax);
}
