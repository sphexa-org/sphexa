//
// Created by Noah Kubli on 11.03.2024.
//

#pragma once
#include "star_data.hpp"

namespace disk
{
template<typename Treal, typename Thydro, typename Tmass>
void computeCentralForceGPU(size_t first, size_t last, const Treal* x, const Treal* y, const Treal* z, Thydro* ax,
                            Thydro* ay, Thydro* az, const Tmass* m, const Treal g, StarData& star);
}
