//
// Created by Noah Kubli on 17.04.2024.
//

#pragma once

#include "star_data.hpp"

namespace disk
{

template<typename Treal, typename Thydro>
void betaCoolingGPU(size_t first, size_t last, const Treal* x, const Treal* y, const Treal* z, const Treal* u,
                    const Thydro* rho, Treal* du, const Treal g, const StarData& star);

template<typename Treal>
double duTimestepGPU(size_t first, size_t last, const Treal* u, const Treal* du);

} // namespace disk
