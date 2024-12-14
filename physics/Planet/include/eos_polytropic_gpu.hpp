#pragma once

namespace disk
{
template<typename Dataset, typename StarData>
extern void computePolytropicEOS_HydroStdGPU(size_t firstParticle, size_t lastParticle, Dataset& d,
                                             const StarData& star);
} // namespace disk
