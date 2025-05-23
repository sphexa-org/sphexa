/*! @file
 * @brief Timestep definition
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <array>
#include <string>

#include "cstone/tree/definitions.h"

namespace sph
{

struct Timestep
{
    static constexpr int maxNumRungs = 4;

    float nextDt, elapsedDt{0}, totDt;
    int   numRungs{1};
    int   substep{0};

    std::array<cstone::LocalIndex, maxNumRungs + 1> rungRanges;
    util::array<float, maxNumRungs>                 dt_m1, dt_drift;

    template<class Archive>
    void loadOrStore(Archive* ar, const std::string& prefix)
    {
        ar->stepAttribute(prefix + "numRungs", &numRungs, 1);
        ar->stepAttribute(prefix + "dt_m1", dt_m1.data(), dt_m1.size());
    }
};

} // namespace sph
