//
// Created by Noah Kubli on 06.12.2024.
//

#pragma once
#include "cstone/tree/definitions.h"

namespace disk
{
struct RemovalStatistics
{
    double               mass;
    cstone::Vec3<double> momentum;
    unsigned             count;

    HOST_DEVICE_FUN friend RemovalStatistics operator+(const RemovalStatistics& a, const RemovalStatistics& b)
    {
        RemovalStatistics result;
        result.mass     = a.mass + b.mass;
        result.momentum = a.momentum + b.momentum;
        result.count    = a.count + b.count;
        return result;
    }
};

} // namespace disk
