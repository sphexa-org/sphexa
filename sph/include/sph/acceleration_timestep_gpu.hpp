//
// Created by Noah Kubli on 19.03.2025.
//

#pragma once

namespace sph
{
template<typename T>
T accelerationTimestepGPU(size_t first, size_t last, const T* ax, const T* ay, const T* az, const T* h);
}
