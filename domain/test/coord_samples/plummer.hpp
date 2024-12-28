/*! @file plummer distribution

    Adapted from: https://github.com/treecode/Bonsai/blob/master/runtime/include/plummer.h
*/

#pragma once

#include <array>
#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>

//! @brief returns a vector of @p n Plummer distributed 3D coordinates
template<class T>
std::array<std::vector<T>, 3> plummer(size_t n)
{
    srand48(42);

    std::array<std::vector<T>, 3> pos;
    pos[0].resize(n);
    pos[1].resize(n);
    pos[2].resize(n);

    T conv = 3.0 * M_PI / 16.0;

    size_t i = 0;
    while (i < n)
    {
        T R = 1.0 / sqrt((pow(drand48(), -2.0 / 3.0) - 1.0));

        if (R < 100.0)
        {
            T Z     = (1.0 - 2.0 * drand48()) * R;
            T theta = 2 * M_PI * drand48();
            T X     = sqrt(R * R - Z * Z) * cos(theta);
            T Y     = sqrt(R * R - Z * Z) * sin(theta);

            X *= conv;
            Y *= conv;
            Z *= conv;

            pos[0][i] = X;
            pos[1][i] = Y;
            pos[2][i] = Z;

            i++;
        }
    }

    T mcm    = 0.0;
    T mass   = T(1) / n;
    T xcm[3] = {0, 0, 0};

    for (i = 0; i < n; i++)
    {
        mcm += mass;
        for (int k = 0; k < 3; k++)
        {
            xcm[k] += mass * pos[k][i];
        }
    }

    for (int k = 0; k < 3; k++)
    {
        xcm[k] /= mcm;
    }

    for (i = 0; i < n; i++)
    {
        for (int k = 0; k < 3; k++)
        {
            pos[k][i] -= xcm[k];
        }
    }

    return pos;
}
