/*! @file
 * @brief Turbulence simulation data initialization
 *
 * @author Axel Sanz <axelsanzlechuga@gmail.com>
 */

#pragma once

#include <iostream>

#include "sph/hydro_turb/turbulence_data.hpp"

namespace sph
{

/*! @brief TODO: Short description of this function
 *
 * @tparam       T                 float or double
 * @param[inout] d                 the turbulence dataset
 * @param[in]    Lx
 * @param[in]    Ly
 * @param[in]    Lz
 * @param[in]    st_maxmodes
 * @param[in]    stirMax
 * @param[in]    stirMin
 * @param[in]    ndim
 * @param[in]    spectForm
 * @param[in]    powerLawExp
 * @param[in]    anglesExp
 *
 * Function description missing
 */
template<class Dataset, class T>
void createStirringModes(Dataset& d, T Lx, T Ly, T Lz, size_t st_maxmodes, T stirMax, T stirMin, size_t ndim,
                         size_t spectForm, T powerLawExp, T anglesExp, bool verbose)
{
    // TODO: this should not take the whole Dataset, only modes and amplitudes are needed
    // TODO: st_maxmodes should not be an input parameter. the caller will have determineNumModes
    // TODO: and it's up to the call-site to decide if they want to create as many as it takes (or skip the check)

    const T twopi = 2.0 * M_PI;

    // characteristic k for scaling the amplitude below
    T kc = stirMin;
    if (spectForm == 1) { kc = 0.5 * (stirMin + stirMax); }

    size_t ikxmin = 0;
    size_t ikymin = 0;
    size_t ikzmin = 0;

    size_t ikxmax = 256;
    size_t ikymax = (ndim > 1) ? 256 : 0;
    size_t ikzmax = (ndim > 2) ? 256 : 0;

    // TODO: this should be a separate function "determineNumModes".
    // determine the number of required modes (in case of full sampling)
    d.numModes = 0;
    for (size_t ikx = ikxmin; ikx <= ikxmax; ikx++)
    {
        T kx = twopi * ikx / Lx;
        for (size_t iky = ikymin; iky <= ikymax; iky++)
        {
            T ky = twopi * iky / Ly;
            for (size_t ikz = ikzmin; ikz <= ikzmax; ikz++)
            {
                T kz = twopi * ikz / Lz;
                T k  = std::sqrt(kx * kx + ky * ky + kz * kz);
                if (k >= stirMin && k <= stirMax)
                {
                    d.numModes += 1;
                    if (ndim > 1) { d.numModes += 1; }
                    if (ndim > 2) { d.numModes += 2; }
                }
            }
        }
    }
    T st_tot_nmodes = d.numModes;

    d.numModes = -1;

    // TODO: should be a separate function
    if (spectForm != 2)
    {
        if (verbose) std::cout << "Generating " << st_tot_nmodes << " driving modes..." << std::endl;

        // prefactor for amplitude normalistion to 1 at kc = 0.5*(st_stirmin+st_stirmax)
        T parab_prefact = -4.0 / ((stirMax - stirMin) * (stirMax - stirMin));

        // for band and parabolic spectrum, use the standard full sampling
        // loop over all kx, ky, kz to generate driving modes
        for (size_t ikx = ikxmin; ikx <= ikxmax; ikx++)
        {
            T kx = twopi * ikx / Lx;
            for (size_t iky = ikymin; iky <= ikymax; iky++)
            {
                T ky = twopi * iky / Ly;
                for (size_t ikz = ikzmin; ikz <= ikzmax; ikz++)
                {
                    T kz = twopi * ikz / Lz;
                    T k  = std::sqrt(kx * kx + ky * ky + kz * kz);

                    if ((k >= stirMin) && (k <= stirMax))
                    {
                        if ((d.numModes + 1 + std::pow(2, ndim - 1)) > st_maxmodes)
                        {
                            std::cout << "init_stir:  number of modes: = " << d.numModes + 1
                                      << " maxstirmodes = " << st_maxmodes << std::endl;
                            std::cout << "Too many stirring modes" << std::endl;
                            break;
                        }

                        T amplitude = 1.0; // Band
                        if (spectForm == 1)
                        {
                            amplitude = std::abs(parab_prefact * (k - kc) * (k - kc) + 1.0);
                        } // Parabola

                        // note: power spectrum ~ amplitude^2 (1D), amplitude^2 * 2pi k (2D), amplitude^2 * 4pi k^2 (3D)
                        amplitude = 2.0 * std::sqrt(amplitude) * std::pow((kc / k), 0.5 * (ndim - 1));

                        d.numModes += 1;
                        d.amplitudes[d.numModes] = amplitude;

                        d.modes[ndim * d.numModes]     = kx;
                        d.modes[ndim * d.numModes + 1] = ky;
                        d.modes[ndim * d.numModes + 2] = kz;

                        if (ndim > 1)
                        {
                            d.numModes += 1;
                            d.amplitudes[d.numModes] = amplitude;

                            d.modes[ndim * d.numModes]     = kx;
                            d.modes[ndim * d.numModes + 1] = -ky;
                            d.modes[ndim * d.numModes + 2] = kz;
                        }

                        if (ndim > 2)
                        {
                            d.numModes += 1;
                            d.amplitudes[d.numModes] = amplitude;

                            d.modes[ndim * d.numModes]     = kx;
                            d.modes[ndim * d.numModes + 1] = ky;
                            d.modes[ndim * d.numModes + 2] = -kz;

                            d.numModes += 1;
                            d.amplitudes[d.numModes] = amplitude;

                            d.modes[ndim * d.numModes]     = kx;
                            d.modes[ndim * d.numModes + 1] = -ky;
                            d.modes[ndim * d.numModes + 2] = -kz;
                        }
                    } // in k range
                }     // ikz
            }         // iky
        }             // ikx
    }

    // TODO: should be a separate function
    if (spectForm == 2)
    {
        std::uniform_real_distribution<T> uniDist(0, 1);
        auto                              uniRng = [&d, &uniDist] { return uniDist(d.gen); };

        // loop between smallest and largest k
        int ikmin = std::max(1, int(stirMin * Lx / twopi + 0.5));
        int ikmax = int(stirMax * Lx / twopi + 0.5);

        for (int ik = ikmin; ik <= ikmax; ik++)
        {
            int nang = std::pow(2, ndim) * ceil(std::pow(ik, anglesExp));
            if (verbose) std::cout << "ik = " << ik << " , number of angles = " << nang << std::endl;

            for (int iang = 1; iang <= nang; iang++)
            {
                T phi = twopi * uniRng(); // phi = [0,2pi] sample the whole sphere
                if (ndim == 1)
                {
                    if (phi < twopi / 2) { phi = 0.0; }
                    if (phi >= twopi / 2) { phi = twopi / 2.0; }
                }

                T theta = twopi / 4.0;
                if (ndim > 2) { theta = std::acos(1.0 - 2.0 * uniRng()); } // theta = [0,pi] sample the whole sphere

                T rand = ik + uniRng() - 0.5;
                T kx   = twopi * std::round(rand * std::sin(theta) * std::cos(phi)) / Lx;
                T ky   = (ndim > 1) ? twopi * std::round(rand * std::sin(theta) * std::sin(phi)) / Ly : 0.0;
                T kz   = (ndim > 2) ? twopi * std::round(rand * std::cos(theta)) / Lz : 0.0;

                T k = std::sqrt(kx * kx + ky * ky + kz * kz);

                if ((k >= stirMin) && (k <= stirMax))
                {
                    if ((d.numModes + 1 + std::pow(2, ndim - 1)) > st_maxmodes)
                    {
                        std::cout << "init_stir:  number of modes: = " << d.numModes + 1
                                  << " maxstirmodes = " << st_maxmodes << std::endl;
                        std::cout << "Too many stirring modes" << std::endl;
                        break;
                    }

                    T amplitude = std::pow(k / kc, powerLawExp); // Power law

                    // note: power spectrum ~ amplitude^2 (1D), amplitude^2 * 2pi k (2D), amplitude^2 * 4pi k^2 (3D)
                    // ...and correct for the number of angles sampled relative to the full sampling (k^2 per k-shell in
                    // 3D)
                    amplitude = std::sqrt(amplitude * (std::pow(ik, ndim - 1) * 4.0 * (std::sqrt(3.0)) / nang)) *
                                std::pow(kc / k, (ndim - 1) / 2.0);

                    d.numModes = d.numModes + 1;

                    d.amplitudes[d.numModes] = amplitude;

                    d.modes[ndim * d.numModes]     = kx;
                    d.modes[ndim * d.numModes + 1] = ky;
                    d.modes[ndim * d.numModes + 2] = kz;
                } // in k range
            }     // loop over angles
        }         // loop over k
    }             // st_spectform .eq. 2
    d.numModes += 1;

    if (verbose) std::cout << "Total Number of Stirring Modes: " << d.numModes << std::endl;
}

} // namespace sph
