/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief FLOP counting for hydrodynamics propagators
 *
 * @author Auto-generated
 */

#pragma once

#include <cstdint>
#include <iostream>
#include <map>
#include <string>

namespace sphexa
{

/*! @brief FLOP counter for hydrodynamics propagator operations
 *
 * Counts floating point operations in hydrodynamics propagators:
 * - Base hydrodynamics (SPH): neighbor interactions, IAD, momentum/energy
 */
class HydroFlopCounter
{
public:
    struct FlopStats
    {
        uint64_t add{0};
        uint64_t multiply{0};
        uint64_t divide{0};
        uint64_t sqrt{0};
        uint64_t exp{0};
        uint64_t cos{0};
        uint64_t sin{0};
        uint64_t pow{0};
        uint64_t cbrt{0};
        uint64_t total{0};
        uint64_t subtract{0};

        void reset()
        {
            add = multiply = divide = sqrt = exp = cos = sin = pow = cbrt = total = subtract = 0;
        }

        void updateTotal() { total = add + multiply + divide + sqrt + exp + cos + sin + pow + cbrt + subtract; }
    };

    HydroFlopCounter() = default;

    void reset() { stats_.reset(); }

    //! @brief Get total FLOPs
    uint64_t getTotalFlops() const { return stats_.total; }

    //! @brief Get detailed statistics
    const FlopStats& getStats() const { return stats_; }

    //! @brief Print FLOP statistics
    void printStats(std::ostream& os, const std::string& prefix = "") const
    {
        os << prefix << "FLOP Statistics for hydrodynamics propagator:\n";
        os << prefix << "  Additions:        " << stats_.add << "\n";
        os << prefix << "  Subtractions:     " << stats_.subtract << "\n";
        os << prefix << "  Multiplications:  " << stats_.multiply << "\n";
        os << prefix << "  Divisions:        " << stats_.divide << "\n";
        os << prefix << "  Square roots:     " << stats_.sqrt << "\n";
        os << prefix << "  Exponentials:     " << stats_.exp << "\n";
        os << prefix << "  Cosines:          " << stats_.cos << "\n";
        os << prefix << "  Sines:            " << stats_.sin << "\n";
        os << prefix << "  Powers:           " << stats_.pow << "\n";
        os << prefix << "  Cube roots:       " << stats_.cbrt << "\n";
        os << prefix << "  Total FLOPs:      " << stats_.total << "\n";
    }

    //! @brief Calculate FLOPs per second given elapsed time
    double getFlopsPerSecond(double elapsedTimeSeconds) const
    {
        if (elapsedTimeSeconds <= 0.0) { return 0.0; }
        return static_cast<double>(stats_.total) / elapsedTimeSeconds;
    }

    //! @brief Print FLOPs per second
    void printFlopsPerSecond(std::ostream& os, double elapsedTimeSeconds, const std::string& prefix = "") const
    {
        double flopsPerSec = getFlopsPerSecond(elapsedTimeSeconds);
        os << prefix << "FLOPs per second: " << flopsPerSec << " (" << (flopsPerSec / 1e9) << " GFLOPs/s, "
           << (flopsPerSec / 1e12) << " TFLOPs/s)\n";
    }

    //! @brief Count FLOPs for base hydrodynamics (SPH operations)
    //! Based on typical SPH operations: neighbor interactions, IAD, momentum/energy
    //! @param numParticles Number of particles
    //! @param avgNeighbors Average number of neighbors per particle
    //! @param numParticlesWithHalos Total particles including halos
    void countBaseHydro(size_t numParticles, double avgNeighbors, size_t numParticlesWithHalos = 0)
    {
        if (numParticlesWithHalos == 0) { numParticlesWithHalos = numParticles; }
        
        // Per neighbor interaction operations (approximate):
        // - Distance calculations: 3 subtracts, 3 multiplies, 1 sqrt per neighbor
        // - Kernel lookups and evaluations
        // - IAD tensor calculations: ~15 multiplies, 6 adds per neighbor
        // - Momentum/energy: ~10 multiplies, 5 adds per neighbor
        
        // XMass computation: per particle per neighbor
        // Based on xmassJLoop kernel analysis:
        // - Per particle: 2 divides, 4 multiplies (initialization + final)
        // - Per neighbor: 3 subtracts, 6 multiplies, 3 adds, 1 sqrt (typical case)
        // - See sph/include/sph/hydro_ve/XMASS_FLOPS_ANALYSIS.md for details
        stats_.divide += numParticles * 2;                    // hInv and veDefinition
        stats_.multiply += numParticles * 4;                  // h3Inv (2) + final (2)
        stats_.subtract += numParticles * avgNeighbors * 3;   // distance components
        stats_.multiply += numParticles * avgNeighbors * 6;   // distance squares, vloc, accumulation
        stats_.add += numParticles * avgNeighbors * 3;        // distance sum, accumulation (typical)
        stats_.sqrt += numParticles * avgNeighbors;           // distance calculation
        
        // VE Def Gradh: per particle per neighbor
        // Based on veDefGradhJLoop kernel analysis:
        // - Per particle: 4 divides, 16 multiplies, 2 subtracts, 1 add (initialization + final)
        // - Per neighbor: 3 subtracts, 10 multiplies, 6 adds, 1 sqrt (typical case)
        // - See sph/include/sph/hydro_ve/VE_DEF_GRADH_FLOPS_ANALYSIS.md for details
        stats_.divide += numParticles * 4;                    // hInv, whomegai/xmassi, rhoi, dhdrho
        stats_.multiply += numParticles * 16;                 // h3Inv (2) + init (2) + scaling (5) + final (7)
        stats_.subtract += numParticles * 2;                  // final gradient computation
        stats_.add += numParticles * 1;                       // final gradient computation
        stats_.subtract += numParticles * avgNeighbors * 3;   // distance components
        stats_.multiply += numParticles * avgNeighbors * 10;  // distance, vloc, dterh, accumulations
        stats_.add += numParticles * avgNeighbors * 6;        // distance sum, dterh, accumulations (typical)
        stats_.sqrt += numParticles * avgNeighbors;           // distance calculation
        
        // EOS: per particle
        // - ~3 multiplies, 1 divide, 1 sqrt per particle
        stats_.multiply += numParticles * 3;
        stats_.divide += numParticles;
        stats_.sqrt += numParticles;
        
        // IAD computation: per particle per neighbor
        // Based on IADJLoop kernel analysis:
        // - Per particle: 56 FLOPs (initialization + final computation)
        // - Per neighbor: 31 FLOPs (typical case)
        // - See sph/include/sph/hydro_ve/IAD_DIVV_CURLV_FLOPS_ANALYSIS.md for details
        stats_.divide += numParticles * 3;                    // hiInv, normalization, factor
        stats_.multiply += numParticles * 39;                 // final IAD tensor computation
        stats_.add += numParticles * 5;                       // normalization sum
        stats_.subtract += numParticles * 9;                  // determinant and tensor components
        stats_.subtract += numParticles * avgNeighbors * 3;   // distance components
        stats_.multiply += numParticles * avgNeighbors * 18;  // distance, vloc, volj_w, tensor accumulation
        stats_.divide += numParticles * avgNeighbors;         // volj_w calculation
        stats_.add += numParticles * avgNeighbors * 8;        // distance sum, tensor accumulation (typical)
        stats_.sqrt += numParticles * avgNeighbors;           // distance calculation
        
        // DivV/CurlV computation: per particle per neighbor
        // Based on divV_curlVJLoop kernel analysis:
        // - Per particle: 18 FLOPs (with curl, typical case)
        // - Per neighbor: 47 FLOPs (typical case)
        stats_.divide += numParticles * 1;                    // hiInv
        stats_.multiply += numParticles * 2;                  // hiInv3
        stats_.divide += numParticles * 1;                    // norm_kxi
        stats_.multiply += numParticles * 6;                  // final computations (div + curl)
        stats_.add += numParticles * 4;                       // divergence and curl
        stats_.subtract += numParticles * 3;                  // curl components
        stats_.sqrt += numParticles * 1;                      // curl magnitude
        stats_.subtract += numParticles * avgNeighbors * 6;   // distance and velocity differences
        stats_.multiply += numParticles * avgNeighbors * 23;  // distance, vloc, termA, accumulations
        stats_.add += numParticles * avgNeighbors * 17;       // distance sum, termA, accumulations (typical)
        stats_.sqrt += numParticles * avgNeighbors;           // distance calculation
        
        // AV switches: per particle per neighbor
        // Based on AVswitchesJLoop kernel analysis:
        // - Per particle: 24 FLOPs (initialization + final computation)
        // - Per neighbor: 47 FLOPs (typical case)
        // - See sph/include/sph/hydro_ve/AV_SWITCHES_FLOPS_ANALYSIS.md for details
        stats_.divide += numParticles * 4;                    // hiInv, decay, alphadot, alphaloc
        stats_.multiply += numParticles * 13;                 // vijsignal_i, hiInv3, graddivv, a_const, alphaloc, decay, alphadot
        stats_.add += numParticles * 5;                       // graddivv, a_const, alpha_i update
        stats_.subtract += numParticles * 1;                  // alphadot calculation
        stats_.sqrt += numParticles * 1;                      // graddivv magnitude
        stats_.subtract += numParticles * avgNeighbors * 7;   // distance, velocity, factor
        stats_.multiply += numParticles * avgNeighbors * 23;  // distance, rv, signal, v1, Wi, termA, factor, graddivv
        stats_.divide += numParticles * avgNeighbors * 2;     // volj, signal speed
        stats_.add += numParticles * avgNeighbors * 14;       // distance, rv, signal, termA, graddivv (typical)
        stats_.sqrt += numParticles * avgNeighbors;           // distance calculation
        
        // Momentum and Energy: per particle per neighbor
        // Based on momentumAndEnergyJLoop kernel analysis:
        // - Per particle: 15 FLOPs (initialization + final computation)
        // - Per neighbor: ~121 FLOPs (without avClean, typical case), ~156 FLOPs (with avClean)
        // - See sph/include/sph/hydro_ve/MOMENTUM_ENERGY_FLOPS_ANALYSIS.md for details
        // Note: This assumes avClean is false. If avClean is true, add ~35 FLOPs per neighbor
        stats_.divide += numParticles * 4;                    // rhoi, hiInv, eta_crit (2)
        stats_.multiply += numParticles * 7;                  // rhoi, hiInv3, eta_crit, final (5)
        stats_.add += numParticles * 1;                       // final
        stats_.cbrt += numParticles * 1;                      // eta_crit calculation
        stats_.subtract += numParticles * avgNeighbors * 8;   // distance, velocity, Atwood, signal
        stats_.multiply += numParticles * avgNeighbors * 70;  // All multiplications (excluding pow)
        stats_.divide += numParticles * avgNeighbors * 6;     // hjInv, rhoj, wij, a_visc, b_visc
        stats_.add += numParticles * avgNeighbors * 33;       // All additions (typical)
        stats_.sqrt += numParticles * avgNeighbors;           // distance
        stats_.pow += numParticles * avgNeighbors;            // Atwood else branch (average ~1.33, rounded to 1)
        
        stats_.updateTotal();
    }

private:
    FlopStats stats_;
};

} // namespace sphexa

