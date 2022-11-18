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
 * @brief A common interface for different kinds of propagators
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 * @author Jose A. Escartin <ja.escartin@gmail.com>
 */

#pragma once

#include <variant>

#include "util/timer.hpp"

namespace sphexa
{

template<class DomainType, class ParticleDataType>
class Propagator
{
    using T = typename ParticleDataType::RealType;

public:
    Propagator(size_t ngmax, size_t ng0, std::ostream& output, size_t rank)
        : timer(output, rank)
        , out(output)
        , rank_(rank)
        , ngmax_(ngmax)
        , ng0_(ng0)
    {
    }

    //! @brief get a list of field strings marked as conserved at runtime
    virtual std::vector<std::string> conservedFields() const = 0;

    //! @brief Marks conserved and dependent fields inside the particle dataset as active, enabling memory allocation
    virtual void activateFields(ParticleDataType& d) = 0;

    virtual void sync(DomainType& domain, ParticleDataType& d) = 0;

    virtual void step(DomainType& domain, ParticleDataType& d) = 0;

    virtual void prepareOutput(ParticleDataType& d, size_t startIndex, size_t endIndex, const cstone::Box<T>&){};
    virtual void finishOutput(ParticleDataType& d){};

    //! @brief this allows the possibility of saving propagator data to file if it is stateful
    virtual void dump(size_t, const std::string&){};

    //! @brief restore state from file if supported and it exists
    virtual void restoreState(const std::string&, MPI_Comm){};

    virtual ~Propagator() = default;

    void printIterationTimings(const DomainType& domain, const ParticleDataType& simData)
    {
        const auto& d = simData.hydro;
        const auto& n = simData.nuclearData;
        if (rank_ == 0)
        {
            double nuclearEnergy       = 0.0;
            size_t n_nuclear_particles = n_nuclear_particles = n.Y[0].size();

            printCheck(d.ttot, d.minDt, d.etot, d.eint, d.ecin, d.egrav, n.enuclear, domain.box(), n_nuclear_particles,
                       d.numParticlesGlobal, domain.nParticles(), domain.globalTree().numLeafNodes(),
                       domain.nParticlesWithHalos() - domain.nParticles(), d.totalNeighbors);

            std::cout << "### Check ### Focus Tree Nodes: " << domain.focusTree().octree().numLeafNodes() << std::endl;
            printTotalIterationTime(d.iteration, timer.duration());
        }
    }

protected:
    MasterProcessTimer timer;
    std::ostream&      out;

    size_t rank_;
    //! maximum number of neighbors per particle
    size_t ngmax_;
    //! target number of neighbors per particle
    size_t ng0_;

    void printTotalIterationTime(size_t iteration, float duration)
    {
        out << "=== Total time for iteration(" << iteration << ") " << duration << "s" << std::endl << std::endl;
    }

    template<class Box>
    void printCheck(double totalTime, double minTimeStep, double totalEnergy, double internalEnergy,
                    double kineticEnergy, double gravitationalEnergy, double nuclearEnergy, const Box& box,
                    size_t numNuclearParticles, size_t totalParticleCount, size_t particleCount, size_t nodeCount,
                    size_t haloCount, size_t totalNeighbors)
    {
        out << "### Check ### Global Tree Nodes: " << nodeCount << ", Particles: " << particleCount
            << ", Halos: " << haloCount;
        out << ", Nuclear particles: " << numNuclearParticles << std::endl;
        out << "### Check ### Computational domain: " << box.xmin() << " " << box.xmax() << " " << box.ymin() << " "
            << box.ymax() << " " << box.zmin() << " " << box.zmax() << std::endl;
        out << "### Check ### Total Neighbors: " << totalNeighbors
            << ", Avg neighbor count per particle: " << totalNeighbors / totalParticleCount << std::endl;
        out << "### Check ### Total time: " << totalTime << ", current time-step: " << minTimeStep << std::endl;
        out << "### Check ### Total energy: " << totalEnergy << ", (internal: " << internalEnergy
            << ", kinetic: " << kineticEnergy;
        out << ", gravitational: " << gravitationalEnergy;
        out << ", nuclear: " << nuclearEnergy;
        out << ")" << std::endl;
    }
};

} // namespace sphexa
