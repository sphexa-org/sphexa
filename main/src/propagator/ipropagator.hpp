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

#include "cstone/sfc/box.hpp"
#include "cstone/primitives/accel_switch.hpp"
#include "io/ifile_io.hpp"
#include "sph/particles_data.hpp"
#include "util/pm_reader.hpp"
#include "util/timer.hpp"

namespace sphexa
{

template<class DomainType, class ParticleDataType>
class Propagator
{
    using T = typename ParticleDataType::RealType;

public:
    Propagator(std::ostream& output, int rank)
        : out(output)
        , timer(output)
        , pmReader(rank)
        , rank_(rank)
    {
    }

    //! @brief get a list of field strings marked as conserved at runtime
    virtual std::vector<std::string> conservedFields() const = 0;

    //! @brief Marks conserved and dependent fields inside the particle dataset as active, enabling memory allocation
    virtual void activateFields(ParticleDataType& d) = 0;

    //! @brief synchronize computational domain
    virtual void sync(DomainType& domain, ParticleDataType& d) = 0;

    //! @brief synchronize domain and compute forces
    virtual void computeForces(DomainType& domain, ParticleDataType& d) = 0;

    //! @brief integrate and/or drift particles in time
    virtual void integrate(DomainType& domain, ParticleDataType& d) = 0;

    //! @brief save particle data fields to file
    virtual void saveFields(IFileWriter*, size_t, size_t, ParticleDataType&, const cstone::Box<T>&){};

    //! @brief save extra customizable stuff
    virtual void saveExtra(IFileWriter*, ParticleDataType&){};

    //! @brief save internal state to file
    virtual void save(IFileWriter*){};

    //! @brief load internal state from file
    virtual void load(const std::string& path, IFileReader*){};

    //! @brief whether conserved quantities are time-synchronized (when completing a full time-step hierarchy)
    virtual bool isSynced() { return true; }

    //! @brief add pm counters if they exist
    void addCounters(const std::string& pmRoot, int numRanksPerNode) { pmReader.addCounters(pmRoot, numRanksPerNode); }

    //! @brief print timing info
    void writeMetrics(IFileWriter* writer, const std::string& outFile)
    {
        timer.writeTimings(writer, outFile);
        pmReader.writeTimings(writer, outFile);
    };

    virtual ~Propagator() = default;

    void printIterationTimings(const DomainType& domain, const ParticleDataType& simData)
    {
        const auto& d   = simData.hydro;
        const auto& box = domain.box();

        auto nodeCount          = domain.globalTree().numLeafNodes();
        auto particleCount      = domain.nParticles();
        auto haloCount          = domain.nParticlesWithHalos() - domain.nParticles();
        auto totalNeighbors     = d.totalNeighbors;
        auto totalParticleCount = d.numParticlesGlobal;

        out << "### Check ### Global Tree Nodes: " << nodeCount << ", Particles: " << particleCount
            << ", Halos: " << haloCount << std::endl;
        out << "### Check ### Computational domain: " << box.xmin() << " " << box.xmax() << " " << box.ymin() << " "
            << box.ymax() << " " << box.zmin() << " " << box.zmax() << std::endl;
        out << "### Check ### Total Neighbors: " << totalNeighbors
            << ", Avg neighbor count per particle: " << totalNeighbors / totalParticleCount << std::endl;
        out << "### Check ### Total time: " << d.ttot - d.minDt << ", current time-step: " << d.minDt << std::endl;
        out << "### Check ### Total energy: " << d.etot << ", (internal: " << d.eint << ", kinetic: " << d.ecin;
        out << ", gravitational: " << d.egrav;
        out << ")" << std::endl;
        out << "### Check ### Focus Tree Nodes: " << domain.focusTree().octreeViewAcc().numLeafNodes << ", maxDepth "
            << domain.focusTree().depth();
        if constexpr (cstone::HaveGpu<typename ParticleDataType::AcceleratorType>{})
        {
            out << ", maxStackNc " << d.devData.stackUsedNc << ", maxStackGravity " << d.devData.stackUsedGravity;
        }
        out << "\n=== Total time for iteration(" << d.iteration << ") " << timer.sumOfSteps() << "s\n\n";
    }

protected:
    static void outputAllocatedFields(IFileWriter* writer, size_t first, size_t last, ParticleDataType& simData)
    {
        auto output = [](size_t first, size_t last, auto& d, IFileWriter* writer)
        {
            auto fieldPointers = d.data();
            auto indicesDone   = d.outputFieldIndices;
            auto namesDone     = d.outputFieldNames;

            for (int i = int(indicesDone.size()) - 1; i >= 0; --i)
            {
                int fidx = indicesDone[i];
                if (d.isAllocated(fidx))
                {
                    int column = std::find(d.outputFieldIndices.begin(), d.outputFieldIndices.end(), fidx) -
                                 d.outputFieldIndices.begin();
                    transferToHost(d, first, last, {d.fieldNames[fidx]});
                    std::visit([writer, c = column, key = namesDone[i]](auto field)
                               { writer->writeField(key, field->data(), c); },
                               fieldPointers[fidx]);
                    indicesDone.erase(indicesDone.begin() + i);
                    namesDone.erase(namesDone.begin() + i);
                }
            }

            if (!indicesDone.empty() && writer->rank() == 0)
            {
                std::cout << "WARNING: the following fields are not in use and therefore not output: ";
                for (int fidx = 0; fidx < indicesDone.size() - 1; ++fidx)
                {
                    std::cout << d.fieldNames[fidx] << ",";
                }
                std::cout << d.fieldNames[indicesDone.back()] << std::endl;
            }
        };

        output(first, last, simData.hydro, writer);
        output(first, last, simData.chem, writer);
    }

    std::ostream& out;
    Timer         timer;
    PmReader      pmReader;
    int           rank_;
};

} // namespace sphexa
