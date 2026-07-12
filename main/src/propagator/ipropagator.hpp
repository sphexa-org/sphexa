/*
 * MIT License
 *
 * SPH-EXA
 * Copyright (c) 2024 CSCS, ETH Zurich, University of Basel, University of Zurich
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
#include "io/ifile_io.hpp"
#include "io/id_tag_utils.hpp"
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
    virtual void saveFields(IFileWriter*, size_t, size_t, ParticleDataType&, const cstone::Box<T>&) {}

    //! @brief save particle subset data fields to file
    // TODO: should be const ParticleDataType& but at some point we use the data() method which is non-const
    void saveSubsetFields(IFileWriter* fileWriter, const std::string& outFileSubset, std::size_t firstIndex, std::size_t lastIndex, ParticleDataType& simData)
    {
        // Find the selected particles positions in dataset
        using ParticleIndexVectorType = decltype(ParticleDataType::HydroData::id);
        using Exec = ParticleDataType::Exec;
        std::span<const uint64_t> ids(simData.hydro.id.data(), simData.hydro.id.size());
        ParticleIndexVectorType selectedParticlesIndexes;
        if constexpr (cstone::execution::HaveGpu<Exec>{}) {
            findTaggedIdsGPU(ids, firstIndex, lastIndex, selectedParticlesIndexes);
        }
        else {
            findTaggedIds(ids, firstIndex, lastIndex, selectedParticlesIndexes);
        }
        fileWriter->addStep(0, selectedParticlesIndexes.size(), outFileSubset);
        simData.hydro.loadOrStoreAttributes(fileWriter);

        outputAllocatedFields(fileWriter, simData, selectedParticlesIndexes);

        fileWriter->closeStep();

        timer.step("SelectedParticlesFileOutput");

    }

    //! @brief save internal state to file
    virtual void save(IFileWriter*) {}

    //! @brief load internal state from file
    virtual void load(const std::string& /* path */, IFileReader*) {}

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

    //! @brief Returns time elapsed since the start of last call to computeForces()
    float stepElapsed() const { return timer.sumOfSteps(); }

    void printIterationTimings(const DomainType& domain, const ParticleDataType& simData)
    {
        if (rank_ > 0) { return; } // global particle and nc counts are only valid on rank 0
        const auto& d   = simData.hydro;
        const auto& box = domain.box();

        auto nodeCount        = domain.globalTree().numLeafNodes;
        auto particleCount    = domain.nParticles();
        auto haloCount        = d.maxHalos;
        auto totalNeighbors   = d.totalNeighbors;
        auto avgNcPerParticle = totalNeighbors / d.numParticlesGlobal;

        if (d.numParticlesGlobalPrev != d.numParticlesGlobal)
        {
            out << "### Check ### Particles (global): " << d.numParticlesGlobal
                << ", differs by: " << std::int64_t(d.numParticlesGlobal) - std::int64_t(d.numParticlesGlobalPrev)
                << std::endl;
        }
        out << "### Check ### Global Tree Nodes: " << nodeCount << ", Particles: " << particleCount
            << ", Halos: " << haloCount << std::endl;
        out << "### Check ### Computational domain: " << box.xmin() << " " << box.xmax() << " " << box.ymin() << " "
            << box.ymax() << " " << box.zmin() << " " << box.zmax() << std::endl;
        out << "### Check ### Total Neighbors: " << totalNeighbors
            << ", Avg neighbor count per particle: " << avgNcPerParticle << std::endl;
        out << "### Check ### Total time: " << d.ttot - d.minDt << ", current time-step: " << d.minDt << std::endl;
        out << "### Check ### Total energy: " << d.etot << ", (internal: " << d.eint << ", kinetic: " << d.ecin;
        out << ", gravitational: " << d.egrav;
        out << ")" << std::endl;
        out << "### Check ### Focus Tree Nodes: " << domain.focusTree().octreeViewAcc().numLeafNodes << ", maxDepth "
            << domain.focusTree().depth();
        if constexpr (d.useGpu)
        {
            out << ", maxStackNc " << d.stackUsedNc << ", maxStackGravity " << d.stackUsedGravity;
        }
        out << "\n=== Total time for iteration(" << d.iteration << ") " << timer.sumOfSteps() << "s\n\n";
    }

protected:
    static void outputAllocatedFields(IFileWriter* writer, ParticleDataType& simData, std::optional<std::span<const uint64_t>> selectedParticlesIndexes = std::nullopt)
    {
        auto output = [](auto& d, IFileWriter* writer, std::optional<std::span<const uint64_t>> selectedParticlesIndexes)
        {
            const bool  subsetOuput = selectedParticlesIndexes.has_value();
            auto        fieldPointers  = d.data();
            auto        selPartIndexes = selectedParticlesIndexes.value_or(std::span<const uint64_t>{});
            const auto& outputFieldIndices = subsetOuput ? d.subsetOutputFieldIndices : d.outputFieldIndices;
            auto        indicesDone    = outputFieldIndices;
            auto        namesDone      = subsetOuput ? d.subsetOutputFieldNames : d.outputFieldNames;

            for (int i = int(indicesDone.size()) - 1; i >= 0; --i)
            {
                int fidx = indicesDone[i];
                if (d.isAllocated(fidx))
                {
                    int column = std::find(outputFieldIndices.begin(), outputFieldIndices.end(), fidx) - outputFieldIndices.begin();
                    std::visit(
                        [writer, c = column, key = namesDone[i], selPartIndexes, subsetOuput](auto field)
                        {
                            if (!subsetOuput)
                            {
                                // Output field for all particles, no selection
                                auto&& tmp = cstone::toHost(*field);
                                writeField(writer, key, tmp.data(), c);
                            }
                            else
                            {
                                // Output field only for tagged particles
                                using ValueType = std::remove_pointer_t<decltype(field)>::value_type;
                                using Exec      = ParticleDataType::Exec;
                                using GatherVec = std::conditional_t<cstone::execution::HaveGpu<Exec>{},
                                                                     cstone::DeviceVector<ValueType>,
                                                                     std::vector<ValueType>>;
                                GatherVec gatherResult(selPartIndexes.size());
                                cstone::gather(selPartIndexes, field->data(), gatherResult.data());
                                auto&& tmp = cstone::toHost(gatherResult);
                                writeField(writer, key, tmp.data(), c);
                            }
                        },
                        fieldPointers[fidx]);
                    indicesDone.erase(indicesDone.begin() + i);
                    namesDone.erase(namesDone.begin() + i);
                }
            }

            if (!indicesDone.empty() && writer->rank() == 0)
            {
                std::cout << (subsetOuput ? "WARNING: the following subset fields are not in use and therefore not output: "
                                     : "WARNING: the following fields are not in use and therefore not output: ");
                for (std::size_t fidx = 0; fidx < indicesDone.size() - 1; ++fidx)
                {
                    std::cout << d.fieldNames[indicesDone[fidx]] << ",";
                }
                std::cout << d.fieldNames[indicesDone.back()] << std::endl;
            }
        };

        output(simData.hydro, writer, selectedParticlesIndexes);
        output(simData.chem, writer, selectedParticlesIndexes);
    }

    void logDomainStats(const DomainType& domain, ParticleDataType& simData)
    {
        timer.logStatistics("numParticles", domain.nParticles());
        timer.logStatistics("numHalos", domain.nParticlesWithHalos() - domain.nParticles());
        timer.logStatistics("numNeighborPairs", simData.hydro.localNeighbors);
        timer.logStatistics("assignment", domain.assignmentStart());

        auto hostMem = simData.hydro.memStats();
        timer.logStatistics("hostMemSizeBytes", hostMem[1]);
        timer.logStatistics("hostCapSizeBytes", hostMem[2]);

        using Exec = ParticleDataType::Exec;
        if constexpr (cstone::execution::HaveGpu<Exec>{})
        {
            auto devMem = simData.hydro.memStats();
            timer.logStatistics("devMemSizeBytes", devMem[1]);
            timer.logStatistics("devCapSizeBytes", devMem[2]);
            timer.logStatistics("devFreeSizeBytes", devMem[3]);
        }
    }

    std::ostream& out;
    Timer         timer;
    PmReader      pmReader;
    int           rank_;
};

} // namespace sphexa
