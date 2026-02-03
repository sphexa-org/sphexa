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

#include "cstone/cuda/cuda_utils.hpp"
#include "cstone/cuda/device_vector.h"
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
    using Acc = typename ParticleDataType::AcceleratorType;
    template<class VType>
    using AccVector = std::conditional_t<cstone::HaveGpu<Acc>{}, cstone::DeviceVector<VType>, std::vector<VType>>;


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

    // TODO: can be in the saveSubsetsFields method below if we don't need to override it in derived classes (for example in case of EOS related ouput)
    //! @brief save particle subset data fields to file
    virtual void saveSubFields(IFileWriter* writer, std::span<const uint64_t> selectedParticlesIndexes, ParticleDataType& simData) 
    {
        outputAllocatedFields(writer, simData, selectedParticlesIndexes);
        timer.step("SubsetFileOutput");
    }

    //! @brief save selected particle data fields to file
    // TODO: ParticleDataType should be const ParticleDataType& but at some point we use the data() method which is non-const
    // TODO: why in saveFields a file name parameter is not present?  
    void saveSubsetsFields(IFileWriter* writer, std::string selParticlesOutFile, size_t first, size_t last, ParticleDataType& simData)
    {
        // Find the selected particles positions in dataset
        // TODO: check selectedParticlesIndexes template parameter type
        AccVector<uint64_t> selectedParticlesIndexes;
        if constexpr (cstone::HaveGpu<typename ParticleDataType::AcceleratorType>{}) {
            // cstone::DeviceVector<uint64_t> selectedParticlesIndexesDev;
            findTaggedIdsGPU(std::span<const uint64_t>(simData.hydro.id.data(), simData.hydro.id.size()), first, last, selectedParticlesIndexes);
            // selectedParticlesIndexes.resize(selectedParticlesIndexesDev.size());
            // size_t transferSize = selectedParticlesIndexesDev.size() * sizeof(uint64_t);
            // memcpyH2D(rawPtr(selectedParticlesIndexesDev), transferSize, selectedParticlesIndexes.data());
        }
        else {
            // TODO: why different from  findTaggedIdsGPU
            findTaggedIds(std::span<const uint64_t>(simData.hydro.id), first, last, selectedParticlesIndexes);
        }
        timer.step("FindTaggedIds");

        writer->addStep(0, selectedParticlesIndexes.size(), selParticlesOutFile);
        writer->stepAttribute("iteration", &simData.hydro.iteration, 1);
        saveSubFields(writer, selectedParticlesIndexes, simData);
        writer->closeStep();
    }

    //! @brief save internal state to file
    virtual void save(IFileWriter*) {}

    //! @brief load internal state from file
    virtual void load(const std::string& path, IFileReader*) {}

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
        const auto& d   = simData.hydro;
        const auto& box = domain.box();

        auto nodeCount          = domain.globalTree().numLeafNodes;
        auto particleCount      = domain.nParticles();
        auto haloCount          = d.maxHalos;
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
            out << ", maxStackNc " << d.stackUsedNc << ", maxStackGravity " << d.stackUsedGravity;
        }
        out << "\n=== Total time for iteration(" << d.iteration << ") " << timer.sumOfSteps() << "s\n\n";
    }

protected:

    // static void outputAllocatedFields(IFileWriter* writer, ParticleDataType& simData)
    // {
    //     auto output = [](auto& d, IFileWriter* writer)
    //     {
    //         auto fieldPointers = d.data();
    //         auto indicesDone   = d.outputFieldIndices;
    //         auto namesDone     = d.outputFieldNames;

    //         for (int i = int(indicesDone.size()) - 1; i >= 0; --i)
    //         {
    //             int fidx = indicesDone[i];
    //             if (d.isAllocated(fidx))
    //             {
    //                 int column = std::find(d.outputFieldIndices.begin(), d.outputFieldIndices.end(), fidx) -
    //                              d.outputFieldIndices.begin();
    //                 std::visit(
    //                     [writer, c = column, key = namesDone[i]](auto field)
    //                     {
    //                         auto&& tmp = toHost(*field);
    //                         writeField(writer, key, tmp.data(), c);
    //                     },
    //                     fieldPointers[fidx]);
    //                 indicesDone.erase(indicesDone.begin() + i);
    //                 namesDone.erase(namesDone.begin() + i);
    //             }
    //         }

    //         if (!indicesDone.empty() && writer->rank() == 0)
    //         {
    //             std::cout << "WARNING: the following fields are not in use and therefore not output: ";
    //             for (int fidx = 0; fidx < indicesDone.size() - 1; ++fidx)
    //             {
    //                 std::cout << d.fieldNames[fidx] << ",";
    //             }
    //             std::cout << d.fieldNames[indicesDone.back()] << std::endl;
    //         }
    //     };

    //     output(simData.hydro, writer);
    //     output(simData.chem, writer);
    // }

    static void outputAllocatedFields(IFileWriter* writer, ParticleDataType& simData, std::optional<std::span<const uint64_t>> selectedParticlesIndexes = std::nullopt)
    {
        auto output = [](auto& d, IFileWriter* writer, std::optional<std::span<const uint64_t>> selectedParticlesIndexes = std::nullopt)
        {
            auto fieldPointers = d.data();
            auto indicesDone   = d.outputFieldIndices;
            auto namesDone     = d.outputFieldNames;
            AccVector<char> buffer;

            for (int i = int(indicesDone.size()) - 1; i >= 0; --i)
            {
                int fidx = indicesDone[i];
                if (d.isAllocated(fidx))
                {
                    int column = std::find(d.outputFieldIndices.begin(), d.outputFieldIndices.end(), fidx) -
                                 d.outputFieldIndices.begin();
                    if(selectedParticlesIndexes != std::nullopt)
                    {
                        std::visit(
                            [writer, c = column, key = namesDone[i], &selectedParticlesIndexes, &buffer](auto field)
                            {
                                const auto selIndexes = selectedParticlesIndexes.value();
                                using ValueType = std::remove_pointer_t<decltype(field)>::value_type;
                                auto packedBuffer = util::packAllocBuffer<ValueType>(buffer, std::vector<size_t>{selIndexes.size()}, 1);
                                constexpr bool gpu = cstone::HaveGpu<Acc>{};
                                cstone::gatherAcc<gpu>(selIndexes, field->data(), packedBuffer[0].data());
                                auto&& tmp = toHost(buffer);
                                writeField(writer, key, tmp.data(), c);
                            },
                            fieldPointers[fidx]);
                    }
                    else
                    {
                    std::visit(
                        [writer, c = column, key = namesDone[i]](auto field)
                        {
                            auto&& tmp = toHost(*field);
                            writeField(writer, key, tmp.data(), c);
                        },
                        fieldPointers[fidx]);
                    }
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

        output(simData.hydro, writer, selectedParticlesIndexes);
        output(simData.chem, writer, selectedParticlesIndexes);
    }


    // // TODO: is there any way to avoid code duplication with outputAllocatedFields()?
    // static void outputSubsetAllocatedFields(IFileWriter* writer, std::span<const uint64_t> selectedParticlesIndexes, 
    //     ParticleDataType& simData)
    // {
    //     auto output = [](auto& d, IFileWriter* writer, const std::span<const uint64_t> selectedParticlesIndexes)
    //     {
    //         auto indicesDone   = d.subsetOutputFieldIndices;
    //         auto namesDone     = d.subsetOutputFieldNames;
    //         // TODO: use AccVector for subsetField (resize/clean if needed)
    //         // TODO: have a look to PackedBuffer and PackAlloc
    //         for (int i = int(indicesDone.size()) - 1; i >= 0; --i)
    //         {
    //             int fidx = indicesDone[i];
    //             if (d.isAllocated(fidx))
    //             {
    //                 int column = std::find(d.subsetOutputFieldIndices.begin(), d.subsetOutputFieldIndices.end(), fidx) -
    //                              d.subsetOutputFieldIndices.begin();
                    
    //                 // TODO: passing the entire data is not needed
    //                 using FieldVariant = std::variant<std::vector<float>, std::vector<double>, std::vector<unsigned>, std::vector<uint64_t>, std::vector<uint8_t>>;                    
    //                 FieldVariant subsetField;
    //                 createHostSubsetFieldDataset(d, selectedParticlesIndexes, fidx, subsetField);

    //                 std::visit([writer, c = column, key = namesDone[i]](auto field)
    //                            { writeField(writer, key, field.data(), c); }, subsetField);

    //                 indicesDone.erase(indicesDone.begin() + i);
    //                 namesDone.erase(namesDone.begin() + i);
    //             }
    //         }

    //         if (!indicesDone.empty() && writer->rank() == 0)
    //         {
    //             std::cout << "WARNING: the following fields are not in use and therefore not output for subset: ";
    //             for (int fidx = 0; fidx < indicesDone.size() - 1; ++fidx)
    //             {
    //                 std::cout << d.fieldNames[fidx] << ",";
    //             }
    //             std::cout << d.fieldNames[indicesDone.back()] << std::endl;
    //         }
    //     };

    //     output(simData.hydro, writer, selectedParticlesIndexes);
    //     output(simData.chem, writer, selectedParticlesIndexes);
    // }


    void logDomainStats(const DomainType& domain, ParticleDataType& simData)
    {
        timer.logStatistics("numParticles", domain.nParticles());
        timer.logStatistics("numHalos", domain.nParticlesWithHalos() - domain.nParticles());
        timer.logStatistics("assignment", domain.assignmentStart());

        auto hostMem = simData.hydro.memStats();
        timer.logStatistics("hostMemSizeBytes", hostMem[1]);
        timer.logStatistics("hostCapSizeBytes", hostMem[2]);

        using AccType = ParticleDataType::AcceleratorType;
        if constexpr (cstone::HaveGpu<AccType>{})
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
