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
 * @brief SPH-EXA application front-end and main function
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 * @author Aurelien Cavelan
 * @author Jose A. Escartin <ja.escartin@gmail.com>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "cstone/domain/domain.hpp"

#include "init/factory.hpp"
#include "io/arg_parser.hpp"
#include "io/factory.hpp"
#include "observables/factory.hpp"
#include "propagator/factory.hpp"
#include "sph/types.hpp"
#include "util/timer.hpp"
#include "util/utils.hpp"

#include "particle_selection.hpp"
#include "simulation_data.hpp"
#include "insitu_viz.h"

#ifdef USE_CUDA
using AccType = cstone::GpuTag;
#else
using AccType = cstone::CpuTag;
#endif

namespace fs = std::filesystem;
using namespace sphexa;

bool stopConditionReached(size_t iteration, double time, const std::string& maxStepStr);
bool syncedWallClockElapsed(float totalTimeElapsed, float wallClockLimit, float dt);
void printHelp(char* binName, int rank);
int  getNumLocalRanks(int);

int main(int argc, char** argv)
{
    auto [rank, numRanks] = initMpi();
    const ArgParser parser(argc, (const char**)argv);

    if (parser.exists("-h") || parser.exists("--h") || parser.exists("-help") || parser.exists("--help"))
    {
        printHelp(argv[0], rank);
        return exitSuccess();
    }

    using Dataset = SimulationData<AccType>;
    using Domain  = cstone::Domain<sph::SphTypes::KeyType, sph::SphTypes::CoordinateType, AccType>;
    using ParticleIndexVectorType = Propagator<Domain, Dataset>::ParticleIndexVectorType;

    const std::string        initCond     = parser.get("--init");
    const size_t             problemSize  = parser.get("-n", 50);
    const std::string        glassBlock   = parser.get("--glass");
    const std::string        propChoice   = parser.get("--prop", std::string("ve"));
    const std::string        maxStepStr   = parser.get("-s", std::string("200"));
    std::vector<std::string> writeExtra   = parser.getCommaList("--wextra");
    std::vector<std::string> outputFields = parser.getCommaList("-f");
    const bool               ascii        = parser.exists("--ascii");
    const bool               quiet        = parser.exists("--quiet");
    const bool               avClean      = parser.exists("--avclean");
    const int                simDuration  = parser.get("--duration", std::numeric_limits<int>::max());
    const std::string        writeFreqStr = parser.get("-w", std::string("0"));
    const bool               writeEnabled = writeFreqStr != "0" || !writeExtra.empty();
    const std::string        profFreqStr  = parser.get("--profile", maxStepStr);
    const bool               profEnabled  = parser.exists("--profile");
    const std::string        pmroot       = parser.get("--pmroot", std::string("/sys/cray/pm_counters"));
    const std::string        idSel        = parser.get("--idSel");
    std::string              outFile      = parser.get("-o", "dump_" + removeModifiers(initCond));


    std::ofstream nullOutput("/dev/null");
    std::ostream& output = (quiet || rank) ? nullOutput : std::cout;
    std::ofstream constantsFile(fs::path(outFile).parent_path() / fs::path("constants.txt"));

    //! @brief evaluate user choice for different kind of actions
    auto fileWriter  = fileWriterFactory(ascii, MPI_COMM_WORLD);
    auto selParticlesFileWriter  = fileWriterFactory(ascii, MPI_COMM_WORLD);
    auto fileReader  = fileReaderFactory(ascii, MPI_COMM_WORLD);
    auto simInit     = initializerFactory<Dataset>(initCond, glassBlock, fileReader.get());
    auto propagator  = propagatorFactory<Domain, Dataset>(propChoice, avClean, output, rank, simInit->constants());
    auto observables = observablesFactory<Dataset>(simInit->constants(), constantsFile);

    // Particle selection for output
//    auto particleSelectionId = particleSelection(idSel, fileReader.get());
    ParticleIndexVectorType selParticlesIds;
    ParticleIndexVectorType localSelectedParticlesIndexes;
    ParticleIndexVectorType localSelectedParticlesIndexes_thrust;
    bool tagSelectedParticles = false;
    bool isSelectedParticleOutputTriggered = true;
    std::string selParticlesOutFile;
    bool saveSelParticles = !idSel.empty();
    saveSelParticles = true;
    selParticlesIds.push_back(3);
    selParticlesIds.push_back(23);
    selParticlesIds.push_back(15000);
    selParticlesIds.push_back(38500);
    selParticlesIds.push_back(46000);
    if(saveSelParticles) {

        // Activate particle selected tagging
        tagSelectedParticles = true;

        // Set file name for selected particles output
        selParticlesOutFile = "selected_particles_" + outFile;
    }

    Dataset simData;
    simData.comm = MPI_COMM_WORLD;

    Timer totalTimer(output);
    MPI_Barrier(MPI_COMM_WORLD);
    totalTimer.start();

    propagator->addCounters(profEnabled ? pmroot : "", getNumLocalRanks(numRanks));
    propagator->activateFields(simData);
    propagator->load(initCond, fileReader.get());
    auto box = simInit->init(rank, numRanks, problemSize, simData, fileReader.get());
    auto& d = simData.hydro;
    transferAllocatedToDevice(d, 0, d.x.size(), propagator->conservedFields());
    simData.setOutputFields(outputFields.empty() ? propagator->conservedFields() : outputFields);

    if (parser.exists("--G")) { d.g = parser.get<double>("--G"); }
    bool  haveGrav = (d.g != 0.0);
    float theta    = parser.get("--theta", haveGrav ? 0.5f : 1.0f);

    if (!parser.exists("-o")) { outFile += fileWriter->suffix(); }
    if (writeEnabled) { writeSettings(simInit->constants(), outFile, fileWriter.get()); }
    if (rank == 0) { std::cout << "Data generated for " << d.numParticlesGlobal << " global particles\n"; }

    uint64_t bucketSizeFocus = 64;
    // ~100 global nodes per rank to decompose the domain with +-1% accuracy
    uint64_t bucketSize = std::max(bucketSizeFocus, d.numParticlesGlobal / (100 * numRanks));
    Domain   domain(rank, numRanks, bucketSize, bucketSizeFocus, theta, box);
    domain.setGrowthAllocRate(simData.hydro.getAllocGrowthRate());

    propagator->sync(domain, simData);

    viz::init_catalyst(argc, argv);
    viz::init_ascent(d, domain.startIndex());

    size_t startIteration    = d.iteration;
    bool   isOutputTriggered = false;

    for (bool keepRunning = true; keepRunning; d.iteration++)
    {
        propagator->computeForces(domain, simData);
        box = domain.box();

        if (propagator->isSynced())
        {
            observables->computeAndWrite(simData, domain.startIndex(), domain.endIndex(), box);
        }

        if (tagSelectedParticles) {

            if (rank == 0) { std::cout << "Execution of selected particle identification\n"; }
            // TODO: Check if selected particle tagging is needed:
            // in a simulation starting from scratch we need to tag the selected particles if saveSelParticles is true and only in first iteration.
            // In a simulation starting from a checkpoint file we need to tag the selected particles only if they are not already tagged? 
            // Implementation of a check for tag existence in a checkpoint file is currently missing.
            // Find the selected particles in local id list and tag them by setting the MSB of the id field
            // TODO: move to GPU/sync with GPU
            std::for_each(selParticlesIds.begin(), selParticlesIds.end(), [&idList = d.id](auto selParticleId){
                const auto selParticleIndex = std::find(idList.begin(), idList.end(), selParticleId) - idList.begin();
                if (selParticleIndex < idList.size()) {
                   idList[selParticleIndex] = idList[selParticleIndex] | msbMask;
                }
            });

            // Update id on device
            // TODO: check transfer index range
            transferToDevice(d, 0, d.id.size(), {"id"});
            tagSelectedParticles = false;
        }

        bool isWallClockReached = syncedWallClockElapsed(totalTimer.elapsed(), simDuration, propagator->stepElapsed());

        isOutputTriggered = isOutputStep(d.iteration, writeFreqStr) ||
                            isOutputTime(d.ttot - d.minDt, d.ttot, writeFreqStr) ||
                            isExtraOutputStep(d.iteration, d.ttot - d.minDt, d.ttot, writeExtra) ||
                            (isWallClockReached && writeEnabled) || isOutputTriggered;

        if (isOutputTriggered && propagator->isSynced())
        {
            fileWriter->addStep(domain.startIndex(), domain.endIndex(), outFile);
            simData.hydro.loadOrStoreAttributes(fileWriter.get());
            box.loadOrStore(fileWriter.get());
            propagator->saveFields(fileWriter.get(), domain.startIndex(), domain.endIndex(), simData, box);
            propagator->save(fileWriter.get());
            fileWriter->closeStep();
            isOutputTriggered = false;
        }


        if (isSelectedParticleOutputTriggered) // && propatagor->isSynced
        {
            // TODO: what about MPI task sync at this point? I'm assuming everything is synced...
            // Reset the list of indexes of subdomain selected particles: this is needed since a specific particle can move to another rank
            localSelectedParticlesIndexes.clear();
            localSelectedParticlesIndexes_thrust.clear();

            // Find the selected particles in local id list and save their indexes
            unsigned int particleIndex = 0;
            // TODO: use d.devData to access device data for a GPU implementation!
            std::for_each(d.id.begin(), d.id.end(), [&localSelectedParticlesIndexes, &particleIndex](auto& particleId){
                if((particleId & msbMask) != 0) {// check MSB
                    localSelectedParticlesIndexes.push_back(particleIndex); // TODO: inefficient due to resizing, avoid push_back usage
                }
                particleIndex++;
            });

            findSelectedParticlesIndexes(d, domain.startIndex(), domain.endIndex(), localSelectedParticlesIndexes_thrust);

            // TODO: debug
            MPI_Barrier(MPI_COMM_WORLD);
            for (auto rankId = 0; rankId < numRanks; ++rankId) {
                if (rank == rankId) {
                    std::cout<<"Rank "<<rankId<<std::endl;
                    std::cout<<"Selected particle positions: "<<localSelectedParticlesIndexes_thrust.size()<<" "
                        <<localSelectedParticlesIndexes.size()<<std::endl;
                    for(auto i = 0; i < localSelectedParticlesIndexes_thrust.size(); ++i) {
                        std::cout<<localSelectedParticlesIndexes_thrust[i]<<" ";
                    }
                    std::cout<<std::endl;
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }


            selParticlesFileWriter->addStep(0, localSelectedParticlesIndexes.size(), selParticlesOutFile);
            simData.hydro.loadOrStoreAttributes(selParticlesFileWriter.get());
            box.loadOrStore(selParticlesFileWriter.get());
            propagator->saveSelParticlesFields(selParticlesFileWriter.get(), domain.startIndex(), domain.endIndex(), localSelectedParticlesIndexes, simData.hydro);
            selParticlesFileWriter->closeStep();
//            isSelectedParticleOutputTriggered = false;

        }

        if (isOutputStep(d.iteration, profFreqStr) || isOutputTime(d.ttot - d.minDt, d.ttot, profFreqStr) ||
            isWallClockReached)
        {
            if (profEnabled) { propagator->writeMetrics(fileWriter.get(), "profile"); }
        }

        keepRunning = not(stopConditionReached(d.iteration, d.ttot, maxStepStr) || isWallClockReached) ||
                      not propagator->isSynced();

        viz::execute(d, domain.startIndex(), domain.endIndex());

        propagator->integrate(domain, simData);
        propagator->printIterationTimings(domain, simData);
    }
    totalTimer.step("Total execution time of " + std::to_string(d.iteration - startIteration) + " iterations of " +
                    initCond + " up to t = " + std::to_string(d.ttot));

    constantsFile.close();
    viz::finalize();
    return exitSuccess();
}

//! @brief check whether the stop conditions based on evolved time (not wall-clock) or iteration count are reached
bool stopConditionReached(size_t iteration, double time, const std::string& maxStepStr)
{
    bool lastIteration = strIsIntegral(maxStepStr) && iteration >= std::stoi(maxStepStr);
    bool simTimeLimit  = !strIsIntegral(maxStepStr) && time > std::stod(maxStepStr);

    return lastIteration || simTimeLimit;
}

/*! @brief check whether wall clock limit was reached on any rank
 *
 * We do this to account for the fact that the total runtime timestamp might not be taken at exactly the same moment
 * across ranks.
 */
bool syncedWallClockElapsed(float totalTimeElapsed, float wallClockLimit, float dt)
{
    // if total time elapsed is getting close to (within dt) of the limit
    if (totalTimeElapsed + dt > wallClockLimit)
    {
        int isLimitReachedAny = totalTimeElapsed > wallClockLimit;
        mpiAllreduce(MPI_IN_PLACE, &isLimitReachedAny, 1, MPI_SUM);
        return isLimitReachedAny;
    }
    return false;
}

int getNumLocalRanks(int defValue)
{
    return getenv("SLURM_NTASKS_PER_NODE") == nullptr ? defValue : std::stoi(getenv("SLURM_NTASKS_PER_NODE"));
}

void printHelp(char* name, int rank)
{
    if (rank == 0)
    {
        printf("\nUsage:\n\n");
        printf("%s [OPTIONS]\n", name);
        printf("\nWhere possible options are:\n\n");

        printf("\t--init \t\t Test case selection (evrard, sedov, noh, isobaric-cube, wind-shock, turbulence)\n"
               "\t\t\t or an HDF5 file with initial conditions\n\n");
        printf("\t-n NUM \t\t Initialize data with (approx when using glass blocks) NUM^3 global particles [50]\n");
        printf("\t--glass FILE\t Use glass block as template to generate initial x,y,z configuration\n\n");

        printf("\t--theta NUM \t Gravity accuracy parameter [default 0.5 when self-gravity is active]\n\n");

        printf("\t--G NUM \t Gravitational constant [default dependent on test-case selection]\n\n");

        printf("\t--prop STRING \t Choice of SPH propagator [default: modern SPH]. For standard SPH, use \"std\" \n\n");

        printf("\t-s NUM \t\t int(NUM):  Number of iterations (time-steps) [200],\n\
                \t real(NUM): Time   of simulation (time-model)\n\n");

        printf("\t--wextra LIST \t Comma-separated list of steps (integers) or ~times (floating point)\n"
               "\t\t\t at which to trigger file output\n"
               "\t\t\t e.g.: --wextra 1,10,0.77 (output at after iteration 1 and 10 and at simulation time 0.77s\n\n");

        printf("\t-w NUM \t\t NUM<=0:    Disable file output [default],\n\
                \t int(NUM):  Dump particle data every NUM iteration steps,\n\
                \t real(NUM): Dump particle data every NUM seconds of simulation (not wall-clock) time \n\n");

        printf("\t-f LIST \t Comma-separated list of field names to write for each dump.\n"
               "\t\t\t e.g: -f x,y,z,h,rho\n"
               "\t\t\t If omitted, the list will be set to all conserved fields,\n"
               "\t\t\t resulting in a restartable output file\n\n");

        printf("\t--ascii \t Dump file in ASCII format [binary HDF5 by default]\n\n");

        printf("\t--outDir PATH \t Path to directory where output will be saved [./].\n\
                    \t Note that directory must exist and be provided with ending slash,\n\
                    \t e.g: --outDir /home/user/folderToSaveOutputFiles/\n\n");

        printf("\t--quiet \t Don't print anything to stdout\n\n");

        printf("\t--duration \t Maximum wall-clock run time of the simulation in seconds.[MAX_INT]\n\n");
    }
}
