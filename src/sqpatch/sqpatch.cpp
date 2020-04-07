#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "sphexa.hpp"
#include "SqPatchDataGenerator.hpp"
#include "SqPatchFileWriter.hpp"

#include <fenv.h>
#pragma STDC FENV_ACCESS ON


using namespace sphexa;

void printHelp(char *binName, int rank);

int main(int argc, char **argv)
{
    std::feclearexcept(FE_ALL_EXCEPT);
#ifdef NDEBUG
    enable_fe_hwexceptions(); // we want to crash if we see NANs or INFs!
#endif

    const int rank = initAndGetRankId();

    const ArgParser parser(argc, argv);

    if (parser.exists("-h") || parser.exists("--h") || parser.exists("-help") || parser.exists("--help"))
    {
        printHelp(argv[0], rank);
        return exitSuccess();
    }

    const size_t cubeSide = parser.getInt("-n", 50);
    const size_t maxStep = parser.getInt("-s", 10);
    const int writeFrequency = parser.getInt("-w", -1);
    const bool quiet = parser.exists("--quiet");
    const std::string outDirectory = parser.getString("--outDir");

    std::ofstream nullOutput("/dev/null");
    std::ostream &output = quiet ? nullOutput : std::cout;

    using Real = double;
    using Dataset = ParticlesData<Real>;
    using Tree = Octree<Real>;

#ifdef USE_MPI
    DistributedDomain<Real, Dataset, Tree> domain;
    const IFileWriter<Dataset> &fileWriter = SqPatchMPIFileWriter<Dataset>();
#else
    Domain<Real, Dataset, Tree> domain;
    const IFileWriter<Dataset> &fileWriter = SqPatchFileWriter<Dataset>();
#endif

    auto d = SqPatchDataGenerator<Real>::generate(cubeSide);
    const Printer<Dataset> printer(d);

    MasterProcessTimer timer(output, d.rank), totalTimer(output, d.rank);

    std::ofstream constantsFile(outDirectory + "constants.txt");

    Tree::bucketSize = 64;
    Tree::minGlobalBucketSize = 512;
    Tree::maxGlobalBucketSize = 2048;
    domain.create(d);

    const size_t nTasks = 1;
    const size_t ngmax = 750;
    const size_t ng0 = 250;
    TaskList taskList = TaskList(domain.clist, nTasks, ngmax, ng0);

    // want to dump on floating point exceptions
    bool fpe_raised = false;
    totalTimer.start();
    for (d.iteration = 0; d.iteration <= maxStep; d.iteration++)
    {
        timer.start();

        domain.update(d);
        timer.step("domain::distribute");
        domain.synchronizeHalos(&d.x, &d.y, &d.z, &d.h, &d.xmass);  // also synchronize VE estimator function xmass!
        timer.step("mpi::synchronizeHalos");
        domain.buildTree(d);
        timer.step("domain::buildTree");
        taskList.update(domain.clist);
        timer.step("updateTasks");
        sph::findNeighbors(domain.octree, taskList.tasks, d);
        if (d.iteration == 0) { // todo: refactor this!
            sph::findNeighbors(domain.octree, taskList.tasks, d);
        }
        timer.step("FindNeighbors");
        sph::computeDensity<Real>(taskList.tasks, d);  // initial guess for density...
        timer.step("Density");
        if (d.iteration == 0) { sph::initFluidDensityAtRest<Real>(taskList.tasks, d); }
#ifdef DO_NEWTONRAPHSON
        if (d.iteration > d.starthNR) {
            sph::newtonRaphson<Real>(taskList.tasks, d);
            timer.step("hNR");
            sph::findNeighbors(domain.octree, taskList.tasks, d);
            timer.step("FindNeighbors");
//            without those, it crashes around 1971... sometimes the debugger shows variable values, sometimes not...
//            without those on the cluster, but with -g and DEBUG flags, I get index out of bounds at line 188 in o.findneighbors() in iteration 1379. Also ngmax=-1419685523 which is odd (maybe display error in gdb from conversion to size_t(as defined) to int (as in function signature))
            for (int iterNR = 0; iterNR < 2; iterNR++) {
                sph::computeDensity<Real>(taskList.tasks, d);
                timer.step("Density");
                sph::newtonRaphson<Real>(taskList.tasks, d);
                timer.step("hNR");
            sph::findNeighbors(domain.octree, taskList.tasks, d);
            timer.step("FindNeighbors");
            }
        }
#endif
        sph::calcGradhTerms<Real>(taskList.tasks, d);
        timer.step("calcGradhTerms");
        sph::computeEquationOfStateSphynxWater<Real>(taskList.tasks, d);
        timer.step("EquationOfState");
        domain.synchronizeHalos(&d.vx, &d.vy, &d.vz, &d.ro, &d.p, &d.c, &d.sumkx, &d.gradh, &d.h, &d.vol);  // also synchronize sumkx after density! Synchronize also h for h[j] accesses in momentum and energy
        timer.step("mpi::synchronizeHalos");
        sph::computeIAD<Real>(taskList.tasks, d);
        timer.step("IAD");
        domain.synchronizeHalos(&d.c11, &d.c12, &d.c13, &d.c22, &d.c23, &d.c33);
        timer.step("mpi::synchronizeHalos");
        sph::computeMomentumAndEnergyIAD<Real>(taskList.tasks, d);
        timer.step("MomentumEnergyIAD");
        sph::computeTimestep<Real, sph::TimestepPress2ndOrder<Real, Dataset>>(taskList.tasks, d);
        timer.step("Timestep"); // AllReduce(min:dt)
        sph::computePositions<Real, sph::computeAcceleration<Real, Dataset>>(taskList.tasks, d);
        timer.step("UpdateQuantities");
        sph::computeTotalEnergy<Real>(taskList.tasks, d);
        timer.step("EnergyConservation"); // AllReduce(sum:ecin,ein)
        sph::updateSmoothingLength<Real>(taskList.tasks, d);
        timer.step("UpdateSmoothingLength");

        const size_t totalNeighbors = sph::neighborsSum(taskList.tasks);
        const size_t maxNeighbors = sph::neighborsMax(taskList.tasks);
        if (d.rank == 0)
        {
            printer.printCheck(d.count, domain.octree.globalNodeCount, d.x.size() - d.count, totalNeighbors, maxNeighbors, output);
            printer.printConstants(d.iteration, totalNeighbors, maxNeighbors, constantsFile);
        }
#ifndef NDEBUG
        fpe_raised = all_check_FPE("after print, rank " + std::to_string(d.rank));
        if (fpe_raised) break;
#endif
        if ((writeFrequency > 0 && d.iteration % writeFrequency == 0) || writeFrequency == 0)
        {
            fileWriter.dumpParticleDataToAsciiFile(d, domain.clist, outDirectory + "dump_sqpatch" + std::to_string(d.iteration) + ".txt");
            // fileWriter.dumpParticleDataToBinFile(d, outDirectory + "dump_sqpatch" + std::to_string(d.iteration) + ".bin");
            timer.step("writeFile");
        }

        timer.stop();

        if (d.rank == 0) printer.printTotalIterationTime(timer.duration(), output);
    }
#ifndef NDEBUG
    if (fpe_raised) {
        fileWriter.dumpParticleDataToAsciiFile(d, domain.clist, outDirectory + "fperrordump_sqpatch" + std::to_string(d.iteration) + "_" + std::to_string(std::time(0)) + ".txt");
    }
#endif

    totalTimer.step("Total execution time of " + std::to_string(maxStep) + " iterations of SqPatch");

    constantsFile.close();

    return exitSuccess();
}

void printHelp(char *name, int rank)
{
    if (rank == 0)
    {
        printf("\nUsage:\n\n");
        printf("%s [OPTIONS]\n", name);
        printf("\nWhere possible options are:\n");
        printf("\t-n NUM \t\t\t NUM^3 Number of particles\n");
        printf("\t-s NUM \t\t\t NUM Number of iterations (time-steps)\n");
        printf("\t-w NUM \t\t\t Dump particles data every NUM iterations (time-steps)\n\n");

        printf("\t--quiet \t\t Don't print anything to stdout\n\n");

        printf("\t--outDir PATH \t\t Path to directory where output will be saved.\
                    \n\t\t\t\t Note that directory must exist and be provided with ending slash.\
                    \n\t\t\t\t Example: --outDir /home/user/folderToSaveOutputFiles/\n");
    }
}
