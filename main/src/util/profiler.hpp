#pragma once

#include <vector>
#include <iostream>
#include <mpi.h>

namespace sphexa
{

class FunctionProfilingData
{
private:
    std::vector<float*> funcTimes;        // vector of timesteps for all ranks
    std::vector<float>  funcMean;         // mean (average)
    std::vector<double> funcStdev;        // standard deviation
    std::vector<double> funcCov;          // c.o.v = std / mean
    std::vector<double> funcLambda;       // lambda = (Lmax / Lmean - 1) * 100%
    std::vector<float*> funcVectorMetric; // vector based metric calculation
    std::vector<double> funcI_2PerStep;   // distance to zero for vector based metric, name tentative
    std::vector<float>  funcSkewness;     // skewness
    std::vector<float>  funcKurtosis;     // kurtosis
    std::vector<float>  totalTimeStep;    // total time-step durations
    std::vector<size_t> numLocalParts;    // number of local particles

public:
    FunctionProfilingData() {}

    ~FunctionProfilingData() {}
};

class Profiler
{
private:
    std::vector<float*> timeSteps;     // vector of timesteps for all ranks
    std::vector<float>  meanPerStep;   // mean (average)
    std::vector<double> stdevPerStep;  // standard deviation
    std::vector<double> covPerStep;    // c.o.v = std / mean
    std::vector<double> lambdaPerStep; // lambda = (Lmax / Lmean - 1) * 100%
    std::vector<float*> vectorMetric;  // vector based metric calculation
    std::vector<double> I_2PerStep;    // distance to zero for vector based metric, name tentative
    std::vector<float>  g1PerStep;     // skewness
    std::vector<float>  g2PerStep;     // kurtosis
    int                 _numFunctions;
    int                 _numRanks;
    int                 _rank;
    std::ofstream       profilingFile;

public:
    Profiler(int rank)
        : _rank(rank)
    {
        MPI_Comm_size(MPI_COMM_WORLD, &_numRanks);
        std::string filename = "profiling-" + std::to_string(_numRanks) + "Ranks.txt";
        if (_rank == 0) profilingFile.open(filename);
    }
    ~Profiler() {}

    void printMetrics(int iteration)
    {
        profilingFile << std::setw(15) << meanPerStep.at(iteration) << "," << std::setw(15) << stdevPerStep.at(iteration)
                  << ",";
        profilingFile << std::setw(15) << covPerStep.at(iteration) << "," << std::setw(15) << I_2PerStep.at(iteration)
                  << ",";
        profilingFile << std::setw(15) << lambdaPerStep.at(iteration); // << ","; << std::setw(15) << I_2PerStep.at(iteration) << ",";
    }

    void printProfilingInfo()
    {
        if (_rank == 0)
        {
            size_t iter = 1;
            profilingFile << std::setw(3) << " ";
            for (int i = 0; i < _numRanks; i++)
            {
                profilingFile << std::setw(15) << "RANK" << i;
            }
            profilingFile << std::setw(16) << " mean" << std::setw(16) << " stdev";
            profilingFile << std::setw(16) << " C.o.V." << std::setw(16) << " I_2";
            profilingFile << std::setw(16) << " lambda" << std::endl;
            for (auto& element : timeSteps)
            {
                calculateMetrics(element);
                profilingFile << std::setw(3) << iter << " ";
                for (int i = 0; i < _numRanks; i++)
                {
                    profilingFile << std::setw(15) << element[i] << ",";
                }
                printMetrics(iter - 1);
                iter++;
                profilingFile << std::endl;
            }
            profilingFile << std::endl;
            std::cout << "Profiling data written." << std::endl;
        }
        profilingFile.close();
    }

    void saveTimings(float duration)
    {
        // float* dur = timeSteps.at(iteration-1);
        // float x = dur[iteration-1];
        float* durations = new float[_numRanks];
        MPI_Gather(&duration, 1, MPI_FLOAT, &durations[_rank], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

        if (_rank == 0)
        {
            timeSteps.push_back(durations);
            // calculate mean, stdev, c.o.v, g1, g2, I_2
            // calculateMetrics(durations);
            // std::cout << "dur1 = " << durations[0] << " , dur2 = " << durations[1] << std::endl;
            // std::cout << "ts1 = " << timeSteps.back()[0] << " , ts2 = " << timeSteps.back()[1] << std::endl;
        }
    }

    void gatherTimings(float duration)
    {
        // float* dur = timeSteps.at(iteration-1);
        // float x = dur[iteration-1];
        float* durations = new float[_numRanks];
        MPI_Gather(&duration, 1, MPI_FLOAT, &durations[_rank], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

        if (_rank == 0)
        {
            timeSteps.push_back(durations);
            // calculate mean, stdev, c.o.v, g1, g2, I_2
            // calculateMetrics(durations);
            // std::cout << "dur1 = " << durations[0] << " , dur2 = " << durations[1] << std::endl;
            // std::cout << "ts1 = " << timeSteps.back()[0] << " , ts2 = " << timeSteps.back()[1] << std::endl;
        }
    }

    // LiB metrics calculations
    void calculateMetrics(float* durations)
    {
        const std::vector<float> durData(durations, durations + _numRanks);
        float* vecMetric = new float[_numRanks];

        float mean   = std::reduce(durData.begin(), durData.end()) / durData.size();
        float sq_sum = std::inner_product(durData.begin(), durData.end(), durData.begin(), 0.0);
        float stdev  = std::sqrt(sq_sum / durData.size() - mean * mean);
        float Lmax   = *std::max_element(durData.begin(), durData.end());
        float lambda = (Lmax/mean - 1.0f) * 100;

        for (int i = 0; i < _numRanks; i++)
        {
            vecMetric[i] = 1 - (durData.at(i) / mean);
        }
        float sumsqVector = std::inner_product(vecMetric, vecMetric + _numRanks, vecMetric, 0.0);

        meanPerStep.push_back(mean);
        stdevPerStep.push_back(stdev);
        lambdaPerStep.push_back(lambda);
        covPerStep.push_back(stdev / mean);
        vectorMetric.push_back(vecMetric);
        I_2PerStep.push_back(std::sqrt(sumsqVector));
    }
};

} // namespace sphexa
