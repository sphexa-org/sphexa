/*
 * MIT License
 *
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
 * @brief Cray power measurement counter reading
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com
 */

#pragma once

#include <cstdio>
#include <filesystem>
#include <functional>
#include <vector>

namespace sphexa
{

static int readPmCounter(const char* fname, unsigned long long* joules, unsigned long long* ms)
{
    auto file = fopen(fname, "r");
    if (file == nullptr) { return 1; }

    if (fscanf(file, "%llu %*s %llu %*s", joules, ms) != 2)
    {
        fprintf(stderr, "ERROR: unable to parse file %s\n", fname);
        fclose(file);
        return 1;
    }

    return fclose(file);
}

class PmReader
{
    using PmType = unsigned long long;

public:
    explicit PmReader(int rank)
        : rank_(rank)
    {
    }

    void addCounters(const std::string& pmRoot, int numRanksPerNode)
    {
        numRanksPerNode_ = numRanksPerNode;
        // energy per compute node, only the first rank per node reads the node energy counter
        {
            std::string path   = pmRoot + "/energy";
            bool        enable = (rank_ % numRanksPerNode == 0) && std::filesystem::exists(path);
            pmCounters.emplace_back("node", path, std::vector<PmType>{}, std::vector<PmType>{}, enable);
        }
        // energy per accelerator
        {
            std::string path   = pmRoot + "/accel" + std::to_string(rank_ % numRanksPerNode) + "_energy";
            bool        enable = std::filesystem::exists(path);
            pmCounters.emplace_back("acc", path, std::vector<PmType>{}, std::vector<PmType>{}, enable);
        }
    }

    void start()
    {
        numStartCalled_++;
        readPm();
    }

    void step() { readPm(); }

    template<class Archive>
    void writeTimings(Archive* ar, const std::string& outFile)
    {
        auto rebaseSeries = [](auto& vec)
        {
            auto baseVal = vec[0];
            for (auto& v : vec)
            {
                v -= baseVal;
            }
            std::vector<float> vecFloat(vec.size());
            std::copy(vec.begin(), vec.end(), vecFloat.begin());
            return vecFloat;
        };

        int numRanks = ar->numRanks();
        for (auto& counter : pmCounters)
        {
            auto  pmName              = get<0>(counter);
            auto& pmValues            = get<2>(counter);
            auto& pmTimeStamps        = get<3>(counter);
            auto  pmValuesRebased     = rebaseSeries(pmValues);
            auto  pmTimeStampsRebased = rebaseSeries(pmTimeStamps);

            ar->addStep(0, pmValues.size(), outFile + ar->suffix());
            ar->stepAttribute("numRanks", &numRanks, 1);
            ar->stepAttribute("numRanksPerNode", &numRanksPerNode_, 1);
            ar->stepAttribute("numIterations", &numStartCalled_, 1);
            ar->writeField(pmName, pmValuesRebased.data(), pmValuesRebased.size());
            ar->writeField(pmName + "_timeStamps", pmTimeStampsRebased.data(), pmTimeStampsRebased.size());
            ar->closeStep();
            pmValues.clear();
            pmTimeStamps.clear();
        }
        numStartCalled_ = 0;
    }

private:
    void readPm()
    {
        for (auto& counter : pmCounters)
        {
            auto   filePath = std::get<1>(counter);
            PmType joules = 0, timeStamp_ms = 0;
            if (get<4>(counter)) { readPmCounter(filePath.c_str(), &joules, &timeStamp_ms); }
            std::get<2>(counter).push_back(joules);
            std::get<3>(counter).push_back(timeStamp_ms);
        }
    }

    int rank_, numRanksPerNode_{0}, numStartCalled_{0};

    //                     name         filepath      counter reading      time-stamp reading  enabled
    std::vector<std::tuple<std::string, std::string, std::vector<PmType>, std::vector<PmType>, bool>> pmCounters;
};

} // namespace sphexa
