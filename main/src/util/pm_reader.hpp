#pragma once

#include <cstdio>
#include <filesystem>
#include <functional>
#include <vector>

namespace sphexa
{

int readPmCounter(const char* fname, unsigned long long* joules, unsigned long long* ms);

class PmReader
{
    using PmType = unsigned long long;

public:
    explicit PmReader(int rank)
        : rank_(rank)
    {
    }

    void addCounters(const std::string& pmRoot, int numRanksPerNode_)
    {
        numRanksPerNode   = numRanksPerNode_;
        using CounterDesc = std::tuple<std::string, std::string, std::function<bool(int, int)>>;
        std::vector<CounterDesc> countersToAdd{
            {"node", pmRoot + "/energy", [](int r, int numRanksPerNode) { return r % numRanksPerNode == 0; }},
            {"acc", pmRoot + "/accel" + std::to_string(rank_ % numRanksPerNode) + "_energy",
             [](int, int) { return true; }}};

        for (auto& c : countersToAdd)
        {
            bool enable = std::filesystem::exists(get<1>(c)) && get<2>(c)(rank_, numRanksPerNode);
            pmCounters.emplace_back(get<0>(c), get<1>(c), std::vector<PmType>{}, std::vector<PmType>{}, enable);
        }
    }

    void start()
    {
        numStartCalled++;
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
            ar->stepAttribute("numRanksPerNode", &numRanks, 1);
            ar->stepAttribute("numIterations", &numStartCalled, 1);
            ar->writeField(pmName, pmValuesRebased.data(), pmValuesRebased.size());
            ar->writeField(pmName + "_timeStamps", pmTimeStampsRebased.data(), pmTimeStampsRebased.size());
            ar->closeStep();
            pmValues.clear();
            pmTimeStamps.clear();
        }
        numStartCalled = 0;
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

    int rank_, numRanksPerNode{0}, numStartCalled{0};

    //                     name         filepath      counter reading      time-stamp reading  enabled
    std::vector<std::tuple<std::string, std::string, std::vector<PmType>, std::vector<PmType>, bool>> pmCounters;
};

} // namespace sphexa
