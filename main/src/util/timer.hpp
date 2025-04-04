#pragma once

#include <chrono>
#include <map>

#if defined(USE_PROFILING_NVTX) || defined(USE_PROFILING_SCOREP)

#ifdef USE_PROFILING_NVTX
#include <nvToolsExt.h>
#define MARK_BEGIN(xx) nvtxRangePush(xx);
#define MARK_END nvtxRangePop();
#endif

#ifdef USE_PROFILING_SCOREP
#include "scorep/SCOREP_User.h"
#define MARK_BEGIN(xx)                                                                                                 \
    {                                                                                                                  \
        SCOREP_USER_REGION(xx, SCOREP_USER_REGION_TYPE_COMMON)
#define MARK_END }
#endif

#else
#define MARK_BEGIN(xx)
#define MARK_END
#endif

namespace sphexa
{

class Timer
{
    typedef std::chrono::high_resolution_clock Clock;
    typedef std::chrono::duration<float>       Time;

public:
    Timer(std::ostream& out)
        : out(out)
    {
    }

    void start()
    {
        numStartCalled++;
        tstart = tlast = Clock::now();
    }

    void step(const std::string& name)
    {
        auto now = Clock::now();
        stepTimes.push_back(stepDuration(now));
        if (!name.empty()) { out << "# " << name << ": " << stepTimes.back() << "s" << std::endl; }
        tlast = now;
    }

    void logStatistics(const std::string& name, float value)
    {
        perfStats[name].push_back(value);
    }

    //! @brief time elapsed between tstart and last call of step()
    [[nodiscard]] float sumOfSteps() const { return std::chrono::duration_cast<Time>(tlast - tstart).count(); }

    //! @brief time elapsed between tstart and now
    [[nodiscard]] float elapsed() const { return std::chrono::duration_cast<Time>(Clock::now() - tstart).count(); }
    [[nodiscard]] float getLastStepTime() const { return stepTimes.back(); }

    template<class Archive>
    void writeTimings(Archive* ar, const std::string& outFile)
    {
        ar->addStep(0, stepTimes.size(), outFile + ar->suffix());
        int numRanks = ar->numRanks();
        ar->stepAttribute("numRanks", &numRanks, 1);
        ar->stepAttribute("numIterations", &numStartCalled, 1);
        ar->writeField("timings", stepTimes.data(), stepTimes.size());
        ar->closeStep();

        for (const auto& item : perfStats)
        {
            if (item.second.empty()) { continue; }
            ar->addStep(0, item.second.size(), outFile + ar->suffix());
            ar->stepAttribute("numRanks", &numRanks, 1);
            ar->stepAttribute("numIterations", &numStartCalled, 1);
            ar->writeField(item.first, item.second.data(), item.second.size());
            ar->closeStep();
        }

        numStartCalled = 0;
        stepTimes.clear();
        for (auto& item : perfStats)
        {
            item.second.clear();
        }
    }

private:
    float stepDuration(auto now) { return std::chrono::duration_cast<Time>(now - tlast).count(); }

    std::ostream&                  out;
    std::chrono::time_point<Clock> tstart, tlast;
    std::vector<float>             stepTimes;
    int                            numStartCalled{0};

    std::map<std::string, std::vector<float>> perfStats;
};

} // namespace sphexa
