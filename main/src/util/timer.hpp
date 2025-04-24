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

    template<class T>
    void logStatistics(const std::string& name, T value)
    {
        if (not std::holds_alternative<std::vector<T>>(perfStats[name])) { perfStats[name] = std::vector<T>{}; }
        std::get<std::vector<T>>(perfStats[name]).push_back(value);
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
        stepTimes.clear();

        for (auto& item : perfStats)
        {
            auto writeField = [ar, outFile, numRanks, name = item.first, numSteps = numStartCalled](auto& field)
            {
                if (field.empty()) { return; }
                ar->addStep(0, field.size(), outFile + ar->suffix());
                ar->stepAttribute("numRanks", &numRanks, 1);
                ar->stepAttribute("numIterations", &numSteps, 1);
                ar->writeField(name, field.data(), field.size());
                ar->closeStep();
                field.clear();
            };
            std::visit(writeField, item.second);
        }

        numStartCalled = 0;
    }

private:
    float stepDuration(auto now) { return std::chrono::duration_cast<Time>(now - tlast).count(); }

    std::ostream&                  out;
    std::chrono::time_point<Clock> tstart, tlast;
    std::vector<float>             stepTimes;
    int                            numStartCalled{0};

    using SupportedTypes   = util::TypeList<float, double, uint32_t, uint64_t, int32_t, int64_t>;
    using SupportedVariant = util::Reduce<std::variant, util::Map<std::vector, SupportedTypes>>;

    std::map<std::string, SupportedVariant> perfStats;
};

} // namespace sphexa
