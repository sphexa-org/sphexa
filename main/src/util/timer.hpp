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
        iteration++;
        numStartCalled++;
        subStep = 0;
        tstart = tlast = Clock::now();
    }

    void step(const std::string& name, bool print = true)
    {
        auto now = Clock::now();
        stepTimes.push_back(std::chrono::duration_cast<Time>(now - tlast).count());
        if (print) { out << "# " << name << ": " << stepTimes.back() << "s" << std::endl; }
        tlast = now;

        subStep++;
        if (subStep > int(stepTimeNames.size())) { stepTimeNames.push_back(name); }
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
        int numRanks           = ar->numRanks();
        int lastIterationSaved = iteration - numStartCalled;
        int numIterations      = numStartCalled;
        int numSubSteps        = int(stepTimeNames.size());

        ar->addStep(0, stepTimes.size(), outFile + ar->suffix());
        ar->stepAttribute("name", "timings");
        ar->stepAttribute("iteration", &lastIterationSaved, 1);
        ar->stepAttribute("numRanks", &numRanks, 1);
        ar->stepAttribute("numIterations", &numIterations, 1);
        ar->stepAttribute("numDataPerIteration", &numSubSteps, 1);
        for (std::size_t i = 0; i < stepTimeNames.size(); ++i)
        {
            ar->stepAttribute(std::string("stepName") + std::to_string(i), stepTimeNames[i]);
        }
        ar->writeField("timings", stepTimes.data(), stepTimes.size());
        ar->closeStep();
        stepTimes.clear();

        for (auto& item : perfStats)
        {
            auto writeField = [ar, outFile, numRanks, name = item.first, it = lastIterationSaved,
                               numSteps = numStartCalled](auto& field)
            {
                if (field.empty()) { return; }
                ar->addStep(0, field.size(), outFile + ar->suffix());
                ar->stepAttribute("name", name);
                ar->stepAttribute("iteration", &it, 1);
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
    std::ostream&                  out;
    std::chrono::time_point<Clock> tstart, tlast;
    int                            subStep{0}, iteration{0}, numStartCalled{0};

    std::vector<float>       stepTimes;
    std::vector<std::string> stepTimeNames;

    using SupportedTypes   = util::TypeList<float, double, uint32_t, uint64_t, int32_t, int64_t>;
    using SupportedVariant = util::Reduce<std::variant, util::Map<std::vector, SupportedTypes>>;

    std::map<std::string, SupportedVariant> perfStats;
};

} // namespace sphexa
