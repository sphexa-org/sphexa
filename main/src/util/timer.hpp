#pragma once

#include <chrono>
#include <iostream>
#include <functional>
#include "profiler.hpp"

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
public:
    typedef std::chrono::high_resolution_clock Clock;
    typedef std::chrono::time_point<Clock>     TimePoint;
    typedef std::chrono::duration<float>       Time;

    Timer(std::ostream& out)
        : out(out)
    {
    }

    float duration() { return std::chrono::duration_cast<Time>(tstop - tstart).count(); }

    float getSimDuration() { return std::chrono::duration_cast<Time>(Clock::now() - tstart).count(); }

    void start() { tstart = tstop = tlast = Clock::now(); }

    void stop() { tstop = Clock::now(); }

    void step()
    {
        stop();
        tlast = tstop;
    }

    float stepDuration() { return std::chrono::duration_cast<Time>(Clock::now() - tlast).count(); }

    void step(const std::string& name)
    {
        stop();
        out << "# " << name << ": " << std::chrono::duration_cast<Time>(tstop - tlast).count() << "s" << std::endl;
        tlast = tstop;
    }

private:
    std::ostream& out;
    TimePoint     tstart, tstop, tlast;
};

class ProfilingTimer : public Timer
{
public:
    ProfilingTimer(std::ostream& out, int rank)
        : Timer(out)
        , rank(rank)
        , profiler(rank)
    {
    }

    void start()
    {
        Timer::start();
        profiler.startEnergyMeasurement();
    }

    void step(const std::string& name)
    {
        profiler.saveFunctionTimings(stepDuration());
        if (rank == 0)
            Timer::step(name);
        else
            Timer::step();
    }

    void stop()
    {
        Timer::stop();
        profiler.saveTimestep(duration());
    }

    void printProfilingInfo(size_t timesteps) { profiler.printProfilingInfoHDF5(timesteps); }

private:
    int      rank;
    Profiler profiler;
};

} // namespace sphexa
