#pragma once

#include <algorithm>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>
#include <cassert>

namespace sphexa
{

//! @brief returns true if all characters of @p str together represent a valid integral number
inline bool strIsIntegral(const std::string& str)
{
    char* ptr;
    std::strtol(str.c_str(), &ptr, 10);
    return (*ptr) == '\0' && !str.empty();
}

class ArgParser
{
public:
    ArgParser(int argc, const char** argv)
        : begin(argv)
        , end(argv + argc)
    {
    }

    //! @brief look for @p option in the supplied cmd-line arguments and convert to T if found
    template<class T = std::string>
    T get(const std::string& option, T def = T{}) const
    {
        auto itr = std::find(begin, end, option);
        if (itr != end && ++itr != end && **itr != '-')
        {
            if constexpr (std::is_arithmetic_v<T>)
            {
                return strIsIntegral(*itr) ? T(std::stoi(*itr)) : T(std::stod(*itr));
            }
            else { return std::string(*itr); }
        }
        return def;
    }

    //! @brief parse a comma-separated list
    std::vector<std::string> getCommaList(const std::string& option) const
    {
        std::string listWithCommas = get(option);

        std::replace(listWithCommas.begin(), listWithCommas.end(), ',', ' ');

        std::vector<std::string> list;
        std::stringstream        ss(listWithCommas);
        std::string              field;
        while (ss >> field)
        {
            list.push_back(field);
        }

        return list;
    }

    bool exists(const std::string& option) const { return std::find(begin, end, option) != end; }

private:
    const char** begin;
    const char** end;
};

/*! @brief Evaluate whether the current step and simulation time should be output (to file)
 *
 * @param step          current simulation step
 * @param t1            simulation time at beginning of current step
 * @param t2            simulation time at end of current step
 * @param extraOutputs  list of strings of integral and/or floating point numbers
 * @return              true if @p step matches any integral numbers in @p extraOutput or
 *                      if any floating point number therein falls into the interval @p [t1, t2)
 */
inline bool isExtraOutputStep(size_t step, double t1, double t2, const std::vector<std::string>& extraOutputs)
{
    auto matchStepOrTime = [step, t1, t2](const std::string& token)
    {
        double time       = std::stod(token);
        bool   isIntegral = strIsIntegral(token);
        return (isIntegral && std::stoul(token) == step) || (!isIntegral && t1 <= time && time < t2);
    };

    return std::any_of(extraOutputs.begin(), extraOutputs.end(), matchStepOrTime);
}

/*! @brief Evaluate whether the current step should be output (to file) according to time frequency
 *
 * @param t1            simulation time at beginning of current step
 * @param t2            simulation time at end of current step
 * @param frequencyStr  frequency time to output the simulation as string
 * @return              true if the interval [t1, t2] contains a positive integer multiple of the output frequency
 */
inline bool isOutputTime(double t1, double t2, const std::string& frequencyStr)
{
    double frequency = std::stod(frequencyStr);
    if (strIsIntegral(frequencyStr) || frequency == 0.0) { return false; }

    double closestMultiple = int(t2 / frequency) * frequency;
    return t2 > frequency && t1 <= closestMultiple && closestMultiple < t2;
}

/*! @brief Evaluate whether the current step should be output (to file) according to iteration frequency
 *
 * @param step          simulation step number
 * @param frequencyStr  iteration frequency to output the simulation as string
 * @return              true if the step is an integral multiple of the output frequency
 */
inline bool isOutputStep(size_t step, const std::string& frequencyStr)
{
    int frequency = std::stoi(frequencyStr);
    return strIsIntegral(frequencyStr) && frequency != 0 && (step % frequency == 0);
}

inline std::string strBeforeSign(const std::string& str, const std::string& sign)
{
    auto commaPos = str.find_first_of(sign);
    return str.substr(0, commaPos);
}

//! @brief If the input string ends with @p sign followed by an integer, return the integer, otherwise return -1
inline std::string strAfterSign(const std::string& str, const std::string& sign)
{
    auto commaPos = str.find_first_of(sign);
    if (commaPos == std::string::npos) { return {}; }

    return str.substr(commaPos + sign.size());
}

//! @brief If the input string ends with @p sign followed by an integer, return the integer, otherwise return -1
inline int numberAfterSign(const std::string& str, const std::string& sign)
{
    std::string afterComma = strAfterSign(str, sign);
    return strIsIntegral(afterComma) ? std::stoi(afterComma) : -1;
}

inline std::string removeModifiers(const std::string& initCond) { return strBeforeSign(strBeforeSign(initCond, ":"), ","); }

} // namespace sphexa
