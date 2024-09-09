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
bool strIsIntegral(const std::string& str);

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
bool isExtraOutputStep(size_t step, double t1, double t2, const std::vector<std::string>& extraOutputs);

/*! @brief Evaluate whether the current step should be output (to file) according to time frequency
 *
 * @param t1            simulation time at beginning of current step
 * @param t2            simulation time at end of current step
 * @param frequencyStr  frequency time to output the simulation as string
 * @return              true if the interval [t1, t2] contains a positive integer multiple of the output frequency
 */
bool isOutputTime(double t1, double t2, const std::string& frequencyStr);

/*! @brief Evaluate whether the current step should be output (to file) according to iteration frequency
 *
 * @param step          simulation step number
 * @param frequencyStr  iteration frequency to output the simulation as string
 * @return              true if the step is an integral multiple of the output frequency
 */
bool isOutputStep(size_t step, const std::string& frequencyStr);

std::string strBeforeSign(const std::string& str, const std::string& sign);

//! @brief If the input string ends with @p sign followed by an integer, return the integer, otherwise return -1
std::string strAfterSign(const std::string& str, const std::string& sign);

//! @brief If the input string ends with @p sign followed by an integer, return the integer, otherwise return -1
int numberAfterSign(const std::string& str, const std::string& sign);

std::string removeModifiers(const std::string& initCond); /*{ return strBeforeSign(strBeforeSign(initCond, ":"), ","); }*/

} // namespace sphexa
