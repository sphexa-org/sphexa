#include <algorithm>

#include "arg_parser.hpp"

namespace sphexa
{

ArgParser::ArgParser(int argc, const char** argv)
    : begin(argv)
    , end(argv + argc)
{
}

std::vector<std::string> ArgParser::getCommaList(const std::string& option) const
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

bool ArgParser::exists(const std::string& option) const { return std::find(begin, end, option) != end; }

bool strIsIntegral(const std::string& str)
{
    char* ptr;
    std::strtol(str.c_str(), &ptr, 10);
    return (*ptr) == '\0' && !str.empty();
}

bool isExtraOutputStep(size_t step, double t1, double t2, const std::vector<std::string>& extraOutputs)
{
    auto matchStepOrTime = [step, t1, t2](const std::string& token)
    {
        double time       = std::stod(token);
        bool   isIntegral = strIsIntegral(token);
        return (isIntegral && std::stoul(token) == step) || (!isIntegral && t1 <= time && time < t2);
    };

    return std::any_of(extraOutputs.begin(), extraOutputs.end(), matchStepOrTime);
}

bool isOutputTime(double t1, double t2, const std::string& frequencyStr)
{
    double frequency = std::stod(frequencyStr);
    if (strIsIntegral(frequencyStr) || frequency == 0.0) { return false; }

    double closestMultiple = int(t2 / frequency) * frequency;
    return t2 > frequency && t1 <= closestMultiple && closestMultiple < t2;
}

bool isOutputStep(size_t step, const std::string& frequencyStr)
{
    int frequency = std::stoi(frequencyStr);
    return strIsIntegral(frequencyStr) && frequency != 0 && (step % frequency == 0);
}

std::string strBeforeSign(const std::string& str, const std::string& sign)
{
    auto commaPos = str.find_first_of(sign);
    return str.substr(0, commaPos);
}

std::string strAfterSign(const std::string& str, const std::string& sign)
{
    auto commaPos = str.find_first_of(sign);
    if (commaPos == std::string::npos) { return {}; }

    return str.substr(commaPos + sign.size());
}

int numberAfterSign(const std::string& str, const std::string& sign)
{
    std::string afterComma = strAfterSign(str, sign);
    return strIsIntegral(afterComma) ? std::stoi(afterComma) : -1;
}

std::string removeModifiers(const std::string& initCond) { return strBeforeSign(strBeforeSign(initCond, ":"), ","); }

} // namespace sphexa
