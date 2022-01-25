#pragma once

#include <algorithm>

namespace sphexa
{

class ArgParser
{
public:
    ArgParser(int argc, char **argv)
        : begin(argv)
        , end(argv + argc)
    {
    }

    std::string getString(const std::string &option, const std::string def = "") const
    {
        char **itr = std::find(begin, end, option);
        if (itr != end && ++itr != end) return std::string(*itr);
        return def;
    }

    int getInt(const std::string &option, const int def = 0) const
    {
        char **itr = std::find(begin, end, option);
        if (itr != end && ++itr != end) return (int)std::stof(*itr);
        return def;
    }

    double getDouble(const std::string &option, const double def = 0.) const
    {
        char **itr = find(begin, end, option);
        if (itr != end && ++itr != end) return std::stod(*itr);
        return def;
    }

    bool exists(const std::string &option) const { return std::find(begin, end, option) != end; }

private:
    char **begin, **end;
};

} // namespace sphexa
