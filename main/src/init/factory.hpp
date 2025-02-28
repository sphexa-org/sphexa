/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
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
 * @brief Select a simulation initializer based on user input choice
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <filesystem>
#include <map>

#include "io/arg_parser.hpp"
#include "isim_init.hpp"

namespace sphexa
{

template<class Dataset>
std::unique_ptr<ISimInitializer<Dataset>> initializerFactory(std::string testCase, std::string glassBlock,
                                                             IFileReader* reader)
{
    std::string testNamedBase = strBeforeSign(testCase, ":");
    std::string settingsFile  = strAfterSign(testCase, ":");

    if (testNamedBase == "sedov")
    {
        if (glassBlock.empty()) { return SimInitializers<Dataset>::makeSedovGrid(); }
        else { return SimInitializers<Dataset>::makeSedovGlass(glassBlock, settingsFile, reader); }
    }
    if (testNamedBase == "noh")
    {
        if (glassBlock.empty()) { throw std::runtime_error("need a valid glass block for Noh implosion\n"); }
        return SimInitializers<Dataset>::makeNoh(glassBlock, settingsFile, reader);
    }
    if (testNamedBase == "gresho-chan")
    {
        if (glassBlock.empty()) { throw std::runtime_error("need a valid glass block for Gresho-Chan\n"); }
        return SimInitializers<Dataset>::makeGreshoChan(glassBlock, settingsFile, reader);
    }
    if (testNamedBase == "isobaric-cube")
    {
        if (glassBlock.empty()) { throw std::runtime_error("need a valid glass block for isobaric cube\n"); }
        return SimInitializers<Dataset>::makeIsobaricCube(glassBlock, settingsFile, reader);
    }
    if (testNamedBase == "wind-shock")
    {
        if (glassBlock.empty()) { throw std::runtime_error("need a valid glass block for Wind shock\n"); }
        return SimInitializers<Dataset>::makeWindShock(glassBlock, settingsFile, reader);
    }
    if (testNamedBase == "evrard")
    {
        if (glassBlock.empty()) { throw std::runtime_error("need a valid glass block for evrard\n"); }
        return SimInitializers<Dataset>::makeEvrard(glassBlock, settingsFile, reader);
    }
    if (testNamedBase == "turbulence")
    {
        if (glassBlock.empty()) { throw std::runtime_error("need a valid glass block for turbulence test\n"); }
        else { return SimInitializers<Dataset>::makeTurbulence(glassBlock, settingsFile, reader); }
    }
    if (testNamedBase == "kelvin-helmholtz")
    {
        if (glassBlock.empty()) { throw std::runtime_error("need a valid glass block for Kelvin-Helmholtz test\n"); }
        else { return SimInitializers<Dataset>::makeKelvinHelmholtz(glassBlock, settingsFile, reader); }
    }
    if (testNamedBase == "evrard-cooling")
    {
        if (glassBlock.empty()) { throw std::runtime_error("need a valid glass block for evrard-cooling\n"); }
        return SimInitializers<Dataset>::makeEvrardCooling(glassBlock, settingsFile, reader);
    }
    if (std::filesystem::exists(strBeforeSign(testCase, ":")))
    {
        return SimInitializers<Dataset>::makeFile(strBeforeSign(testCase, ":"), numberAfterSign(testCase, ":"), reader);
    }
    if (std::filesystem::exists(strBeforeSign(testCase, ",")))
    {
        return SimInitializers<Dataset>::makeFileSplit(strBeforeSign(testCase, ","), numberAfterSign(testCase, ","),
                                                       reader);
    }

    auto msg = "supplied value of --init " + (testCase.empty() ? "[empty string]" : "(\"" + testCase + "\")") +
               " is not a usable file or supported test case\n";
    if (reader->rank() == 0) { std::cout << msg; }
    MPI_Abort(MPI_COMM_WORLD, 1);

    return std::unique_ptr<ISimInitializer<Dataset>>{};
}

} // namespace sphexa
