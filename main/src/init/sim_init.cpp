/*
 * MIT License
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *               2024 University of Basel
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
 * @brief Translation unit for the simulation initializer library
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "cstone/primitives/accel_switch.hpp"

#include "isim_init.hpp"

#include "evrard_init.hpp"
#include "file_init.hpp"
#include "gresho_chan.hpp"
#include "isobaric_cube_init.hpp"
#include "kelvin_helmholtz_init.hpp"
#include "noh_init.hpp"
#include "sedov_init.hpp"
#include "turbulence_init.hpp"
#include "wind_shock_init.hpp"
#ifdef SPH_EXA_HAVE_GRACKLE
#include "evrard_cooling_init.hpp"
#endif

namespace sphexa
{

template<class Dataset>
std::unique_ptr<ISimInitializer<Dataset>>
SimInitializers<Dataset>::makeEvrard(std::string glassBlock, std::string settingsFile, IFileReader* reader)
{
    return std::make_unique<EvrardGlassSphere<Dataset>>(glassBlock, settingsFile, reader);
}

#ifdef SPH_EXA_HAVE_GRACKLE
template<class Dataset>
std::unique_ptr<ISimInitializer<Dataset>>
SimInitializers<Dataset>::makeEvrardCooling(std::string glassBlock, std::string settingsFile, IFileReader* reader)
{
    return std::make_unique<EvrardGlassSphereCooling<Dataset>>(glassBlock, settingsFile, reader);
}
#else
template<class Dataset>
std::unique_ptr<ISimInitializer<Dataset>>
SimInitializers<Dataset>::makeEvrardCooling(std::string /*glass*/, std::string /*settingsFile*/, IFileReader*)
{
    throw std::runtime_error("Missing GRACKLE build option for evrard-cooling\n");
    return nullptr;
}
#endif

template<class Dataset>
std::unique_ptr<ISimInitializer<Dataset>> SimInitializers<Dataset>::makeFile(std::string testCase, int initStep,
                                                                             IFileReader* reader)
{
    return std::make_unique<FileInit<Dataset>>(testCase, initStep, reader);
}

template<class Dataset>
std::unique_ptr<ISimInitializer<Dataset>> SimInitializers<Dataset>::makeFileSplit(std::string testCase, int numSplits,
                                                                                  IFileReader* reader)
{
    return std::make_unique<FileSplitInit<Dataset>>(testCase, numSplits, reader);
}

template<class Dataset>
std::unique_ptr<ISimInitializer<Dataset>>
SimInitializers<Dataset>::makeGreshoChan(std::string glassBlock, std::string settingsFile, IFileReader* reader)
{
    return std::make_unique<GreshoChan<Dataset>>(glassBlock, settingsFile, reader);
}

template<class Dataset>
std::unique_ptr<ISimInitializer<Dataset>>
SimInitializers<Dataset>::makeKelvinHelmholtz(std::string glassBlock, std::string settingsFile, IFileReader* reader)
{
    return std::make_unique<KelvinHelmholtzGlass<Dataset>>(glassBlock, settingsFile, reader);
}

template<class Dataset>
std::unique_ptr<ISimInitializer<Dataset>>
SimInitializers<Dataset>::makeIsobaricCube(std::string glassBlock, std::string settingsFile, IFileReader* reader)
{
    return std::make_unique<IsobaricCubeGlass<Dataset>>(glassBlock, settingsFile, reader);
}

template<class Dataset>
std::unique_ptr<ISimInitializer<Dataset>>
SimInitializers<Dataset>::makeNoh(std::string glassBlock, std::string settingsFile, IFileReader* reader)
{
    return std::make_unique<NohGlassSphere<Dataset>>(glassBlock, settingsFile, reader);
}

template<class Dataset>
std::unique_ptr<ISimInitializer<Dataset>>
SimInitializers<Dataset>::makeSedovGlass(std::string glassBlock, std::string settingsFile, IFileReader* reader)
{
    return std::make_unique<SedovGlass<Dataset>>(glassBlock, settingsFile, reader);
}

template<class Dataset>
std::unique_ptr<ISimInitializer<Dataset>> SimInitializers<Dataset>::makeSedovGrid()
{
    return std::make_unique<SedovGrid<Dataset>>();
}

template<class Dataset>
std::unique_ptr<ISimInitializer<Dataset>>
SimInitializers<Dataset>::makeTurbulence(std::string glassBlock, std::string settingsFile, IFileReader* reader)
{
    return std::make_unique<TurbulenceGlass<Dataset>>(glassBlock, settingsFile, reader);
}

template<class Dataset>
std::unique_ptr<ISimInitializer<Dataset>>
SimInitializers<Dataset>::makeWindShock(std::string glassBlock, std::string settingsFile, IFileReader* reader)
{
    return std::make_unique<WindShockGlass<Dataset>>(glassBlock, settingsFile, reader);
}

#ifdef USE_CUDA
template struct SimInitializers<SimulationData<cstone::GpuTag>>;
#else
template struct SimInitializers<SimulationData<cstone::CpuTag>>;
#endif

} // namespace sphexa
