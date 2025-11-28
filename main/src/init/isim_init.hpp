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
 * @brief Test-case simulation data initialization
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <filesystem>
#include <map>

#include "cstone/sfc/box.hpp"
#include "io/ifile_io.hpp"
#include "io/id_tag_utils.hpp"
#include "io/id_tag_setup.hpp"
#include "sphexa/simulation_data.hpp"

#include "utils.hpp"
#include "settings.hpp"

namespace sphexa
{

template<class Dataset>
class ISimInitializer
{
public:
    virtual cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t, Dataset& d,
                                                         IFileReader*) const = 0;

    virtual const InitSettings& constants() const = 0;

     /*! @brief Id tagging initialization and execution
     *
     * @param[in]     reader         parameter file reader
     * @param[in]     settingsFile   settings file path
     * @param[in]     printLog       activate logging
     * @param[inout]  particlesData  particle data to perform selection on
     * @param[in]     initStep       time step at which selection is done
     */
     // TODO: I have to pass a ref to the entire dataset because I could need the coordinates, if selection is geometrical
     void runTagging(IFileReader* reader, std::string settingsFile, bool printLog, Dataset::HydroData& particlesData, int initStep = 0) const
     {
        if (not settingsFile.empty())
        {
            std::vector<IdSelectionSphere> selSpheres;
            std::vector<unsigned int> sphereGroupIds;
            std::vector<uint64_t> selList;
            std::vector<unsigned int> selListGroupIds;
            readFileTaggingAttributes(settingsFile, reader, selSpheres, sphereGroupIds, selList, selListGroupIds);

            idTaggingSetupCheck(selSpheres, sphereGroupIds, selList, selListGroupIds, printLog);

            if(selList.size() > 0)
            {
                if (printLog)
                {
                    std::cout<<"Tagging particles in id list"<<initStep<<std::endl;
                }
                tagIdsInList(particlesData.id, 0, particlesData.id.size(), selList, selListGroupIds);
            }

            if(selSpheres.size() > 0)
            {
                if (printLog)
                {
                    std::cout<<"Tagging particles in spheres"<<initStep<<std::endl;
                }
                tagIdsInSphere(particlesData.id, particlesData.x, particlesData.y, particlesData.z,
                    0, particlesData.id.size(), selSpheres, sphereGroupIds);
            }
        }
     };

    virtual ~ISimInitializer() = default;
};

template<class Dataset>
struct SimInitializers
{
    using InitPtr = std::unique_ptr<ISimInitializer<Dataset>>;

    static InitPtr makeEvrard(std::string glassBlock, std::string settingsFile, IFileReader* reader);
    static InitPtr makeEvrardCooling(std::string glassBlock, std::string settingsFile, IFileReader* reader);
    static InitPtr makeFile(std::string testCase, int initStep, IFileReader* reader);
    static InitPtr makeFileSplit(std::string testCase, int numsplits, IFileReader* reader);
    static InitPtr makeGreshoChan(std::string glassBlock, std::string settingsFile, IFileReader* reader);
    static InitPtr makeKelvinHelmholtz(std::string glassBlock, std::string settingsFile, IFileReader* reader);
    static InitPtr makeIsobaricCube(std::string glassBlock, std::string settingsFile, IFileReader* reader);
    static InitPtr makeNoh(std::string glassBlock, std::string settingsFile, IFileReader* reader);
    static InitPtr makeSedovGlass(std::string glassBlock, std::string settingsFile, IFileReader* reader);
    static InitPtr makeSedovGrid();
    static InitPtr makeTurbulence(std::string glassBlock, std::string settingsFile, IFileReader* reader);
    static InitPtr makeWindShock(std::string glassBlock, std::string settingsFile, IFileReader* reader);
};

extern template struct SimInitializers<SimulationData<cstone::CpuTag>>;
extern template struct SimInitializers<SimulationData<cstone::GpuTag>>;

} // namespace sphexa
