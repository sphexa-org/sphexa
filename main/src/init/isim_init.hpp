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

    ISimInitializer(std::string settingsFile)
        : settingsFile_(std::move(settingsFile))
    {
    }

    virtual cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t, Dataset& d,
                                                         IFileReader*) const = 0;

    virtual const InitSettings& constants() const = 0;

    virtual ~ISimInitializer() = default;

    const IdTaggingSetup& taggingSetup() const { return taggingSetup_; }

protected:
    /*! @brief Id tagging initialization and execution
     *
     * @param[in]     reader         parameter file reader
     * @param[in]     printLog       activate logging
     * @param[inout]  particlesData  particle data to perform selection on
     * @param[in]     initStep       time step at which selection is done
     */
    // TODO: I have to pass a ref to the entire dataset because I could need the coordinates, if selection is geometrical
    void runTagging(IFileReader* reader, bool printLog, Dataset::HydroData& particlesData) const
    {
        taggingSetup_.selSpheres.clear();
        taggingSetup_.sphereGroupIds.clear();
        taggingSetup_.selList.clear();
        taggingSetup_.selListGroupIds.clear();

        if (not settingsFile_.empty())
        {
            readFileTaggingAttributes(settingsFile_, reader, taggingSetup_.selSpheres, taggingSetup_.sphereGroupIds,
                taggingSetup_.selList, taggingSetup_.selListGroupIds);
            std::cout<<"Tagging setup read from file: "<<settingsFile_<<std::endl;
            std::cout<<" - Number of id selection spheres: "<<taggingSetup_.selSpheres.size()<<std::endl;
            idTaggingSetupCheck(taggingSetup_.selSpheres, taggingSetup_.sphereGroupIds, taggingSetup_.selList, 
                taggingSetup_.selListGroupIds, printLog);

            if(taggingSetup_.selList.size() > 0)
            {
                if (printLog)
                {
                    std::cout<<"Tagging particles in id lists"<<std::endl;
                }
                tagIdsInList(particlesData.id, 0, particlesData.id.size(), taggingSetup_.selList, taggingSetup_.selListGroupIds);
            }

            if(taggingSetup_.selSpheres.size() > 0)
            {
                if (printLog)
                {
                    std::cout<<"Tagging particles in spheres"<<std::endl;
                }
                tagIdsInSphere(particlesData.id, particlesData.x, particlesData.y, particlesData.z,
                    0, particlesData.id.size(), taggingSetup_.selSpheres, taggingSetup_.sphereGroupIds);
            }
        }
    };

    // May be empty, if no settings file is provided (e.g., restart from dump)
    std::string settingsFile_;
    // TODO: is it necessary to store the tagging setup? we only need it in writeSettings and writeTaggingSettings
    mutable IdTaggingSetup taggingSetup_;

};

template<class Dataset>
struct SimInitializers
{
    using InitPtr = std::unique_ptr<ISimInitializer<Dataset>>;

    static InitPtr makeEvrard(std::string glassBlock, std::string settingsFile, IFileReader* reader);
    static InitPtr makeEvrardCooling(std::string glassBlock, std::string settingsFile, IFileReader* reader);
    static InitPtr makeFile(std::string testCase, int initStep, IFileReader* reader);
    // static InitPtr makeFileSplit(std::string testCase, int numsplits, IFileReader* reader);
    static InitPtr makeGreshoChan(std::string glassBlock, std::string settingsFile, IFileReader* reader);
    static InitPtr makeKelvinHelmholtz(std::string glassBlock, std::string settingsFile, IFileReader* reader);
    static InitPtr makeIsobaricCube(std::string glassBlock, std::string settingsFile, IFileReader* reader);
    static InitPtr makeNoh(std::string glassBlock, std::string settingsFile, IFileReader* reader);
    static InitPtr makeSedovGlass(std::string glassBlock, std::string settingsFile, IFileReader* reader);
    static InitPtr makeSedovGrid(std::string settingsFile);
    static InitPtr makeTurbulence(std::string glassBlock, std::string settingsFile, IFileReader* reader);
    static InitPtr makeWindShock(std::string glassBlock, std::string settingsFile, IFileReader* reader);
};

extern template struct SimInitializers<SimulationData<cstone::CpuTag>>;
extern template struct SimInitializers<SimulationData<cstone::GpuTag>>;

} // namespace sphexa
