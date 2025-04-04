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
 * @author Christopher Bignamini <christopher.bignamini@gmail.com>
 */

#pragma once

#include <filesystem>
#include <map>

#include "cstone/sfc/box.hpp"
#include "io/ifile_io.hpp"
#include "io/id_tag_utils.hpp"
#include "sphexa/simulation_data.hpp"

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

    const IdSubsets& subsets() const  { return idSubsets_; }

    virtual ~ISimInitializer() = default;

protected:

    // TODO: Base::initSubsets(settings_, &d);// I have to pass a ref to the entire dataset because I could need the coordinates, if selection is geometrical
    /*! @brief Id tagging
    *
    * @param[in]  settings    spherical volume definition
    * @param[in]  printLog    activate logging
    * @param[out] d           time step at which selection is done
    */
    void initSubsets(const InitSettings& settings, bool printLog, Dataset::HydroData& particlesData, int initStep = 0) const
    {
        auto idSelectionSphereRadius = settings.find("id_selection_sphere_radius");
        if(idSelectionSphereRadius != settings.end()) {
            if(printLog) { std::cout << "Execution of id subset tagging in sphere" << std::endl; }
            initSubsets(IdSelectionSphere{std::get<ScalarValue>(settings.at("id_selection_sphere_center_x").getValue()),
                std::get<ScalarValue>(settings.at("id_selection_sphere_center_y").getValue()), std::get<ScalarValue>(settings.at("id_selection_sphere_center_z").getValue()),
                std::get<ScalarValue>(idSelectionSphereRadius->second.getValue())}, initStep,
                particlesData.x, particlesData.y, particlesData.z, particlesData.id);
        }
        auto idSelectionList = settings.find("id_selection_list");
        if(idSelectionList != settings.end()) {
            if(printLog) { std::cout << "Execution of id subset tagging in list" << std::endl; }
            if(idSelectionList->second.isVector()) {
                initSubsets(std::get<VectorValue>(idSelectionList->second.getValue()),
                initStep, particlesData.id);
            }
            else {
                initSubsets(IdVectorType{IdType(std::get<ScalarValue>(idSelectionList->second.getValue()))},
                initStep, particlesData.id);
            }
        }
        // TODO: if we only want the subset selection attributes in the subset file, I can delete them from settings_ here, after subset initialization.
        // If that is the case, do not forgot to call Base::resetConstants(settings_): maybe it's not needed in the simulation but it will keep the settings_
        // consistent along the inheritance hierachy
    }

    /*! @brief Id tagging in spherical volume
    *
    * @param[in]  selSphereData    spherical volume definition
    * @param[in]  initStep         time step at which selection is done
    * @param[in]  x                x coordinates
    * @param[in]  y                y coordinates
    * @param[in]  z                z coordinates
    * @param[out] ids              id list from hydro data
    */
    void initSubsets(const IdSelectionSphere& selSphereData, int initStep, const std::vector<CoordinateType>& x,
        const std::vector<CoordinateType>& y, const std::vector<CoordinateType>& z, IdVectorType& ids) const
    {
        tagIdsInSphere(ids, x, y, z, 0, ids.size(), selSphereData);

        idSubsets_["id_selection_sphere"] = IdSelectionSettings{selSphereData, initStep};
    }

    /*! @brief Id tagging from list
    *
    * @param[in]  selectedIds    ids to be tagged
    * @param[in]  initStep       time step at which selection is done
    * @param[out] ids            id list from hydro data
    */
    void initSubsets(const IdVectorType& selectedIds, int initStep, IdVectorType& ids) const
    {
        tagIdsInList(ids, 0, ids.size(), selectedIds);

        idSubsets_["id_selection_list"] = IdSelectionSettings{selectedIds, initStep};
    }

    mutable IdSubsets idSubsets_;

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
