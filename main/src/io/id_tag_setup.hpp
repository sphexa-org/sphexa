/*
 * MIT License
 *
 * Copyright (c) 2025 CSCS, ETH Zurich, University of Basel, University of Zurich
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
 * @brief Id tagging setup utilities
 *
 * @author Christopher Bignamini <christopher.bignamini@gmail.com>
 */

#pragma once

#include <vector>

#include "ifile_io.hpp"
#include "id_tag_utils.hpp"

namespace sphexa
{
    /*! @brief List of supported id tagging parameters in the settings file
     */
    inline constexpr std::array<std::string_view, 8> taggingAttributes = {
        "o_subset",
        "w_subset",
        "wextra_subset",
        "f_subset",
        "id_selection_spheres",
        "id_selection_spheres_group_ids",
        "id_selection_list",
        "id_selection_list_group_ids"
    };

    /*! @brief Id tagging output parameters
     */
    struct IdTaggingOutputSetup
    {
        // Tagged id output file name
        std::string outFile;
        // Write frequency (iterations or time)
        std::string writeFreqStr;
        // Output fields
        std::vector<std::string> outputFields;
        // Extra writes steps/times
        std::vector<std::string> writeExtra;
    };
    
    /*! @brief Id tagging setup parameters
     */
    struct IdTaggingSetup
    {
        std::vector<IdSelectionSphere> selSpheres;
        std::vector<unsigned int> sphereGroupIds;
        std::vector<uint64_t> selList;
        std::vector<unsigned int> selListGroupIds;
    };

    // /*! @brief Id tagging setup checks and default groups assignment
    //  *
    //  * @param[inout]  selSpheres        sphere selection data
    //  * @param[inout]  sphereGroupIds  group ids for sphere selections
    //  * @param[inout]  selList           list selection data
    //  * @param[inout]  selListGroupIds    group ids for list selections
    //  * @param[in]     printLog        flag to enable printing of setup summary
    //  */
    void idTaggingSetupCheck(std::vector<IdSelectionSphere>& selSpheres, std::vector<unsigned int>& sphereGroupIds, 
                             std::vector<uint64_t>& selList, std::vector<unsigned int>& selListGroupIds, bool printLog);

    // /*! @brief Id tagging groups assignment
    //  *
    //  * @param[in]     selSpheresNum     number of spheres
    //  * @param[inout]  sphere_group_ids  group ids for sphere selections
    //  * @param[in]     selListSize       id selection list size
    //  * @param[inout]  list_group_ids    group ids for list selections
    //  */
    void idTaggingGroupsInit(unsigned int selSpheresNum, std::vector<unsigned int>& sphereGroupIds,
                             unsigned int selListSize, std::vector<unsigned int>& selListGroupIds);


    /*! @brief Id tagging output handling parameter retrieval
     *
     * @param[in]  initCond               initial condition
     * @param[in]  reader                 parameter file reader
     * @param[in]  outputFileSuffix       suffix to append to output file name
     * @param[out] taggingOutputSetup     id tagging output setup
     *
     * @return true if id tagging output is requested, false otherwise
     */
    bool readFileTaggingOutputAttributes(const std::string& initCond, IFileReader* reader,
                                         const std::string& outputFileSuffix, IdTaggingOutputSetup& taggingOutputSetup);

}
