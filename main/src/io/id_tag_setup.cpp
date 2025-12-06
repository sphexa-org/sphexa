// #include <cmath>
#include <iostream>
#include <ranges>
#include <numeric>
// #include <stdexcept>

#include "io/id_tag_setup.hpp"

namespace sphexa
{

    void idTaggingGroupsInit(unsigned int selSpheresNum, std::vector<unsigned int>& sphereGroupIds,
        unsigned int selListSize, std::vector<unsigned int>& selListGroupIds)
    {
        // If group ids are not provided, default values are assigned
        if(selSpheresNum > 0 && sphereGroupIds.size() == 0)
        {   
            uint32_t defaultSphereIdsStart = 0;
            if(selListGroupIds.size() > 0)
            {
                defaultSphereIdsStart = *std::max_element(selListGroupIds.begin(), selListGroupIds.end()) + 1;
            }

            sphereGroupIds.resize(selSpheresNum);
            std::iota(sphereGroupIds.begin(), sphereGroupIds.end(), defaultSphereIdsStart);
            std::cout<<"WARNING: sphere group ids not provided, assigning default values starting from "<<defaultSphereIdsStart<<std::endl;
        }

        if(selListSize > 0 && selListGroupIds.size() == 0)
        {
            uint32_t defaultListId = 0;
            if(sphereGroupIds.size() > 0)
            {
                defaultListId = *std::max_element(sphereGroupIds.begin(), sphereGroupIds.end()) + 1;
            }
            selListGroupIds.resize(selListSize, defaultListId);
            std::cout<<"WARNING: list group ids not provided, assigning default value = "<<defaultListId<<std::endl;
        }
    }

    void idTaggingSetupCheck(std::vector<IdSelectionSphere>& selSpheres, std::vector<unsigned int>& sphereGroupIds, 
        std::vector<uint64_t>& selList, std::vector<unsigned int>& selListGroupIds, bool printLog)
    {
        // Check sphere selection data
        // for(unsigned int i=0; i<selSpheres.size(); ++i)
        for(auto sphere : selSpheres)
        {
            if(sphere[3] <= 0.0)
            {
                throw std::runtime_error("Invalid sphere selection settings: sphere radius must be positive.");
            }
        }

        // Check list selection data: // TODO: move to the right place, id are non negative by construction
        for(auto id : selList)
        {
            if(id < 0)
            {
                throw std::runtime_error("Invalid list selection settings: particle ids must be non-negative.");
            }
        }
        // Check for unique ids
        // TODO: check if tagging expects sorted list
        std::vector<uint64_t> selListSorted = selList;
        std::sort(selListSorted.begin(), selListSorted.end());
        auto it = std::adjacent_find(selListSorted.begin(), selListSorted.end());
        if(it != selListSorted.end())
        {
            throw std::runtime_error("Invalid id selection list: particle ids must be unique.");
        }

        // Groups id checks
        if(sphereGroupIds.size() != 0 && sphereGroupIds.size() != selSpheres.size())
        {
            throw std::runtime_error("Inconsistent sphere selection settings: number of spheres and sphere ids do not match.");
        }
        // Check for negative group ids
        for(auto groupId : sphereGroupIds)
        {
            if(groupId < 0)
            {
                throw std::runtime_error("Invalid sphere selection settings: sphere group ids must be non-negative.");
            }
        }
        if(selListGroupIds.size() != 0 && selListGroupIds.size() != selList.size())
        {
            throw std::runtime_error("Inconsistent list selection settings: number of selected ids and list group ids do not match.");
        }
        for(auto groupId : selListGroupIds)
        {
        // Check for negative group ids
            if(groupId < 0)
            {
                throw std::runtime_error("Invalid list selection settings: list group ids must be non-negative.");
            }
        }
        // Check for group ids duplicates between sphere and list selections 
        if(selListGroupIds.size() != 0 && sphereGroupIds.size() != 0)
        {
            // TODO: not sure this is needed
        }

        // Assign default group ids if not provided
        if(selListGroupIds.size() == 0 || sphereGroupIds.size() == 0)
        {
            idTaggingGroupsInit(selSpheres.size(), sphereGroupIds, selList.size(), selListGroupIds);
        }

        if(printLog)
        {
            if(selSpheres.size() > 0 || selList.size() > 0)
            {
                std::cout<<"Id tagging settings:"<<std::endl;
                if(selSpheres.size() > 0)
                {
                    std::cout<<" - Spherical selection:"<<std::endl;
                    for(auto i = 0; i< selSpheres.size(); ++i)
                    {
                        std::cout<<"   Sphere "<<i<<": center=("
                                <<selSpheres[i][0]<<", "
                                <<selSpheres[i][1]<<", "
                                <<selSpheres[i][2]<<"), radius="
                                <<selSpheres[i][3]<<", group id="
                                <<sphereGroupIds[i]<<std::endl;
                    }
                }
                if(selList.size() > 0)
                {
                    std::cout<<" - List selection with:"<<std::endl;
                    // TODO: find better way to log data here
                    std::cout<<"   Id list: "<<selList[0]<<" - "<<selList[selList.size()-1]<<std::endl;
                    std::cout<<"   Group ids: "<<selListGroupIds[0]<<" - "<<selListGroupIds[selListGroupIds.size()-1]<<std::endl;
                }
            }
        }
    }

    bool readFileTaggingOutputAttributes(const std::string& settingsFile, IFileReader* reader,
                                         const std::string initCond, const std::string outputFileSuffix, 
                                         std::string& outFileSubset, std::string& writeFreqStrSubset, 
                                         std::vector<std::string>& outputFieldsSubset,
                                         std::vector<std::string>& writeExtraSubset)
    {   

        bool writeEnabledSubset = false;

        if (not settingsFile.empty())
        {
            outFileSubset.clear();
            writeFreqStrSubset = std::string("0");
            outputFieldsSubset.clear();
            writeExtraSubset.clear();

            reader->setStep(settingsFile, -1, FileMode::independent);

            auto fileAttributes = reader->fileAttributes();

            if(std::ranges::find(fileAttributes, std::string("w_subset")) != fileAttributes.end())
            {
                auto attr_size = reader->fileAttributeSize("w_subset");
                if(attr_size != 1) // TODO: should I move this check to anoter place?
                    throw std::runtime_error("Invalid id tagging output settings: write frequency must be a single value.");
                double writeFreqValue;
                reader->fileAttribute("w_subset", &writeFreqValue, attr_size);
                if(std::floor(writeFreqValue) == writeFreqValue)
                {
                    // TODO: is there already something to format numbers from string in the codebase?
                    // TODO: check for negative values
                    writeFreqStrSubset = std::format("{}", static_cast<int>(writeFreqValue));
                }
                else
                {
                    writeFreqStrSubset = std::format("{:.15g}", writeFreqValue);
                }
            }

            if(std::ranges::find(fileAttributes, std::string("wextra_subset")) != fileAttributes.end())
            {
                auto attr_size = reader->fileAttributeSize("wextra_subset");
                std::vector<double> writeExtraSubsetTemp(attr_size);
                reader->fileAttribute("wextra_subset", writeExtraSubsetTemp.data(), attr_size);
                writeExtraSubset.resize(attr_size);
                for(unsigned int i=0; i<attr_size; ++i)
                {
                    if(std::floor(writeExtraSubsetTemp[i]) == writeExtraSubsetTemp[i])
                    {
                        writeExtraSubset[i] = std::format("{}", static_cast<int>(writeExtraSubsetTemp[i]));
                    }
                    else
                    {
                        writeExtraSubset[i] = std::format("{:.15g}", writeExtraSubsetTemp[i]);
                    }
                }
                // TODO: add checks on writeExtraSubset values
            }

            if(std::stod(writeFreqStrSubset) > 0.0 || writeExtraSubset.size() > 0)
            {
                writeEnabledSubset = true;

                if(std::ranges::find(fileAttributes, std::string("o_subset")) != fileAttributes.end())
                {
                    // TODO: add check on attribute type/size
                    reader->fileAttribute("o_subset", outFileSubset);
                }
                else
                {
                    std::cout<<"WARNING: o_subset not provided, using default naming convention"<<std::endl;
                    outFileSubset =  "dump_subset_" + initCond;
                }
                outFileSubset += outputFileSuffix;

                if(std::ranges::find(fileAttributes, std::string("f_subset")) != fileAttributes.end())
                {
                    std::string outputFieldsStrSubset;
                    reader->fileAttribute("f_subset", outputFieldsStrSubset);
                    for (auto part : outputFieldsStrSubset | std::views::split(',')) {
                        outputFieldsSubset.emplace_back(part.begin(), part.end());
                    }
                }
                else
                {
                    std::cout<<"WARNING: f_subset not provided, all fields will be printed for the tagged id subsets."<<std::endl;
                }
            }
            reader->closeStep();
        }
        return writeEnabledSubset;
    }
}
