#include <cmath>
#include <format>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include "io/id_tag_setup.hpp"

namespace sphexa
{
    void idTaggingSetupInit(InitSettings& settings, bool printSetup)
    {
        // Check id tagging output options
        // If output is not provided or not positive, skip tagging setup
        if(!settings.count("w_subset") || *settings.at("w_subset").data() <= 0.) return;

        // Check presence of tagging selection options
        const bool idSphereSelectionRequested = settings.count("id_selection_spheres");
        const bool idListSelectionRequested   = settings.count("id_selection_list");

        // Preliminary check on subset settings consistency
        // Check if spherical selection is requested
        if(idSphereSelectionRequested)
        {

            if(settings.at("id_selection_spheres").size() % 4 != 0)
            {
                throw std::runtime_error("Invalid sphere selection settings: each sphere must be defined by 4 values (x,y,z,radius).");
            }

            // Check if sphere ids are provided
            if(settings.count("id_selection_spheres_group_ids"))
            {
                // Check if data sizes are consistent
                if(settings.at("id_selection_spheres").size()/4 != settings.at("id_selection_spheres_group_ids").size())
                {
                    throw std::runtime_error("Inconsistent sphere selection settings: number of spheres and sphere ids do not match.");
                }

                // Check if provided group ids are non-negative integers
                for(auto groupId : settings.at("id_selection_spheres_group_ids"))
                {
                    if(groupId < 0.0)
                    {
                        throw std::runtime_error("Invalid sphere selection settings: sphere group ids must be non-negative.");
                    }
                }
            }

            // Check if radius values are positive
            for(unsigned int i=0; i<settings.at("id_selection_spheres").size()/4; ++i)
            {
                if(settings.at("id_selection_spheres").data()[4*i + 3] <= 0.0)
                {
                    throw std::runtime_error("Invalid sphere selection settings: sphere radius must be positive.");
                }
            }
        }

        // Check if list selection is requested
        if(idListSelectionRequested)
        {
            // Check if provided ids are non-negative integers
            for(auto id : settings.at("id_selection_list"))
            {
                if(id < 0.0)
                {
                    throw std::runtime_error("Invalid list selection settings: particle ids must be non-negative.");
                }
            }

            // Check if provided ids are unique
            std::vector<double> ids = settings.at("id_selection_list");
            std::sort(ids.begin(), ids.end());
            auto it = std::adjacent_find(ids.begin(), ids.end());
            if(it != ids.end())
            {
                throw std::runtime_error("Invalid list selection settings: particle ids must be unique.");
            }

            // Check if list ids are provided
            if(settings.count("id_selection_list_group_ids"))
            {
                // Check if data sizes are consistent
                if(settings.at("id_selection_list").size() != settings.at("id_selection_list_group_ids").size())
                {
                    throw std::runtime_error("Inconsistent list selection settings: number of lists and group ids do not match.");
                }

                // Check if provided group ids are non-negative integers
                for(auto groupId : settings.at("id_selection_list_group_ids"))
                {
                    if(groupId < 0.0)
                    {
                        throw std::runtime_error("Invalid list selection settings: list group ids must be non-negative.");
                    }
                }
            }
        }

        // Check for group ids duplicates between sphere and list selections 
        if(settings.count("id_selection_spheres_group_ids") && settings.count("id_selection_list_group_ids"))
        {
            // TODO: not sure this is needed
        }

        // If group ids are not provided, default values are assigned
        if(idSphereSelectionRequested && !settings.count("id_selection_spheres_group_ids"))
        {   
            uint32_t defaultSphereIdsStart = 0;
            if(settings.count("id_selection_list_group_ids"))
            {
                defaultSphereIdsStart = max(settings.at("id_selection_list_group_ids")) + 1;
            }

            settings["id_selection_spheres_group_ids"] = VectorValue(settings.at("id_selection_spheres").size()/4);
            std::iota(settings["id_selection_spheres_group_ids"].begin(), settings["id_selection_spheres_group_ids"].end(), defaultSphereIdsStart);
            std::cout<<"WARNING: sphere group ids not provided, assigning default values starting from "<<defaultSphereIdsStart<<std::endl;
        }

        if(idListSelectionRequested && !settings.count("id_selection_list_group_ids"))
        {
            uint32_t defaultListId = 0;
            if(settings.count("id_selection_spheres_group_ids"))
            {
                defaultListId = max(settings.at("id_selection_spheres_group_ids")) + 1;
            }
            settings["id_selection_list_group_ids"] = VectorValue(settings.at("id_selection_list").size(), defaultListId);
            std::cout<<"WARNING: list group ids not provided, assigning default value = "<<defaultListId<<std::endl;
        }

        if(printSetup)
        {
            if(idSphereSelectionRequested || idListSelectionRequested)
            {
                std::cout<<"Id tagging settings:"<<std::endl;
                if(settings.count("id_selection_spheres"))
                {
                    std::cout<<" - Spherical selection:"<<std::endl;
                    for(unsigned int i=0; i<settings.at("id_selection_spheres").size()/4; ++i)
                    {
                        std::cout<<"   Sphere "<<i<<": center=("
                                <<settings.at("id_selection_spheres").data()[4*i + 0]<<", "
                                <<settings.at("id_selection_spheres").data()[4*i + 1]<<", "
                                <<settings.at("id_selection_spheres").data()[4*i + 2]<<"), radius="
                                <<settings.at("id_selection_spheres").data()[4*i + 3]<<", group id="
                                <<settings.at("id_selection_spheres_group_ids").data()[i]<<std::endl;
                    }
                }
                if(settings.count("id_selection_list"))
                {
                    std::cout<<" - List selection with:"<<std::endl;
                    std::cout<<"   Id list: "<<static_cast<uint64_t>(settings.at("id_selection_list").data()[0])<<", "
                        <<static_cast<uint64_t>(settings.at("id_selection_list").data()[1])<<",..."
                        <<static_cast<uint64_t>(settings.at("id_selection_list").data()[settings.at("id_selection_list").size()-1])<<std::endl;
                    std::cout<<"   Group ids: "<<static_cast<uint64_t>(settings.at("id_selection_list_group_ids").data()[0])<<", "
                        <<static_cast<uint64_t>(settings.at("id_selection_list_group_ids").data()[1])<<",..."
                        <<static_cast<uint64_t>(settings.at("id_selection_list_group_ids").data()[settings.at("id_selection_list_group_ids").size()-1])<<std::endl;
                }
            }
        }
    };

    // TODO: remove not needed debug logging
    bool idTaggingOutputParameterRetrieval(const InitSettings& settings, const std::string initCond, const std::string outputFileSuffix, 
                                           std::string& outFileSubset, std::string& writeFreqStrSubset, std::vector<std::string>& outputFieldsSubset,
                                           std::vector<std::string>& writeExtraSubset)
    {

        outFileSubset.clear();
        writeFreqStrSubset = std::string("0");
        outputFieldsSubset.clear();
        writeExtraSubset.clear();

        const bool writeEnabledSubset = (settings.count("w_subset") && *settings.at("w_subset").data() > 0.) ||
            settings.count("wextra_subset");

        if(writeEnabledSubset) {

            if(settings.count("o_subset")) {
                outFileSubset = static_cast<StringValue>(settings.at("o_subset"));
            }
            else {
                std::cout<<"WARNING: o_subset not provided, using default naming convention"<<std::endl;
                outFileSubset =  "dump_subset_" + initCond;
            }
            outFileSubset += outputFileSuffix;

            if(settings.count("f_subset")) {
                // outputFieldsSubset = settings.at("f_subset").toStrings();
                // std::vector<std::string> result;
                for (auto part : static_cast<StringValue>(settings.at("f_subset")) | std::views::split(',')) {
                    outputFieldsSubset.emplace_back(part.begin(), part.end());
                }
            }
            else {
                std::cout<<"WARNING: f_subset not provided, all fields will be printed for the tagged id subsets."<<std::endl;
            }

            if(settings.count("w_subset")) {
                std::cout << "Write frequency subset: " << std::endl;
                if(settings.at("w_subset").isScalar()) {
                    // If integer, format without decimal point
                    if (std::floor(settings.at("w_subset")) == settings.at("w_subset")) {
                        writeFreqStrSubset = std::format("{}", static_cast<int>(settings.at("w_subset")));
                    }
                    else {
                        // TODO: is there a way to avoid dereferencing and data()?
                        writeFreqStrSubset = std::format("{:.15g}", *settings.at("w_subset").data());
                    }
                }
                else {
                    throw std::runtime_error("w_subset parameter must be a scalar value");
                }
            }

            if(settings.count("wextra_subset")) {
                for(auto val : settings.at("wextra_subset")) {
                    if(std::floor(val) == val) {
                        writeExtraSubset.push_back(std::format("{}", static_cast<int>(val)));
                    }
                    else {
                        writeExtraSubset.push_back(std::format("{:.15g}", val));
                    }
                }
            }
        }
        return writeEnabledSubset;
    };
}
