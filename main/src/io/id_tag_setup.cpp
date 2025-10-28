#include <iostream>
#include <numeric>
#include <stdexcept>

#include "io/id_tag_setup.hpp"

namespace sphexa
{
    void idTaggingSetupInit(InitSettings& settings)
    {
        std::cout << "Initializing id tagging setup" << std::endl;

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
    };

    // TODO: remove not needed debug logging
    bool idTaggingOutputParameterRetrieval(const InitSettings& settings, const std::string initCond, const std::string outputFileSuffix, 
                                           std::string& outFileSubset, std::string& writeFreqStrSubset, std::vector<std::string>& outputFieldsSubset,
                                           std::vector<std::string>& writeExtraSubset)
    {
        const bool writeEnabledSubset = settings.count("w_subset") && *settings.at("w_subset").data() > 0.;

        if(writeEnabledSubset) {
            if(settings.count("o_subset")) {
                std::cout<<"is o_subset scalar "<<settings.at("o_subset").isScalar()<<std::endl;
                std::cout<<"is o_subset vector "<<settings.at("o_subset").isVector()<<std::endl;
                outFileSubset = settings.at("o_subset").toStrings()[0];
            }
            else {
                std::cout<<"o_subset not provided, using default naming convention"<<std::endl;
                outFileSubset =  "dump_subset_" + initCond;
            }
            outFileSubset += outputFileSuffix;
            std::cout<<"Subset output file: " << outFileSubset << std::endl;
            std::cout<<std::endl;
            if(settings.count("f_subset")) {
                std::cout<<"is f_subset scalar "<<settings.at("f_subset").isScalar()<<std::endl;
                std::cout<<"is f_subset vector "<<settings.at("f_subset").isVector()<<std::endl;
                outputFieldsSubset = settings.at("f_subset").toStrings();
                for(const auto& field : outputFieldsSubset) {
                    std::cout << "Subset output field: " << field << std::endl;
                }
                std::cout<<std::endl;
            }
            writeFreqStrSubset = settings.at("w_subset").toStrings()[0];
            std::cout<<"Subset write frequency: " << writeFreqStrSubset << std::endl;
            std::cout<<std::endl;

            if(settings.count("wextra_subset")) {
                std::cout<<"is wextra_subset scalar "<<settings.at("wextra_subset").isScalar()<<std::endl;
                std::cout<<"is wextra_subset vector "<<settings.at("wextra_subset").isVector()<<std::endl;
                writeExtraSubset = settings.at("wextra_subset").toStrings();
                for(const auto& freq : writeExtraSubset) {
                    std::cout << "Subset extra output freq: " << freq << std::endl;
                }
                std::cout<<std::endl;
            }
        }
        return writeEnabledSubset;
    };
}
