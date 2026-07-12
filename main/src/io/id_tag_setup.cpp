#include <filesystem>
#include <iostream>
#include <numeric>
#include <ranges>

#include "arg_parser.hpp"
#include "id_tag_setup.hpp"

namespace sphexa
{
    std::optional<IdTaggingOutputSetup> readFileTaggingOutputAttributes(const std::string& initCond, IFileReader* reader,
                                                                         const std::string& outputFileSuffix)
    {
        IdTaggingOutputSetup taggingOutputSetup;

        // Determine tagging parameter file path (parameter file or restart file)
        std::string settingsFile{};
        if(std::filesystem::exists(strBeforeSign(initCond, ":")))
        {
            // Restart file provided
            settingsFile = strBeforeSign(initCond, ":");
        }
        else if(std::filesystem::exists(strAfterSign(initCond, ":")))
        {
            // Parameter file provided
            settingsFile = strAfterSign(initCond, ":");
        }

        if (not settingsFile.empty())
        {
            reader->setStep(settingsFile, -1, FileMode::independent);

            auto fileAttributes = reader->fileAttributes();

            auto hasAttribute = [&fileAttributes](std::string_view name)
            { return std::ranges::find(fileAttributes, name) != fileAttributes.end(); };

            if(hasAttribute("w_subset"))
            {
                auto attr_size = reader->fileAttributeSize("w_subset");
                double writeFreqValue;
                reader->fileAttribute("w_subset", &writeFreqValue, attr_size);
                taggingOutputSetup.writeFreqStr = numToParamStr(writeFreqValue);
            }

            if(hasAttribute("wextra_subset"))
            {
                auto attr_size = reader->fileAttributeSize("wextra_subset");
                std::vector<double> writeExtraSubsetTemp(attr_size);
                reader->fileAttribute("wextra_subset", writeExtraSubsetTemp.data(), attr_size);
                taggingOutputSetup.writeExtra.resize(attr_size);
                for(unsigned int i=0; i<attr_size; ++i)
                {
                    taggingOutputSetup.writeExtra[i] = numToParamStr(writeExtraSubsetTemp[i]);
                }
            }

            if(std::stod(taggingOutputSetup.writeFreqStr) > 0.0 || taggingOutputSetup.writeExtra.size() > 0)
            {
                if(hasAttribute("o_subset"))
                {
                    reader->fileAttribute("o_subset", taggingOutputSetup.outFile);
                }
                else
                {
                    taggingOutputSetup.outFile =  "dump_subset_" + removeModifiers(initCond);
                }
                taggingOutputSetup.outFile += outputFileSuffix;

                if(hasAttribute("f_subset"))
                {
                    std::string outputFieldsStrSubset;
                    reader->fileAttribute("f_subset", outputFieldsStrSubset);
                    taggingOutputSetup.outputFields = splitCommaList(outputFieldsStrSubset);
                }

                reader->closeStep();
                return taggingOutputSetup;
            }
            reader->closeStep();
        }
        return std::nullopt;
    }
}
