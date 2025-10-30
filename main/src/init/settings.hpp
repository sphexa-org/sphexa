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
 * @author Christopher Bignamini <christopher.bignamini@gmail.com>
 */

#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <variant>
#include <vector>

#include "io/ifile_io.hpp"

namespace sphexa
{

using ScalarValue = double;
using VectorValue = std::vector<double>;

//! @brief Wrapper for scalar-type and vector-type parameters
struct Param {

    std::variant<ScalarValue, VectorValue> value;
    bool isEncodedString = false; //! @brief Flag to indicate if VectorValue contains encoded string data

public:

    Param(void) : value(ScalarValue{}), isEncodedString(false) {}

    // Conversion of all arithmetic types to ScalarValue
    // TODO: needed for the cases in the original code where an integer or enum is assigned to a Param
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T> || std::is_enum_v<T>>>
    Param(T v) : value(static_cast<ScalarValue>(v)), isEncodedString(false) {}

    Param(const VectorValue& v, bool isEncoded = false) : value(v), isEncodedString(isEncoded) {}
    Param(VectorValue&& v, bool isEncoded = false) : value(std::move(v)), isEncodedString(isEncoded) {}

    // Conversion of all arithmetic types to ScalarValue
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T> || std::is_enum_v<T>>>
    Param& operator=(T v) { value = static_cast<ScalarValue>(v); isEncodedString = false; return *this; }

    Param& operator=(const VectorValue& v) { value = v; isEncodedString = false; return *this; }
    Param& operator=(VectorValue&& v) { value = std::move(v); isEncodedString = false; return *this; }

    // Check stored type
    bool isScalar() const { return std::holds_alternative<ScalarValue>(value); }
    bool isVector() const { return std::holds_alternative<VectorValue>(value); }
    bool isString() const { return isEncodedString && isVector(); }

    // Implilcit conversion to ScalarValue
    // This allows: T r = constants.at("r"), needed to keep backward compatibility with existing code
    operator ScalarValue() const {
        if (!isScalar()) {
            throw std::runtime_error("Parameter is not a scalar value");
        }
        return std::get<ScalarValue>(value);
    }

    // Implilcit conversion to VectorValue
    operator VectorValue() const {
        if (!isVector()) {
            throw std::runtime_error("Parameter is not a vector value");
        }
        return std::get<VectorValue>(value);
    }

    const auto* data() const {
        if (isScalar()) {
            return &std::get<ScalarValue>(value);
        } else {
            return std::get<VectorValue>(value).data();
        }
    }

    auto* data() {
        if (isScalar()) {
            return &std::get<ScalarValue>(value);
        } else {
            return std::get<VectorValue>(value).data();
        }
    }

    auto begin() {
        if (isScalar()) {
            throw std::runtime_error("Cannot get begin iterator of scalar parameter");
        } else {
            return std::get<VectorValue>(value).begin();
        }
    }

    auto end() {
        if (isScalar()) {
            throw std::runtime_error("Cannot get end iterator of scalar parameter");
        } else {
            return std::get<VectorValue>(value).end();
        }
    }

    size_t size() const {
        if (isScalar()) {
            return 1;
        } else {
            // TODO: in case of an encoded string, shold we exlcude the commas from the size?
            return std::get<VectorValue>(value).size();
        }
    }

    std::vector<std::string> toStrings() const
    {
        if (isScalar()) {
            // TODO: find a better way to handle this case
            std::cout<<"Warning: converting scalar Param to string vector of size 1"<<std::endl;
            return std::vector<std::string>{std::to_string(std::get<ScalarValue>(value))};
        }

        std::vector<std::string> result;
        if(isEncodedString)
        {
            std::string current;
            const double comma_ascii_code = 44.0;

            for (double code : std::get<VectorValue>(value))
            {
                if (code == comma_ascii_code)
                {
                    // Found a separator, save current string
                    if (!current.empty())
                    {
                        result.push_back(current);
                        current.clear();
                    }
                }
                else
                {
                    char ch = static_cast<char>(static_cast<int>(code));
                    current += ch;
                }
            }

            // Add the last string
            if (!current.empty())
            {
                result.push_back(current);
            }
        }
        else
        {
            // This is a numeric vector, convert to strings
            for (double value : std::get<VectorValue>(value))
            {
                result.push_back(std::to_string(value));
            }
        }
        return result;
    }

};

using InitSettings = std::map<std::string, Param>;

//! @brief Get maximum value from a Param
inline ScalarValue max(const Param& param) {
    if (param.isScalar()) {
        return std::get<ScalarValue>(param.value);
    } else {
        const auto& vec = std::get<VectorValue>(param.value);
        if (vec.empty()) {
            throw std::runtime_error("Cannot get max of empty vector parameter");
        }
        return *std::max_element(vec.begin(), vec.end());
    }
}

//! @brief Get minimum value from a Param
inline ScalarValue min(const Param& param) {
    if (param.isScalar()) {
        return std::get<ScalarValue>(param.value);
    } else {
        const auto& vec = std::get<VectorValue>(param.value);
        if (vec.empty()) {
            throw std::runtime_error("Cannot get min of empty vector parameter");
        }
        return *std::min_element(vec.begin(), vec.end());
    }
}

/*! @brief Convert comma-separated string to VectorValue
 *
 * @param[in] input  String to be encoded (e.g., "x,y,rho" or "output_filename")
 *
 * @return VectorValue containing character codes as doubles (including commas)
 *
 * Examples:
 *   std::string fields = "x,y,z";
 *   VectorValue codes = stringToVectorValue(fields);  // {120.0, 44.0, 121.0, 44.0, 122.0}
 *
 *   std::string fields = "abc,def";
 *   VectorValue codes = stringToVectorValue(fields);  // {97.0, 98.0, 99.0, 44.0, 100.0, 101.0, 102.0}
 *
 *   std::string single = "xyz";
 *   VectorValue codes = stringToVectorValue(single);  // {120.0, 121.0, 122.0} (no comma separator)
 */
inline VectorValue stringToVectorValue(const std::string& input)
{
    VectorValue result;
    result.reserve(input.size());

    // Encode all characters including commas
    std::for_each(input.begin(), input.end(), [&](char ch) {
        result.push_back(static_cast<double>(static_cast<unsigned char>(ch)));
    });

    return result;
}


//! @brief write @p InitSettings as file attributes of a new file @p path
inline void writeSettings(const InitSettings& settings, const std::string& path, IFileWriter* writer)
{
    if (std::filesystem::exists(path))
    {
        throw std::runtime_error("Cannot write settings: file " + path + " already exists\n");
    }

    writer->addStep(0, 0, path);
    for (auto it = settings.cbegin(); it != settings.cend(); ++it)
    {
        if(it->second.isScalar() && !it->second.isString())
            writer->fileAttribute(it->first, it->second.data(), 1);
        else
            std::cout<<"WARNING: skipping writing non-scalar setting: "<<it->first<<std::endl;
    }
    writer->closeStep();
}

//! @brief Used to initialize particle dataset attributes from builtin named test-cases
class BuiltinWriter
{
public:
    using FieldType = util::Reduce<std::variant, util::Map<std::add_pointer_t, IO::Types>>;

    explicit BuiltinWriter(InitSettings attrs)
        : attributes_(std::move(attrs))
    {
    }

    [[nodiscard]] static int rank() { return -1; }

    void stepAttribute(const std::string& key, FieldType val, int64_t /*size*/)
    {
        std::visit([this, &key](auto arg) { *arg = attributes_.at(key); }, val);
    };

private:
    InitSettings attributes_;
};

} // namespace sphexa
