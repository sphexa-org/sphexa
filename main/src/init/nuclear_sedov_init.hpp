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
 * @brief Nuclear sedov data initializing
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */

#include "sphnnet/initializers.hpp"

#include "nnet/parameterization/net87/net87.hpp"
#include "nnet/parameterization/net14/net14.hpp"

#include "sedov_init.hpp"

namespace sphexa
{

std::map<std::string, double> nuclearSedovConstants()
{
    std::map<std::string, double> ret{
        {"dim", 3},           {"gamma", 5. / 3.}, {"omega", 0.},           {"r0", 0.},    {"r1", 0.5}, {"mTotal", 1e9},
        {"energyTotal", 1e9}, {"T0", 1e9},        {"width", 0.1},          {"rho0", 1e9}, {"u0", 1e1}, {"p0", 0.},
        {"vr0", 0.},          {"cs0", 0.},        {"firstTimeStep", 1e-6}, {"mui", 10}};

    ret["ener0"] = ret["energyTotal"] / std::pow(M_PI, 1.5) / 1. / std::pow(ret["width"], 3.0);
    return ret;
}

template<class Dataset>
class NuclearSedovGlass : public ISimInitializer<Dataset>
{
    std::string                   glassBlock;
    std::map<std::string, double> constants_;

public:
    NuclearSedovGlass(std::string initBlock)
        : glassBlock(initBlock)
    {
        constants_ = nuclearSedovConstants();
    }

    /*! @brief initialize particle data with a constant density cube
     *
     * @param[in]    rank             MPI rank ID
     * @param[in]    numRanks         number of MPI ranks
     * @param[in]    cbrtNumPart      the cubic root of the global number of particles to generate
     * @param[inout] d                particle dataset
     * @return                        the global coordinate bounding box
     */
    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cbrtNumPart,
                                                 Dataset& simData) const override
    {
        auto& d       = simData.hydro;
        auto& n       = simData.nuclearData;
        using KeyType = typename Dataset::KeyType;
        using T       = typename Dataset::RealType;

        /* !!!!!!!!!!!!!!!!!!!! */
        /* hydro initialization */
        /* !!!!!!!!!!!!!!!!!!!! */

        std::vector<T> xBlock, yBlock, zBlock;
        fileutils::readTemplateBlock(glassBlock, xBlock, yBlock, zBlock);
        size_t blockSize = xBlock.size();

        size_t multiplicity  = std::rint(cbrtNumPart / std::cbrt(blockSize));
        d.numParticlesGlobal = multiplicity * multiplicity * multiplicity * blockSize;

        T              r = constants_.at("r1");
        cstone::Box<T> globalBox(-r, r, cstone::BoundaryType::periodic);

        auto [keyStart, keyEnd] = partitionRange(cstone::nodeRange<KeyType>(0), rank, numRanks);
        assembleCube<T>(keyStart, keyEnd, globalBox, multiplicity, xBlock, yBlock, zBlock, d.x, d.y, d.z);

        size_t last_first = d.x.size();
        d.resize(last_first);
        initSedovFields(d, constants_);

        /* !!!!!!!!!!!!!!!!!!!!!! */
        /* nuclear initialization */
        /* !!!!!!!!!!!!!!!!!!!!!! */

        util::array<double, 87> Y0_87, X_87;
        for (int i = 0; i < 86; ++i)
        {
            X_87[i] = 0;
        }

        if (n.numSpecies == 14)
        {
            X_87[1] = 0.5;
            X_87[2] = 0.5;

            for (int i = 0; i < 14; ++i)
            {
                Y0_87[i] = X_87[i] / nnet::net14::constants::A[i];
            }
        }
        else if (n.numSpecies == 86 || n.numSpecies == 87)
        {
            X_87[nnet::net86::constants::net14_species_order[1]] = 0.5;
            X_87[nnet::net86::constants::net14_species_order[2]] = 0.5;

            for (int i = 0; i < 86; ++i)
            {
                Y0_87[i] = X_87[i] / nnet::net86::constants::A[i];
            }
        }
        else
        {
            throw std::runtime_error("not able to initialize " + std::to_string(n.numSpecies) + " nuclear species !");
        }

        sphnnet::initializeNuclearPointers(0, last_first, simData);
        sphnnet::initNuclearDataFromConst(0, last_first, simData, Y0_87);

        // initialize dt
        std::fill(n.dt.begin(), n.dt.end(), nnet::constants::initial_dt);

        // initialize temp
        size_t n_nuclear_particles = n.Y[0].size();
        std::fill(getHost<"temp">(n).begin(), getHost<"temp">(n).begin() + n_nuclear_particles, constants_.at("T0"));

        return globalBox;
    }

    const std::map<std::string, double>& constants() const override { return constants_; }
};

} // namespace sphexa