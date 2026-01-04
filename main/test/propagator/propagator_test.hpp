/*
 * MIT License
 *
 * SPH-EXA
 * Copyright (c) 2024 CSCS, ETH Zurich, University of Basel, University of Zurich
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
 * @brief Propagator unit test utility class
 *
 * @author Christopher Bignamini <christopher.bignamini@gmail.com>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <variant>

#include "cstone/domain/domain.hpp"
#include "propagator/ipropagator.hpp"
#include "sph/types.hpp"
#include "sphexa/simulation_data.hpp"

#ifdef USE_CUDA
using AccType = cstone::GpuTag;
using FieldVariant = std::variant<cstone::DeviceVector<float>*, cstone::DeviceVector<double>*, cstone::DeviceVector<unsigned>*,
                                    cstone::DeviceVector<uint64_t>*, cstone::DeviceVector<uint8_t>*>;
#else
using AccType = cstone::CpuTag;
using FieldVariant = std::variant<std::vector<float>*, std::vector<double>*, std::vector<unsigned>*,
                                    std::vector<uint64_t>*, std::vector<uint8_t>*>;
#endif

using Dataset = sphexa::SimulationData<AccType>;
using Domain  = cstone::Domain<sph::SphTypes::KeyType, sph::SphTypes::CoordinateType, AccType>;

template<class DomainType, class ParticleDataType>
class PropagatorTest: public sphexa::Propagator<DomainType, ParticleDataType>
{
    using Base = sphexa::Propagator<DomainType, ParticleDataType>;

public:

    using Base::outputField;
    using Base::AccVector;

    PropagatorTest(std::ostream& output, int rank)
        : Base(output, rank) {}

    std::vector<std::string> conservedFields() const override 
    {
        return {};
    }
    void activateFields(ParticleDataType& d) override {};
    void sync(DomainType& domain, ParticleDataType& d) override {};
    void computeForces(DomainType& domain, ParticleDataType& d) override {};
    void integrate(DomainType& domain, ParticleDataType& d) override {};
};

