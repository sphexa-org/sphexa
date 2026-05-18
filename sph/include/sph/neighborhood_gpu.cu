#include "neighborhood_gpu.hpp"

namespace sph
{

DeviceNeighborhoodData::DeviceNeighborhoodData()
    : impl(std::make_unique<Impl>())
{
}
DeviceNeighborhoodData::~DeviceNeighborhoodData() {}

} // namespace sph
