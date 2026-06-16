#include "neighborhood_gpu.hpp"

namespace sph
{

void DeviceNeighborhoodData::setType(NeighborhoodType type) { impl->neighborhoodType = type; }

DeviceNeighborhoodData::DeviceNeighborhoodData()
    : impl(std::make_unique<Impl>())
{
}
DeviceNeighborhoodData::~DeviceNeighborhoodData() {}

} // namespace sph
