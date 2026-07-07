#include "neighborhood_gpu.hpp"

namespace sph
{

void DeviceNeighborhoodData::disableNeighborLists() { impl->useNeighborLists = false; }

DeviceNeighborhoodData::DeviceNeighborhoodData()
    : impl(std::make_unique<Impl>())
{
}
DeviceNeighborhoodData::~DeviceNeighborhoodData() {}

} // namespace sph
