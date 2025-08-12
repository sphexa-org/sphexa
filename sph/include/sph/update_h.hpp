#pragma once

#include <cmath>
#include <iostream>
#include <vector>

#include "cstone/traversal/ijloop/common.hpp"
#include "sph/hydro_ve/xmass_kern.hpp"
#include "sph/kernels.hpp"
#include "sph/sph_gpu.hpp"

namespace sph
{

using cstone::LocalIndex;

template<class T>
void updateSmoothingLengthCpu(size_t startIndex, size_t endIndex, unsigned ng0, const unsigned* nc, T* h)
{
#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; i++)
    {
        h[i] = updateH(ng0, nc[i], h[i]);

#ifndef NDEBUG
        if (std::isinf(h[i]) || std::isnan(h[i])) printf("ERROR::h(%u) ngi %d h %f\n", i, nc[i], h[i]);
#endif
    }
}

template<class Dataset>
void updateSmoothingLength(const GroupView& grp, Dataset& d)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        updateSmoothingLengthGpu(grp, d.ng0, rawPtr(d.devData.nc), rawPtr(d.devData.h));
        syncGpu();
    }
    else { updateSmoothingLengthCpu(grp.firstBody, grp.lastBody, d.ng0, rawPtr(d.nc), rawPtr(d.h)); }
}

template<unsigned Stride = 1, class Interaction, class Postamble, class Tc, class T, class Input, class Output>
constexpr void jLoop(const cstone::Box<Tc>& box, const Tc* __restrict__ x, const Tc* __restrict__ y,
                     const Tc* __restrict__ z, const T* __restrict__ h, const cstone::LocalIndex i,
                     const cstone::LocalIndex* __restrict__ neighbors, const unsigned ng, Interaction&& interaction,
                     Postamble&& postamble, Input&& input, Output&& output)
{
    const auto constInput = cstone::ijloop::makeConst(std::forward<Input>(input));
    const auto iData      = cstone::ijloop::loadParticleData(x, y, z, h, constInput, i);
    const bool usePbc     = cstone::ijloop::requiresPbcHandling(box, iData);

    auto result = interaction(iData, iData, cstone::Vec3<Tc>{0, 0, 0}, Tc(0));
    for (unsigned nb = 0; nb < ng; ++nb)
    {
        const LocalIndex j     = neighbors[nb * Stride];
        const auto       jData = cstone::ijloop::loadParticleData(x, y, z, h, constInput, j);

        const auto [ijPosDiff, distSq] = cstone::ijloop::posDiffAndDistSq(usePbc, box, iData, jData);

        cstone::ijloop::updateResult(result, interaction(iData, jData, ijPosDiff, distSq));
    }

    cstone::ijloop::storeParticleData(std::forward<Output>(output), i,
                                      postamble(iData, cstone::ijloop::unwrapModifiers(result)));
}

template<class Tc, class T, class KeyType, class... Args>
void updateSmoothingLengthIterativeCpu(const Tc* x, const Tc* y, const Tc* z, T* h, unsigned* nc, LocalIndex firstId,
                                       LocalIndex lastId, const cstone::Box<Tc>& box,
                                       const cstone::OctreeNsView<Tc, KeyType>& treeView, unsigned ng0, unsigned ngmax,
                                       Args&&... args)
{
    LocalIndex numWork = lastId - firstId;

    unsigned ngmin = ng0 / 4;

    size_t        numFails     = 0;
    constexpr int maxIteration = 10;

#pragma omp parallel shared(numFails)
    {
        std::vector<LocalIndex> neighbors(ngmax);

#pragma omp for reduction(+ : numFails)
        for (LocalIndex i = 0; i < numWork; ++i)
        {
            LocalIndex id    = i + firstId;
            unsigned   ncSph = 1 + findNeighbors(id, x, y, z, h, treeView, box, ngmax, neighbors.data());

            int iteration = 0;
            while ((ngmin > ncSph || (ncSph - 1) > ngmax) && iteration++ < maxIteration)
            {
                h[id] = updateH(ng0, ncSph, h[id]);
                ncSph = 1 + findNeighbors(id, x, y, z, h, treeView, box, ngmax, neighbors.data());
            }
            numFails += (iteration >= maxIteration);

            nc[id] = ncSph;

            if constexpr (sizeof...(Args))
                jLoop(box, x, y, z, h, id, neighbors.data(), std::min(ncSph - 1, ngmax), std::forward<Args>(args)...);
        }
    }

    if (numFails)
    {
        std::cout << "Coupled h-neighbor count updated failed to converge for " << numFails << " particles"
                  << std::endl;
    }
}

template<class T, class Dataset, class... Args>
void updateSmoothingLengthIterative(const cstone::GroupView& groups, Dataset& d, const cstone::Box<T>& box,
                                    Args&&... args)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        updateSmoothingLengthIterativeGpu(groups, d, box, std::forward<Args>(args)...);
    }
    else
    {
        updateSmoothingLengthIterativeCpu(d.x.data(), d.y.data(), d.z.data(), d.h.data(), d.nc.data(), groups.firstBody,
                                          groups.lastBody, box, d.treeView, d.ng0, d.ngmax,
                                          std::forward<Args>(args)...);
    }
}

template<class Tc, class Dataset>
void updateSmoothingLengthIterativeAndComputeDensity(const cstone::GroupView& groups, Dataset& d,
                                                     const cstone::Box<Tc>& box)
{
    using T = typename decltype(d.rho)::value_type;
    const T*           wh;
    std::tuple<float*> input, output;
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>())
    {
        wh     = rawPtr(d.devData.wh);
        input  = {rawPtr(d.devData.m)};
        output = {rawPtr(d.devData.rho)};
    }
    else
    {
        wh     = d.wh.data();
        input  = {rawPtr(d.m)};
        output = {rawPtr(d.rho)};
    }
    updateSmoothingLengthIterative(groups, d, box, XmassInteraction<T>{wh}, XmassToDensityPostamble<T, Tc>{d.K}, input,
                                   output);
}

template<class Tc, class Dataset>
void updateSmoothingLengthIterativeAndComputeXMass(const cstone::GroupView& groups, Dataset& d,
                                                   const cstone::Box<Tc>& box)
{
    using T = typename decltype(d.rho)::value_type;
    const T*           wh;
    std::tuple<float*> input, output;
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>())
    {
        wh     = rawPtr(d.devData.wh);
        input  = {rawPtr(d.devData.m)};
        output = {rawPtr(d.devData.xm)};
    }
    else
    {
        wh     = d.wh.data();
        input  = {rawPtr(d.m)};
        output = {rawPtr(d.xm)};
    }
    updateSmoothingLengthIterative(groups, d, box, XmassInteraction<T>{wh}, XmassPostamble<T, Tc>{d.K}, input, output);
}

} // namespace sph
