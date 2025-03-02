#pragma once

#include "cstone/sfc/box.hpp"
#include "cstone/traversal/groups.hpp"
#include "cstone/tree/octree.hpp"
#include "cstone/tree/definitions.h"
#include "sph/timestep.h"

namespace sph
{

using cstone::GroupData;
using cstone::GroupView;

template<class Dataset>
extern void computeIADGpu(const GroupView&, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
extern void computeMomentumEnergyStdGpu(const GroupView&, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

namespace cuda
{

template<class Dataset>
extern void computeXMass(const GroupView&, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
void computeDensity(const GroupView&, Dataset& d, const cstone::Box<typename Dataset::RealType>& box);

template<class Dataset>
extern void computeVeDefGradh(const GroupView&, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
extern void computeIadDivvCurlv(const GroupView&, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
extern void computeAVswitches(const GroupView&, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<bool avClean, class Dataset>
extern void computeMomentumEnergy(const GroupView&, float*, Dataset&, const cstone::Box<typename Dataset::RealType>&);

template<class Tt, class Tm, class Th>
extern void computeIdealGasEOS(size_t firstParticle, size_t lastParticle, Tm mui, Tt gamma, const Tt* temp, const Tt* u,
                               const Tm* m, const Th* kx, const Th* xm, const Th* gradh, Th* prho, Th* c, Th* rho,
                               Th* p);

template<class Th, class Tu>
extern void computeIsothermalEOS(size_t first, size_t last, Th cConst, Th* c, Th* rho, Th* p, const Th* m, const Th* kx,
                                 const Th* xm, const Th* gradh, Th* prho, Tu* temp);

template<class Th, class Tu>
extern void computeIsothermalEOS(size_t firstParticle, size_t lastParticle, Th cConst, Th* c, Th* rho, Th* p,
                                 const Th* m, const Th* kx, const Th* xm, const Th* gradh, Th* prho, Tu* temp);

template<class Th, class Tt>
extern void computePolytropicEOS(size_t firstParticle, size_t lastParticle, Tt polytropic_const, Tt polytropic_index,
                                 Th* rho, Th* p, const Th* m, const Th* kx, const Th* xm, const Th* gradh, Th* prho,
                                 Tt* temp, Th* c);

template<class Dataset>
extern void computeIdealGasEOS_HydroStd(size_t firstParticle, size_t lastParticle, Dataset& d);

template<typename Dataset>
extern void computeIsothermalEOS_HydroStd(size_t firstParticle, size_t lastParticle, Dataset& d);

template<typename Dataset>
extern void computePolytropicEOS_HydroStd(size_t firstParticle, size_t lastParticle, Dataset& d);

} // namespace cuda

template<class Tc, class Thydro, class Tm1, class Tdu>
extern void driftPositionsGpu(const GroupView& grp, float dt, float dt_back,
                              util::array<float, Timestep::maxNumRungs> dt_m1, Tc* x, Tc* y, Tc* z, Thydro* vx,
                              Thydro* vy, Thydro* vz, const Tm1* x_m1, const Tm1* y_m1, const Tm1* z_m1,
                              const Thydro* ax, const Thydro* ay, const Thydro* az, const uint8_t* rung, Tc* temp,
                              Tc* u, Tdu* du, Tm1* du_m1, Thydro* mui, Tc gamma, Tc constCv);

template<class Tc, class Tv, class Ta, class Tdu, class Tm1, class Tu, class Thydro>
extern void computePositionsGpu(const GroupView& grp, float dt, util::array<float, Timestep::maxNumRungs> dt_m1, Tc* x,
                                Tc* y, Tc* z, Tv* vx, Tv* vy, Tv* vz, Tm1* x_m1, Tm1* y_m1, Tm1* z_m1, Ta* ax, Ta* ay,
                                Ta* az, const uint8_t* rung, Tu* temp, Tu* u, Tdu* du, Tm1* du_m1, Thydro* h,
                                Thydro* mui, Tc gamma, Tc constCv, const cstone::Box<Tc>& box);

template<class Th>
extern void updateSmoothingLengthGpu(const GroupView&, unsigned ng0, const unsigned* nc, Th* h);

template<class T>
extern void groupDivvTimestepGpu(float Krho, const GroupView&, const T* divv, float* groupDt);

template<class T>
extern void groupAccTimestepGpu(float etaAcc, const GroupView&, const T* ax, const T* ay, const T* az, float* groupDt);

void storeRungGpu(const GroupView& grp, uint8_t rung, uint8_t* particleRungs);

//! @brief max number of particles per group used in neighbor search for SPH
unsigned nsGroupSize();

} // namespace sph
