#pragma once

#if defined(USE_CUDA)
#include "sph/sph.cuh"
#endif

#include "sph/find_neighbors.hpp"

#include "sph/hydro_std/density.hpp"
#include "sph/hydro_std/iad.hpp"
#include "sph/hydro_std/momentum_energy.hpp"
#include "sph/kernels.hpp"
#include "sph/eos.hpp"
#include "sph/timestep.hpp"
#include "sph/positions.hpp"
#include "sph/total_energy.hpp"
#include "sph/update_h.hpp"

#include "sph/hydro_ve/av_switches.hpp"
#include "sph/hydro_ve/ve_def_gradh.hpp"
#include "sph/hydro_ve/iad_divv_curlv.hpp"
#include "sph/hydro_ve/momentum_energy.hpp"
#include "sph/hydro_ve/xmass.hpp"