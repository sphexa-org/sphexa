#pragma once

#ifdef USE_MPI
#include "mpi.h"
#endif

#include "Task.hpp"
#include "ArgParser.hpp"
#include "Timer.hpp"
#include "FileUtils.hpp"
#include "Printer.hpp"
#include "utils.hpp"

#if defined(USE_CUDA)
#include "sph/cuda/sph.cuh"
#endif

#include "sph/density.hpp"
#include "sph/IAD.hpp"
#include "sph/momentumAndEnergyIAD.hpp"
#include "sph/kernels.hpp"
#include "sph/equationOfState.hpp"
#include "sph/timestep.hpp"
#include "sph/positions.hpp"
#include "sph/totalEnergy.hpp"
#include "sph/updateSmoothingLength.hpp"
