
#pragma once

#ifdef __HIPCC__

#include <hipcub/hipcub.hpp>

namespace cub = hipcub;

#else

#include <cub/cub.cuh>

#endif
