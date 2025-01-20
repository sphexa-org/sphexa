/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich, University of Zurich, 2021 University of Basel
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Traits classes to manage GPU device acceleration behavior
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <type_traits>

namespace cstone
{

struct CpuTag
{
};
struct GpuTag
{
};

template<class AccType>
struct HaveGpu : public std::integral_constant<int, std::is_same_v<AccType, GpuTag>>
{
};

//! @brief The type member of this trait evaluates to CpuCaseType if Accelerator == CpuTag and GpuCaseType otherwise
template<class Accelerator, template<class...> class CpuCaseType, template<class...> class GpuCaseType, class = void>
struct AccelSwitchType
{
};

template<class Accelerator, template<class...> class CpuCaseType, template<class...> class GpuCaseType>
struct AccelSwitchType<Accelerator, CpuCaseType, GpuCaseType, std::enable_if_t<!HaveGpu<Accelerator>{}>>
{
    template<class... Args>
    using type = CpuCaseType<Args...>;
};

template<class Accelerator, template<class...> class CpuCaseType, template<class...> class GpuCaseType>
struct AccelSwitchType<Accelerator, CpuCaseType, GpuCaseType, std::enable_if_t<HaveGpu<Accelerator>{}>>
{
    template<class... Args>
    using type = GpuCaseType<Args...>;
};

template<class Accelerator, class CpuCaseType, class GpuCaseType, class = void>
struct AccelSwitchTypeSimple
{
};

template<class Accelerator, class CpuCaseType, class GpuCaseType>
struct AccelSwitchTypeSimple<Accelerator, CpuCaseType, GpuCaseType, std::enable_if_t<!HaveGpu<Accelerator>{}>>
{
    using type = CpuCaseType;
};

template<class Accelerator, class CpuCaseType, class GpuCaseType>
struct AccelSwitchTypeSimple<Accelerator, CpuCaseType, GpuCaseType, std::enable_if_t<HaveGpu<Accelerator>{}>>
{
    using type = GpuCaseType;
};

} // namespace cstone
