/*
 * Ryoanji N-body solver
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Ewald correction to gravity due to periodic boundaries
 *
 * @author Jonathan Coles <jonathan.coles@cscs.ch>
 */

#pragma once

#include "ryoanji/nbody/types.h"

namespace ryoanji
{

//! @brief Ewald input settings with recommended defaults
struct EwaldSettings
{
    int    numReplicaShells     = 1;
    double lCut                 = 2.6;
    double hCut                 = 2.8;
    double alpha_scale          = 2.0;
    double small_R_scale_factor = 3.0e-3; // Gasoline. PKDGrav3, ChaNGa: 1.2e-3
};

} // namespace ryoanji
