// #pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include "../common.h"
//                           _____      _ _  __ //
//                          / ____|    | | |/ / //
//    ___  _ __   ___ _ __ | |  __     | | ' / //
//   / _ \| '_ \ / _ \ '_ \| | |_ |_   | |  < //
//  | (_) | |_) |  __/ | | | |__| | |__| | . \ //
//   \___/| .__/ \___|_| |_|\_____|\____/|_|\_\ //
//        | | //
//        |_| //
//                                                                                //
// Copyright 2022 Mattia Montanari, University of Oxford //
//                                                                               //
// This program is free software: you can redistribute it and/or modify it under
// // the terms of the GNU General Public License as published by the Free
// Software  // Foundation, either version 3 of the License. You should have
// received a copy   // of the GNU General Public License along with this
// program. If not, visit       //
//                                                                                //
//     https://www.gnu.org/licenses/ //
//                                                                                //
// This program is distributed in the hope that it will be useful, but WITHOUT
// // ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS  // FOR A PARTICULAR PURPOSE. See GNU General Public License for
// details.          //

/**
 * @file openGJK.h
 * @author Mattia Montanari
 * @date 1 Jan 2023
 * @brief Main interface of OpenGJK containing quick reference and API
 * documentation.
 *
 * @see https://www.mattiamontanari.com/opengjk/
 */

// #ifndef OPENGJK_H__
// #define OPENGJK_H__

/*! @brief Invoke the GJK algorithm to compute the minimum distance between two
 * polytopes.
 *
 * The simplex has to be initialised prior the call to this function. */
__global__ void compute_minimum_distance(gkPolytope* polytypes1, gkPolytope* polytypes2,
                                                gkSimplex* simplices, gkFloat* distances, int n);

// #endif  // OPENGJK_H__
