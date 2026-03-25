
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
 * @author Marcus Headlund and Vismay Churiwala
 * @date 1 Jan 2026
 * @brief Main interface of the EPA algorithm.
 *
 */

#ifndef EPA_CPU_H__
#define EPA_CPU_H__

#include "../common.h"

#ifdef __cplusplus
#define restrict
#endif

#ifdef __cplusplus
extern "C" {
#endif
/*! @brief Invoke the EPA algorithm to compute the collision information between two colliding
 * polytopes.
 *
 * The simplex has to be initialised prior the call to this function. */
  void compute_epa(
  const gkPolytope* bd1,
  const gkPolytope* bd2,
  gkSimplex* simplex,
  gkFloat* distance,
  gkFloat witness1[3],
  gkFloat witness2[3],
  gkFloat contact_normal[3]);

#ifdef __cplusplus
}
#endif
#endif  // EPA_CPU_H__
