/*
 *                          _____      _ _  __
 *                         / ____|    | | |/ /
 *   ___  _ __   ___ _ __ | |  __     | | ' /
 *  / _ \| '_ \ / _ \ '_ \| | |_ |_   | |  <
 * | (_) | |_) |  __/ | | | |__| | |__| | . \
 *  \___/| .__/ \___|_| |_|\_____|\____/|_|\_\
 *       | |
 *       |_|
 *
 * Copyright 2022-2026 Mattia Montanari, University of Oxford
 * Copyright 2025-2026 Vismay Churiwala, Marcus Hedlund
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3. See https://www.gnu.org/licenses/
 */

/**
 * @file common.h
 * @author Mattia Montanari, Vismay Churiwala, Marcus Hedlund
 * @date 22 Jan 2026
 * @brief GPU (CUDA) implementation of OpenGJK - Common
 * data structure definitions for CPU and GPU algorithms.
 *
 * CUDA implementation with warp-level parallelism for
 * high-performance collision detection on NVIDIA GPUs.
 *
 * @see https://github.com/vismaychuriwala/OpenGJK-GPU
 * @see https://www.mattiamontanari.com/opengjk/
 */

#ifndef COMMON_H__
#define COMMON_H__

#include <float.h>

/*! @brief Precision of floating-point numbers.
 *
 * Default is set to 64-bit (Double). Comment/uncomment the line below to switch
 * between 32-bit and 64-bit precision. */
#define USE_32BITS  // Comment this line to use 64-bit (double) precision

#ifdef USE_32BITS
#define gkFloat float
#define gkEpsilon FLT_EPSILON
#define gkSqrt sqrtf
#define gkFmax fmaxf
#else
#define gkFloat double
#define gkEpsilon DBL_EPSILON
#define gkSqrt sqrt
#define gkFmax fmax
#endif

/*! @brief Data structure for convex polytopes.
 *
 * Polytopes are three-dimensional shapes and the GJK algorithm works directly
 * on their convex-hull. However the convex-hull is never computed explicitly,
 * instead each GJK-iteration employs a support function that has a cost
 * linearly dependent on the number of points defining the polytope. */
typedef struct gkPolytope{
  int numpoints; /*!< Number of points defining the polytope. */
  gkFloat s[3]; /*!< Furthest point returned by the support function and updated
                   at each GJK-iteration. For the first iteration this value is
                   a guess - and this guess not irrelevant. */
  int s_idx; /*!< Index of the furthest point returned by the support function.
              */
  gkFloat* coord; /*!< Coordinates of the points of the polytope. This is owned
                      by user who manages and garbage-collects the memory for
                      these coordinates. */
} gkPolytope;

/*! @brief Data structure for simplex.
 *
 * The simplex is updated at each GJK-iteration. For the first iteration this
 * value is a guess - and this guess not irrelevant. */
typedef struct gkSimplex{
  int nvrtx;               /*!< Number of points defining the simplex. */
  gkFloat vrtx[4][3];      /*!< Coordinates of the points of the simplex. */
  int vrtx_idx[4][2];      /*!< Indices of the points of the simplex. */
  gkFloat witnesses[2][3]; /*!< Coordinates of the witness points. */
} gkSimplex;

#endif  // COMMON_H__