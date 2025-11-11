#pragma once

#define USE_32BITS 1

#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include "openGJK.h"

#ifndef WARPPARALLELOPENGJK_H__
#define WARPPARALLELOPENGJK_H__

/*! @brief Invoke the warp-parallel GJK algorithm to compute the minimum distance between two
 * polytopes using 16 threads per collision.
 *
 * The simplex has to be initialised prior to the call to this function. */
__global__ void compute_minimum_distance_warp_parallel(gkPolytope* polytypes1, gkPolytope* polytypes2,
  gkSimplex* simplices, gkFloat* distances, int n);

#endif  // WARPPARALLELOPENGJK_H__
