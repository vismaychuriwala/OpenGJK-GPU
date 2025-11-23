#pragma once

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

/*! @brief Invoke the warp-parallel EPA algorithm to compute penetration depth and witness points
 * for colliding polytopes using 32 threads (one warp) per collision.
 *
 * This should be called after GJK when a collision is detected (simplex has 4 vertices).
 * The function expands the GJK simplex into a full polytope to find the closest points
 * on the surfaces of the two polytopes.
 *
 * @param polytopes1 First set of polytopes
 * @param polytopes2 Second set of polytopes
 * @param simplices Simplex results from GJK (should have 4 vertices for collisions)
 * @param distances Output array for penetration depths (or distances if no collision)
 * @param witness1 Output array for witness points on first polytope (3 floats per collision)
 * @param witness2 Output array for witness points on second polytope (3 floats per collision)
 * @param n Number of polytope pairs to process
 */
__global__ void compute_epa_warp_parallel(gkPolytope* polytopes1, gkPolytope* polytopes2,
  gkSimplex* simplices, gkFloat* distances, gkFloat* witness1, gkFloat* witness2, int n);

#endif  // WARPPARALLELOPENGJK_H__
