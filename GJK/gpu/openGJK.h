#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include "../common.h"

#ifndef OPENGJK_H__
#define OPENGJK_H__

/*! @brief Invoke the warp-parallel GJK algorithm to compute the minimum distance between two
 * polytopes using 16 threads per collision.
 *
 * The simplex has to be initialised prior to the call to this function. */
__global__ void compute_minimum_distance(const gkPolytope* polytypes1, const gkPolytope* polytypes2,
  gkSimplex* simplices, gkFloat* distances, const int n);

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
 * @param contact_normals Output array for contact normals (3 floats per collision, points from polytope1 to polytope2)
 * @param n Number of polytope pairs to process
 */
__global__ void compute_epa(const gkPolytope* polytopes1, const gkPolytope* polytopes2,
  gkSimplex* simplices, gkFloat* distances, gkFloat* witness1, gkFloat* witness2, gkFloat* contact_normals, const int n);

#endif  // OPENGJK_H__
