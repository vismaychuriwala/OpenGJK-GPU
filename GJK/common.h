#ifndef COMMON_H__
#define COMMON_H__

#include <float.h>

#define USE_32BITS 1

/*! @brief Precision of floating-point numbers.
 *
 * Default is set to 64-bit (Double). Change this to quickly play around with
 * 16- and 32-bit. */
#if USE_32BITS
#define gkFloat float
#define gkEpsilon FLT_EPSILON
#define gkSqrt sqrtf
#else
#define gkFloat double
#define gkEpsilon DBL_EPSILON
#define gkSqrt sqrt
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