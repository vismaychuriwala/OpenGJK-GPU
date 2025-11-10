//                           _____      _ _  __                                   //
//                          / ____|    | | |/ /                                   //
//    ___  _ __   ___ _ __ | |  __     | | ' /                                    //
//   / _ \| '_ \ / _ \ '_ \| | |_ |_   | |  <                                     //
//  | (_) | |_) |  __/ | | | |__| | |__| | . \                                    //
//   \___/| .__/ \___|_| |_|\_____|\____/|_|\_\                                   //
//        | |                                                                     //
//        |_|                                                                     //
//                                                                                //
// Copyright 2022 Mattia Montanari, University of Oxford                          //
//                                                                                //
// This program is free software: you can redistribute it and/or modify it under  //
// the terms of the GNU General Public License as published by the Free Software  //
// Foundation, either version 3 of the License. You should have received a copy   //
// of the GNU General Public License along with this program. If not, visit       //
//                                                                                //
//     https://www.gnu.org/licenses/                                              //
//                                                                                //
// This program is distributed in the hope that it will be useful, but WITHOUT    //
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS  //
// FOR A PARTICULAR PURPOSE. See GNU General Public License for details.          //

/// @author Mattia Montanari
/// @date July 2022

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include "examples/gpu/example.h"
#include "examples/cpu/example.h"

#define fscanf_s fscanf
#define M_PI 3.14159265358979323846  /* pi */

// Test configuration
#define NUM_POLYTOPES 1000
#define VERTS_PER_POLYTOPE 1000

/// @brief Function for reading input file with body's coordinates (flattened array version).
int
readinput(const char* inputfile, gkFloat** pts, int* out) {
  int npoints = 0;
  int idx = 0;
  FILE* fp;

  /* Open file. */
#ifdef WIN32
  errno_t err;
  if ((err = fopen_s(&fp, inputfile, "r")) != 0) {
#else
  if ((fp = fopen(inputfile, "r")) == NULL) {
#endif
    fprintf(stdout, "ERROR: input file %s not found!\n", inputfile);
    fprintf(stdout, "  -> The file must be in the folder from which this "
                    "program is launched\n\n");
    return 1;
  }

  /* Read number of input vertices. */
  if (fscanf_s(fp, "%d", &npoints) != 1) {
    return 1;
  }

  /* Allocate memory as flattened array. */
  gkFloat* arr = (gkFloat*)malloc(npoints * 3 * sizeof(gkFloat));

  /* Read and store vertices' coordinates. */
  for (idx = 0; idx < npoints; idx++) {
#ifdef USE_32BITS
    if (fscanf_s(fp, "%f %f %f\n", &arr[idx*3 + 0], &arr[idx*3 + 1], &arr[idx*3 + 2]) != 3) {
      return 1;
    }
#else
    if (fscanf_s(fp, "%lf %lf %lf\n", &arr[idx*3 + 0], &arr[idx*3 + 1], &arr[idx*3 + 2]) != 3) {
      return 1;
    }
#endif
  }
  fclose(fp);

  *pts = arr;
  *out = idx;

  return (0);
}

/// @brief Generate a polytope with specified number of vertices and random offset
gkFloat* generatePolytope(int numVerts, gkFloat offsetX, gkFloat offsetY, gkFloat offsetZ) {
  gkFloat* verts = (gkFloat*)malloc(numVerts * 3 * sizeof(gkFloat));

  // Generate random points on a sphere (simple polytope generation)
  for (int i = 0; i < numVerts; i++) {
    float theta = (float)rand() / RAND_MAX * 2.0f * M_PI;
    float phi = (float)rand() / RAND_MAX * M_PI;
    float r = 1.0f + (float)rand() / RAND_MAX * 0.5f;

    verts[i*3 + 0] = r * sin(phi) * cos(theta) + offsetX;
    verts[i*3 + 1] = r * sin(phi) * sin(theta) + offsetY;
    verts[i*3 + 2] = r * cos(phi) + offsetZ;
  }

  return verts;
}

/**
 * @brief Main program - CUDA version with unique polytopes.
 *
 */
int
main() {
  printf("OpenGJK Performance Testing\n");
  printf("============================\n");
  printf("Polytopes: %d\n", NUM_POLYTOPES);
  printf("Vertices per polytope: %d\n\n", VERTS_PER_POLYTOPE);

  /* Allocate arrays for all polytope vertex data */
  gkFloat** vrtx1_array = (gkFloat**)malloc(NUM_POLYTOPES * sizeof(gkFloat*));
  gkFloat** vrtx2_array = (gkFloat**)malloc(NUM_POLYTOPES * sizeof(gkFloat*));

  /* Generate unique polytopes with random offsets */
  for (int i = 0; i < NUM_POLYTOPES; i++) {
    // Random offset for each polytope pair
    gkFloat offset1_x = ((gkFloat)rand() / RAND_MAX - 0.5f) * 10.0f;
    gkFloat offset1_y = ((gkFloat)rand() / RAND_MAX - 0.5f) * 10.0f;
    gkFloat offset1_z = ((gkFloat)rand() / RAND_MAX - 0.5f) * 10.0f;

    gkFloat offset2_x = ((gkFloat)rand() / RAND_MAX - 0.5f) * 10.0f;
    gkFloat offset2_y = ((gkFloat)rand() / RAND_MAX - 0.5f) * 10.0f;
    gkFloat offset2_z = ((gkFloat)rand() / RAND_MAX - 0.5f) * 10.0f;

    vrtx1_array[i] = generatePolytope(VERTS_PER_POLYTOPE, offset1_x, offset1_y, offset1_z);
    vrtx2_array[i] = generatePolytope(VERTS_PER_POLYTOPE, offset2_x, offset2_y, offset2_z);
  }

  /* Allocate arrays for polytope pairs */
  gkPolytope* polytopes1 = (gkPolytope*)malloc(NUM_POLYTOPES * sizeof(gkPolytope));
  gkPolytope* polytopes2 = (gkPolytope*)malloc(NUM_POLYTOPES * sizeof(gkPolytope));
  gkSimplex* simplices = (gkSimplex*)malloc(NUM_POLYTOPES * sizeof(gkSimplex));
  gkFloat* distances = (gkFloat*)malloc(NUM_POLYTOPES * sizeof(gkFloat));
  gkFloat* gpu_distances = (gkFloat*)malloc(NUM_POLYTOPES * sizeof(gkFloat));

  /* Initialize polytope pairs with unique data */
  for (int i = 0; i < NUM_POLYTOPES; i++) {
    polytopes1[i].numpoints = VERTS_PER_POLYTOPE;
    polytopes1[i].coord = vrtx1_array[i];

    polytopes2[i].numpoints = VERTS_PER_POLYTOPE;
    polytopes2[i].coord = vrtx2_array[i];

    simplices[i].nvrtx = 0;  // Initialize simplex as empty
  }

  /* Invoke the GJK procedure on GPU for all pairs */
  GJK::GPU::computeDistances(NUM_POLYTOPES, polytopes1, polytopes2, simplices, gpu_distances);

  /* Print GPU results */
  printf("GPU time: %.4f ms\n", GJK::GPU::timer().getGpuElapsedTimeForPreviousOperation());
  printf("GPU distance (first pair): %.6f\n", gpu_distances[0]);
  printf("GPU distance (last pair): %.6f\n", gpu_distances[NUM_POLYTOPES - 1]);
  printf("GPU witnesses (first pair): (%.3f, %.3f, %.3f) and (%.3f, %.3f, %.3f)\n\n",
         simplices[0].witnesses[0][0], simplices[0].witnesses[0][1], simplices[0].witnesses[0][2],
         simplices[0].witnesses[1][0], simplices[0].witnesses[1][1], simplices[0].witnesses[1][2]);

  /* Reset simplices for CPU run */
  for (int i = 0; i < NUM_POLYTOPES; i++) {
    simplices[i].nvrtx = 0;
  }

  /* Invoke the GJK procedure on CPU for all pairs */
  GJK::CPU::computeDistances(NUM_POLYTOPES, polytopes1, polytopes2, simplices, distances);

  /* Print CPU results */
  printf("CPU time: %.4f ms\n", GJK::CPU::timer().getCpuElapsedTimeForPreviousOperation());
  printf("CPU distance (first pair): %.6f\n", distances[0]);
  printf("CPU distance (last pair): %.6f\n", distances[NUM_POLYTOPES - 1]);
  printf("CPU witnesses (first pair): (%.3f, %.3f, %.3f) and (%.3f, %.3f, %.3f)\n\n",
         simplices[0].witnesses[0][0], simplices[0].witnesses[0][1], simplices[0].witnesses[0][2],
         simplices[0].witnesses[1][0], simplices[0].witnesses[1][1], simplices[0].witnesses[1][2]);

  /* Print speedup */
  float speedup = GJK::CPU::timer().getCpuElapsedTimeForPreviousOperation() /
                  GJK::GPU::timer().getGpuElapsedTimeForPreviousOperation();
  printf("Speedup: %.2fx\n\n", speedup);

  /* Validate results - compare first 100 distances */
  int test_count = (NUM_POLYTOPES < 100) ? NUM_POLYTOPES : 100;
  bool all_passed = true;
  const gkFloat tolerance = 1e-5f;

  for (int i = 0; i < test_count; i++) {
    gkFloat diff = fabs(gpu_distances[i] - distances[i]);
    if (diff > tolerance) {
      all_passed = false;
      printf("Mismatch at index %d: GPU=%.6f, CPU=%.6f, diff=%.6e\n",
             i, gpu_distances[i], distances[i], diff);
    }
  }

  if (all_passed) {
    printf("\033[32mValidation PASSED\033[0m: First %d results match within tolerance (%.0e)\n",
           test_count, tolerance);
  } else {
    printf("\033[31mValidation FAILED\033[0m: Some results do not match\n");
  }

  /* Free all allocated memory */
  for (int i = 0; i < NUM_POLYTOPES; i++) {
    free(vrtx1_array[i]);
    free(vrtx2_array[i]);
  }
  free(vrtx1_array);
  free(vrtx2_array);
  free(polytopes1);
  free(polytopes2);
  free(simplices);
  free(distances);
  free(gpu_distances);

  printf("\nTesting complete!\n");
  return (0);
}
