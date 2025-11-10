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

#include "GJK/openGJK.h"
#include "examples/gpu/example.h"

#define fscanf_s fscanf
#define NUM_POLYTOPE_PAIRS 10000  // Number of polytope pairs to test

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

/**
 * @brief Main program - CUDA version with minimal changes.
 *
 */
int
main() {
  /* Number of vertices defining body 1 and body 2, respectively.          */
  int nvrtx1, nvrtx2;
  /* Specify name of input files for body 1 and body 2, respectively.      */
  char inputfileA[40] = "../examples/userP.dat", inputfileB[40] = "../examples/userQ.dat";
  /* Pointers to vertices' coordinates of body 1 and body 2, respectively. */
  gkFloat* vrtx1_base = NULL;
  gkFloat* vrtx2_base = NULL;

  /* Import base coordinates from files */
  if (readinput(inputfileA, &vrtx1_base, &nvrtx1)) {
    return (1);
  }
  if (readinput(inputfileB, &vrtx2_base, &nvrtx2)) {
    return (1);
  }

  /* Allocate arrays for multiple polytope pairs */
  gkPolytope* polytopes1 = (gkPolytope*)malloc(NUM_POLYTOPE_PAIRS * sizeof(gkPolytope));
  gkPolytope* polytopes2 = (gkPolytope*)malloc(NUM_POLYTOPE_PAIRS * sizeof(gkPolytope));
  gkSimplex* simplices = (gkSimplex*)malloc(NUM_POLYTOPE_PAIRS * sizeof(gkSimplex));
  gkFloat* distances = (gkFloat*)malloc(NUM_POLYTOPE_PAIRS * sizeof(gkFloat));

  /* Replicate the base polytope data for each pair */
  for (int i = 0; i < NUM_POLYTOPE_PAIRS; i++) {
    polytopes1[i].numpoints = nvrtx1;
    polytopes1[i].coord = vrtx1_base;  // All point to same base data

    polytopes2[i].numpoints = nvrtx2;
    polytopes2[i].coord = vrtx2_base;  // All point to same base data

    simplices[i].nvrtx = 0;  // Initialize simplex as empty
  }

  /* Invoke the GJK procedure on GPU for all pairs */
  GJK::GPU::computeDistances(NUM_POLYTOPE_PAIRS, polytopes1, polytopes2, simplices, distances);

  /* Print results for first pair only */
  printf("Testing %d polytope pairs\n", NUM_POLYTOPE_PAIRS);
  printf("Distance between bodies (first pair): %f\n", distances[0]);
  printf("GPU time: %.4f ms\n", GJK::GPU::timer().getGpuElapsedTimeForPreviousOperation());
  printf("Witnesses (first pair): (%f, %f, %f) and (%f, %f, %f)\n",
         simplices[0].witnesses[0][0], simplices[0].witnesses[0][1], simplices[0].witnesses[0][2],
         simplices[0].witnesses[1][0], simplices[0].witnesses[1][1], simplices[0].witnesses[1][2]);

  /* Free memory */
  free(vrtx1_base);
  free(vrtx2_base);
  free(polytopes1);
  free(polytopes2);
  free(simplices);
  free(distances);

  return (0);
}
