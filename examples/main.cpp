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
#include "example.h"

#define fscanf_s fscanf

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
  /* Squared distance computed by openGJK.                                 */
  gkFloat dd;
  /* Structure of simplex used by openGJK.                                 */
  gkSimplex s;
  /* Number of vertices defining body 1 and body 2, respectively.          */
  int nvrtx1, nvrtx2;
  /* Structures of body 1 and body 2, respectively.                        */
  gkPolytope bd1;
  gkPolytope bd2;
  /* Specify name of input files for body 1 and body 2, respectively.      */
  char inputfileA[40] = "../examples/userP.dat", inputfileB[40] = "../examples/userQ.dat";
  /* Pointers to vertices' coordinates of body 1 and body 2, respectively. */
  gkFloat* vrtx1 = NULL;
  gkFloat* vrtx2 = NULL;

  /* Import coordinates of object 1. */
  if (readinput(inputfileA, &vrtx1, &nvrtx1)) {
    return (1);
  }
  bd1.coord = vrtx1;
  bd1.numpoints = nvrtx1;

  /* Import coordinates of object 2. */
  if (readinput(inputfileB, &vrtx2, &nvrtx2)) {
    return (1);
  }
  bd2.coord = vrtx2;
  bd2.numpoints = nvrtx2;

  /* Initialise simplex as empty */
  s.nvrtx = 0;

  /* Allocate device memory */
  gkPolytope* d_bd1;
  gkPolytope* d_bd2;
  gkSimplex* d_s;
  gkFloat* d_distance;
  gkFloat* d_coord1;
  gkFloat* d_coord2;

  cudaMalloc(&d_bd1, sizeof(gkPolytope));
  cudaMalloc(&d_bd2, sizeof(gkPolytope));
  cudaMalloc(&d_s, sizeof(gkSimplex));
  cudaMalloc(&d_distance, sizeof(gkFloat));
  cudaMalloc(&d_coord1, nvrtx1 * 3 * sizeof(gkFloat));
  cudaMalloc(&d_coord2, nvrtx2 * 3 * sizeof(gkFloat));

  /* Copy coordinate data to device */
  cudaMemcpy(d_coord1, vrtx1, nvrtx1 * 3 * sizeof(gkFloat), cudaMemcpyHostToDevice);
  cudaMemcpy(d_coord2, vrtx2, nvrtx2 * 3 * sizeof(gkFloat), cudaMemcpyHostToDevice);

  /* Set up polytope structures with device pointers */
  bd1.coord = d_coord1;
  bd2.coord = d_coord2;
  cudaMemcpy(d_bd1, &bd1, sizeof(gkPolytope), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bd2, &bd2, sizeof(gkPolytope), cudaMemcpyHostToDevice);
  cudaMemcpy(d_s, &s, sizeof(gkSimplex), cudaMemcpyHostToDevice);

  /* Invoke the GJK procedure on GPU */
  launch_gjk_kernel(d_bd1, d_bd2, d_s, d_distance, 1);

  /* Copy results back */
  cudaMemcpy(&dd, d_distance, sizeof(gkFloat), cudaMemcpyDeviceToHost);
  cudaMemcpy(&s, d_s, sizeof(gkSimplex), cudaMemcpyDeviceToHost);

  /* Print distance between objects. */
  printf("Distance between bodies %f\n", dd);
  printf("Witnesses: (%f, %f, %f) and (%f, %f, %f)\n",
         s.witnesses[0][0], s.witnesses[0][1], s.witnesses[0][2],
         s.witnesses[1][0], s.witnesses[1][1], s.witnesses[1][2]);

  /* Free memory */
  free(vrtx1);
  free(vrtx2);
  cudaFree(d_bd1);
  cudaFree(d_bd2);
  cudaFree(d_s);
  cudaFree(d_distance);
  cudaFree(d_coord1);
  cudaFree(d_coord2);

  return (0);
}
