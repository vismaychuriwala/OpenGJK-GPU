// This example demonstrates the use of the GPU GJK and EPA algorithms together.
// It shows how to use computeDistances followed by computeCollisionInformation
// to get detailed collision information including penetration depth and contact normals.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "examples/gpu/example.h"
#include "GJK/common.h"

/**
 * @brief Main program demonstrating GJK and EPA collision detection using GPU API.
 * 
 * This example creates two cubes:
 * - Cube 1: centered at (0, 0, 0), size 2x2x2 (8 vertices)
 * - Cube 2: rotated 45° around all axes, translated by (1, 0, 0), size 2x2x2 (8 vertices)
 * 
 * It first runs GJK to detect collision, then runs EPA to compute penetration depth
 * and contact information.
 */
int
main() {
  const int numVerts = 8;
  const gkFloat M_PI = 3.14159265358979323846;
  const gkFloat angle = 45.0f * M_PI / 180.0f; // 45 degrees in radians
  const gkFloat cos_a = cos(angle);
  const gkFloat sin_a = sin(angle);
  
  // Allocate memory for cube vertices (flattened array: 8 vertices * 3 coordinates)
  gkFloat* cube1_verts = (gkFloat*)malloc(numVerts * 3 * sizeof(gkFloat));
  gkFloat* cube2_verts = (gkFloat*)malloc(numVerts * 3 * sizeof(gkFloat));
  
  // Cube 1: centered at (0, 0, 0), size 2x2x2
  // Vertices: (-1,-1,-1), (-1,-1,1), (-1,1,-1), (-1,1,1), (1,-1,-1), (1,-1,1), (1,1,-1), (1,1,1)
  int idx = 0;
  for (int x = -1; x <= 1; x += 2) {
    for (int y = -1; y <= 1; y += 2) {
      for (int z = -1; z <= 1; z += 2) {
        cube1_verts[idx * 3 + 0] = (gkFloat)x;
        cube1_verts[idx * 3 + 1] = (gkFloat)y;
        cube1_verts[idx * 3 + 2] = (gkFloat)z;
        idx++;
      }
    }
  }
  
  // Cube 2: generate at origin first, then rotate and translate
  idx = 0;
  for (int x = -1; x <= 1; x += 2) {
    for (int y = -1; y <= 1; y += 2) {
      for (int z = -1; z <= 1; z += 2) {
        gkFloat px = (gkFloat)x;
        gkFloat py = (gkFloat)y;
        gkFloat pz = (gkFloat)z;
        
        // Rotate around X axis by 45°
        gkFloat temp_y = py * cos_a - pz * sin_a;
        gkFloat temp_z = py * sin_a + pz * cos_a;
        py = temp_y;
        pz = temp_z;
        
        // Rotate around Y axis by 45°
        gkFloat temp_x = px * cos_a + pz * sin_a;
        temp_z = -px * sin_a + pz * cos_a;
        px = temp_x;
        pz = temp_z;
        
        // Rotate around Z axis by 45°
        temp_x = px * cos_a - py * sin_a;
        temp_y = px * sin_a + py * cos_a;
        px = temp_x;
        py = temp_y;
        
        // Translate by (1, 0, 0)
        cube2_verts[idx * 3 + 0] = px + 1.0f;
        cube2_verts[idx * 3 + 1] = py;
        cube2_verts[idx * 3 + 2] = pz;
        idx++;
      }
    }
  }
  
  // Set up polytopes
  gkPolytope polytope1, polytope2;
  polytope1.numpoints = numVerts;
  polytope1.coord = cube1_verts;
  polytope2.numpoints = numVerts;
  polytope2.coord = cube2_verts;
  
  // Initialize simplex and distance
  gkSimplex simplex;
  simplex.nvrtx = 0;
  gkFloat distance;
  
  // Arrays for EPA results
  gkFloat witness1[3], witness2[3];
  gkFloat contact_normal[3];
  
  // Step 1: Run GJK to detect collision
  gkFloat distances[1];
  gkSimplex simplices[1] = {simplex};
  simplices[0].nvrtx = 0;
  
  GJK::GPU::computeDistances(1, &polytope1, &polytope2, simplices, distances);
  
  distance = distances[0];
  simplex = simplices[0];
  
  
  // Step 2: Run EPA to compute penetration depth and contact information
  GJK::GPU::computeCollisionInformation(1, &polytope1, &polytope2, simplices, distances, 
                                        witness1, witness2, contact_normal);
  
  distance = distances[0];
  simplex = simplices[0];
  
  printf("Penetration depth: %.6f\n", -distance);
  printf("Witness point on cube 1: (%.6f, %.6f, %.6f)\n", 
         witness1[0], witness1[1], witness1[2]);
  printf("Witness point on cube 2: (%.6f, %.6f, %.6f)\n", 
         witness2[0], witness2[1], witness2[2]);
  printf("Contact normal (from cube 1 to cube 2): (%.6f, %.6f, %.6f)\n", 
         contact_normal[0], contact_normal[1], contact_normal[2]);
  printf("\n");
  
  // Free memory
  free(cube1_verts);
  free(cube2_verts);
  
  return (0);
}

