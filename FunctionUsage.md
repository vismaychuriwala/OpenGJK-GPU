
## Usage

### GJK Only (Distance Computation)

For computing minimum distances between polytopes without penetration depth:

```cpp
#include "GJK/gpu/warpParallelGJK.h"

// Prepare polytope data
gkPolytope polytope1, polytope2;
polytope1.numpoints = numVertices1;
polytope1.coord = vertexArray1;  // Flattened array: [x0, y0, z0, x1, y1, z1, ...]
polytope2.numpoints = numVertices2;
polytope2.coord = vertexArray2;

// Allocate result arrays
gkSimplex* simplices = (gkSimplex*)malloc(n * sizeof(gkSimplex));
gkFloat* distances = (gkFloat*)malloc(n * sizeof(gkFloat));

// Initialize simplices
for (int i = 0; i < n; i++) {
    simplices[i].nvrtx = 0;
}

// Run GJK (warp-parallel version recommended)
GJK::GPU::computeDistancesWarpParallel(n, &polytope1, &polytope2, simplices, distances);

// Results:
// - distances[i]: Minimum distance between polytopes (0.0 indicates collision)
// - simplices[i]: Contains witness points in simplices[i].witnesses[0] and simplices[i].witnesses[1]
```

**Available Functions:**
- `GJK::GPU::computeDistances()` - Standard GPU implementation (1 thread per collision)
- `GJK::GPU::computeDistancesWarpParallel()` - Warp-parallel implementation (16 threads per collision, recommended)

### GJK + EPA (Collision Detection with Penetration Depth)

For full collision detection including penetration depth and contact normals:

```cpp
#include "GJK/gpu/warpParallelGJK.h"

// Prepare polytope data (same as above)
gkPolytope polytope1, polytope2;
// ... initialize polytopes ...

// Allocate result arrays
gkSimplex* simplices = (gkSimplex*)malloc(n * sizeof(gkSimplex));
gkFloat* distances = (gkFloat*)malloc(n * sizeof(gkFloat));
gkFloat* witness1 = (gkFloat*)malloc(n * 3 * sizeof(gkFloat));  // 3 floats per collision
gkFloat* witness2 = (gkFloat*)malloc(n * 3 * sizeof(gkFloat));  // 3 floats per collision
gkFloat* contact_normals = (gkFloat*)malloc(n * 3 * sizeof(gkFloat));  // Optional: 3 floats per collision

// Initialize simplices
for (int i = 0; i < n; i++) {
    simplices[i].nvrtx = 0;
}

// Run GJK + EPA
GJK::GPU::computeGJKAndEPA(n, &polytope1, &polytope2, simplices, distances, 
                            witness1, witness2, contact_normals);

// Results:
// - distances[i]: 
//   * Positive value: Separation distance (polytopes are not colliding)
//   * Negative value: Penetration depth (polytopes are overlapping, magnitude is depth)
//   * Zero: Polytopes are just touching
// - witness1[i*3 + 0/1/2]: Contact point on polytope1 (x, y, z)
// - witness2[i*3 + 0/1/2]: Contact point on polytope2 (x, y, z)
// - contact_normals[i*3 + 0/1/2]: Contact normal pointing from polytope1 to polytope2
//   * For polytope2's normal, use: -contact_normals[i*3 + 0/1/2]
```

### Example: Single Collision Check

```cpp
// Single polytope pair
gkPolytope poly1, poly2;
// ... initialize polytopes ...

gkSimplex simplex;
simplex.nvrtx = 0;
gkFloat distance;
gkFloat witness1[3], witness2[3];
gkFloat contact_normal[3];

GJK::GPU::computeGJKAndEPA(1, &poly1, &poly2, &simplex, &distance, 
                            witness1, witness2, contact_normal);

if (distance < 0.0f) {
    printf("Collision detected! Penetration depth: %.6f\n", -distance);
    printf("Contact point on poly1: (%.3f, %.3f, %.3f)\n", 
           witness1[0], witness1[1], witness1[2]);
    printf("Contact point on poly2: (%.3f, %.3f, %.3f)\n", 
           witness2[0], witness2[1], witness2[2]);
    printf("Contact normal: (%.3f, %.3f, %.3f)\n", 
           contact_normal[0], contact_normal[1], contact_normal[2]);
} else {
    printf("No collision. Separation distance: %.6f\n", distance);
}
```