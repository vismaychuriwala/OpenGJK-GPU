#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <string>

#include "examples/gpu/example.h"
#include "examples/cpu/example.h"

#define EPA_TESTING 1

#define fscanf_s fscanf
#define M_PI 3.14159265358979323846  /* pi */

#define NUM_POLYTOPES 1000
#define RANDOM_VERTS 0

#if RANDOM_VERTS
  #define MIN_VERTS 100
  #define MAX_VERTS 2000
#else
  #define VERTS_PER_POLYTOPE 1000
#endif

#define TEST_API 1
#define SAVE_PERFORMANCE_DATA_TO_FILE 0
#define OUTPUT_FILE "../data/gpu_performance_results_!.csv"

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
  srand((unsigned int)time(NULL));

  printf("OpenGJK Performance Testing\n");
  printf("============================\n");
  printf("Polytopes: %d\n", NUM_POLYTOPES);
#if RANDOM_VERTS
  printf("Vertices per polytope: %d-%d (randomized)\n", MIN_VERTS, MAX_VERTS);
#else
  printf("Vertices per polytope: %d\n", VERTS_PER_POLYTOPE);
#endif
#ifdef USE_32BITS
  printf("Precision: 32-bit (float)\n\n");
#else
  printf("Precision: 64-bit (double)\n\n");
#endif

  gkFloat** vrtx1_array = (gkFloat**)malloc(NUM_POLYTOPES * sizeof(gkFloat*));
  gkFloat** vrtx2_array = (gkFloat**)malloc(NUM_POLYTOPES * sizeof(gkFloat*));
  int* nverts1 = (int*)malloc(NUM_POLYTOPES * sizeof(int));
  int* nverts2 = (int*)malloc(NUM_POLYTOPES * sizeof(int));

  for (int i = 0; i < NUM_POLYTOPES; i++) {
#if RANDOM_VERTS
    nverts1[i] = MIN_VERTS + rand() % (MAX_VERTS - MIN_VERTS + 1);
    nverts2[i] = MIN_VERTS + rand() % (MAX_VERTS - MIN_VERTS + 1);
#else
    nverts1[i] = VERTS_PER_POLYTOPE;
    nverts2[i] = VERTS_PER_POLYTOPE;
#endif

    gkFloat offset1_x = ((gkFloat)rand() / RAND_MAX - 0.5f) * 10.0f;
    gkFloat offset1_y = ((gkFloat)rand() / RAND_MAX - 0.5f) * 10.0f;
    gkFloat offset1_z = ((gkFloat)rand() / RAND_MAX - 0.5f) * 10.0f;

    gkFloat offset2_x = ((gkFloat)rand() / RAND_MAX - 0.5f) * 10.0f;
    gkFloat offset2_y = ((gkFloat)rand() / RAND_MAX - 0.5f) * 10.0f;
    gkFloat offset2_z = ((gkFloat)rand() / RAND_MAX - 0.5f) * 10.0f;

    vrtx1_array[i] = generatePolytope(nverts1[i], offset1_x, offset1_y, offset1_z);
    vrtx2_array[i] = generatePolytope(nverts2[i], offset2_x, offset2_y, offset2_z);
  }

  gkPolytope* polytopes1 = (gkPolytope*)malloc(NUM_POLYTOPES * sizeof(gkPolytope));
  gkPolytope* polytopes2 = (gkPolytope*)malloc(NUM_POLYTOPES * sizeof(gkPolytope));
  gkSimplex* simplices = (gkSimplex*)malloc(NUM_POLYTOPES * sizeof(gkSimplex));
  gkSimplex* gpu_simplices = (gkSimplex*)malloc(NUM_POLYTOPES * sizeof(gkSimplex));
  gkSimplex* warm_up_gpu_simplices = (gkSimplex*)malloc(NUM_POLYTOPES * sizeof(gkSimplex));
  gkFloat* distances = (gkFloat*)malloc(NUM_POLYTOPES * sizeof(gkFloat));
  gkFloat* gpu_distances = (gkFloat*)malloc(NUM_POLYTOPES * sizeof(gkFloat));
  gkFloat* warm_up_gpu_distances = (gkFloat*)malloc(NUM_POLYTOPES * sizeof(gkFloat));

  for (int i = 0; i < NUM_POLYTOPES; i++) {
    polytopes1[i].numpoints = nverts1[i];
    polytopes1[i].coord = vrtx1_array[i];

    polytopes2[i].numpoints = nverts2[i];
    polytopes2[i].coord = vrtx2_array[i];

    simplices[i].nvrtx = 0;
    gpu_simplices[i].nvrtx = 0;
    warm_up_gpu_simplices[i].nvrtx = 0;
  }

  /* Warm up GPU to reduce measurement discrepancy */
  GJK::GPU::computeDistances(NUM_POLYTOPES, polytopes1, polytopes2, warm_up_gpu_simplices, warm_up_gpu_distances);
  float warm_up_gpu_time = GJK::GPU::timer().getGpuElapsedTimeForPreviousOperation();

  /* Invoke the GJK procedure on GPU for all pairs */
  GJK::GPU::computeDistances(NUM_POLYTOPES, polytopes1, polytopes2, gpu_simplices, gpu_distances);
  float gpu_time = GJK::GPU::timer().getGpuElapsedTimeForPreviousOperation();

  /* Reset simplices for CPU run */
  for (int i = 0; i < NUM_POLYTOPES; i++) {
    simplices[i].nvrtx = 0;
  }

  /* Invoke the GJK procedure on CPU for all pairs */
  GJK::CPU::computeDistances(NUM_POLYTOPES, polytopes1, polytopes2, simplices, distances);
  float cpu_time = GJK::CPU::timer().getCpuElapsedTimeForPreviousOperation();

  /* Validate results first to determine coloring */
  int test_count = (NUM_POLYTOPES < 100) ? NUM_POLYTOPES : 100;
  bool gpu_passed = true;
  const gkFloat tolerance = 1e-5f;

  /* Validate GPU vs CPU */
  for (int i = 0; i < test_count; i++) {
    gkFloat diff = fabs(gpu_distances[i] - distances[i]);
    if (diff > tolerance) {
      gpu_passed = false;
      break;
    }
  }

  /* Print execution times with color based on validation */
  printf("\n");
  printf("================================================================================\n");
  printf("                           EXECUTION TIMES                                      \n");
  printf("================================================================================\n");
  printf("GPU:                       %s%.4f ms\033[0m\n", gpu_passed ? "\033[36m" : "\033[31m", gpu_time);
  printf("CPU:                       \033[36m%.4f ms\033[0m\n", cpu_time);

  printf("\n");
  printf("================================================================================\n");
  printf("                          PERFORMANCE COMPARISON                                \n");
  printf("================================================================================\n");

  /* Print speedup comparison */
  float speedup = cpu_time / gpu_time;

  printf("CPU vs GPU:                ");
  if (speedup > 1.0f) {
    printf("\033[32m%.2fx speedup\033[0m\n", speedup);
  } else {
    printf("\033[33m%.2fx slowdown\033[0m\n", speedup);
  }

  printf("\n");
  printf("================================================================================\n");
  printf("                            VALIDATION RESULTS                                  \n");
  printf("================================================================================\n");

  /* Print detailed validation results */
  printf("GPU vs CPU:                ");
  bool has_errors = false;
  for (int i = 0; i < test_count; i++) {
    gkFloat diff = fabs(gpu_distances[i] - distances[i]);
    if (diff > tolerance) {
      if (!has_errors) printf("\n"); // Move to new line for error details
      has_errors = true;
      printf("  Mismatch at index %d: GPU=%.6f, CPU=%.6f, diff=%.6e\n",
             i, gpu_distances[i], distances[i], diff);
    }
  }

  if (gpu_passed) {
    printf("\033[32mPASSED\033[0m (first %d results within %.0e tolerance)\n",
           test_count, tolerance);
  } else {
    printf("\033[31mFAILED\033[0m\n");
  }

  printf("\n");
  printf("================================================================================\n");
  printf("                            DISTANCE RESULTS                                    \n");
  printf("================================================================================\n");
  printf("GPU:\n");
  printf("  Distance (first pair):   %.6f\n", gpu_distances[0]);
  printf("  Distance (last pair):    %.6f\n", gpu_distances[NUM_POLYTOPES - 1]);
  printf("  Witnesses (first pair):  (%.3f, %.3f, %.3f) and (%.3f, %.3f, %.3f)\n",
         gpu_simplices[0].witnesses[0][0], gpu_simplices[0].witnesses[0][1], gpu_simplices[0].witnesses[0][2],
         gpu_simplices[0].witnesses[1][0], gpu_simplices[0].witnesses[1][1], gpu_simplices[0].witnesses[1][2]);

  printf("\nCPU:\n");
  printf("  Distance (first pair):   %.6f\n", distances[0]);
  printf("  Distance (last pair):    %.6f\n", distances[NUM_POLYTOPES - 1]);
  printf("  Witnesses (first pair):  (%.3f, %.3f, %.3f) and (%.3f, %.3f, %.3f)\n",
         simplices[0].witnesses[0][0], simplices[0].witnesses[0][1], simplices[0].witnesses[0][2],
         simplices[0].witnesses[1][0], simplices[0].witnesses[1][1], simplices[0].witnesses[1][2]);
  printf("================================================================================\n");

#if TEST_API
  printf("\n");
  printf("================================================================================\n");
  printf("                        TESTING HIGH-LEVEL API                                  \n");
  printf("================================================================================\n");

  // --- Test 1: computeDistances ---
  {
    gkSimplex* t_simplices = (gkSimplex*)malloc(NUM_POLYTOPES * sizeof(gkSimplex));
    gkFloat*   t_distances = (gkFloat*)malloc(NUM_POLYTOPES * sizeof(gkFloat));
    for (int i = 0; i < NUM_POLYTOPES; i++) t_simplices[i].nvrtx = 0;

    GJK::GPU::computeDistances(NUM_POLYTOPES, polytopes1, polytopes2, t_simplices, t_distances);
    float t1_time = GJK::GPU::timer().getGpuElapsedTimeForPreviousOperation();

    bool passed = true;
    for (int i = 0; i < test_count; i++) {
      if (fabs(t_distances[i] - gpu_distances[i]) > tolerance) { passed = false; break; }
    }
    printf("computeDistances:                      %s  %.4f ms\n", passed ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m", t1_time);

    free(t_simplices);
    free(t_distances);
  }

  // --- Test: compute_minimum_distance_indexed ---
  {
    gkPolytope*      idx_polytopes = (gkPolytope*)malloc(2 * NUM_POLYTOPES * sizeof(gkPolytope));
    gkCollisionPair* idx_pairs     = (gkCollisionPair*)malloc(NUM_POLYTOPES * sizeof(gkCollisionPair));
    gkSimplex*       idx_simplices = (gkSimplex*)malloc(NUM_POLYTOPES * sizeof(gkSimplex));
    gkFloat*         idx_distances = (gkFloat*)malloc(NUM_POLYTOPES * sizeof(gkFloat));
    for (int i = 0; i < NUM_POLYTOPES; i++) {
      idx_polytopes[2*i]   = polytopes1[i];
      idx_polytopes[2*i+1] = polytopes2[i];
      idx_pairs[i].idx1 = 2*i;
      idx_pairs[i].idx2 = 2*i+1;
      idx_simplices[i].nvrtx = 0;
    }

    GJK::GPU::timer().startGpuTimer();
    compute_minimum_distance_indexed(2 * NUM_POLYTOPES, NUM_POLYTOPES,
                                     idx_polytopes, idx_pairs,
                                     idx_simplices, idx_distances);
    GJK::GPU::timer().endGpuTimer();
    float t_idx_time = GJK::GPU::timer().getGpuElapsedTimeForPreviousOperation();

    bool passed = true;
    for (int i = 0; i < test_count; i++) {
      if (fabs(idx_distances[i] - gpu_distances[i]) > tolerance) { passed = false; break; }
    }
    printf("compute_minimum_distance_indexed:      %s  %.4f ms\n", passed ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m", t_idx_time);

    free(idx_polytopes);
    free(idx_pairs);
    free(idx_simplices);
    free(idx_distances);
  }

  // --- Test: compute_minimum_distance_indexed_device ---
  {
    gkPolytope*      idx_polytopes = (gkPolytope*)malloc(2 * NUM_POLYTOPES * sizeof(gkPolytope));
    gkCollisionPair* idx_pairs     = (gkCollisionPair*)malloc(NUM_POLYTOPES * sizeof(gkCollisionPair));
    gkSimplex*       idx_simplices = (gkSimplex*)malloc(NUM_POLYTOPES * sizeof(gkSimplex));
    gkFloat*         idx_distances = (gkFloat*)malloc(NUM_POLYTOPES * sizeof(gkFloat));
    for (int i = 0; i < NUM_POLYTOPES; i++) {
      idx_polytopes[2*i]   = polytopes1[i];
      idx_polytopes[2*i+1] = polytopes2[i];
      idx_pairs[i].idx1 = 2*i;
      idx_pairs[i].idx2 = 2*i+1;
      idx_simplices[i].nvrtx = 0;
    }

    // Staging buffer: pack all coords contiguously
    int total_verts = 0;
    for (int i = 0; i < 2 * NUM_POLYTOPES; i++) total_verts += idx_polytopes[i].numpoints;

    gkFloat*         d_coords    = nullptr;
    gkPolytope*      d_polytopes = nullptr;
    gkCollisionPair* d_pairs     = nullptr;
    gkSimplex*       d_simplices = nullptr;
    gkFloat*         d_distances = nullptr;

    cudaMalloc(&d_coords,    total_verts * 3 * sizeof(gkFloat));
    cudaMalloc(&d_polytopes, 2 * NUM_POLYTOPES * sizeof(gkPolytope));
    cudaMalloc(&d_pairs,     NUM_POLYTOPES * sizeof(gkCollisionPair));
    cudaMalloc(&d_simplices, NUM_POLYTOPES * sizeof(gkSimplex));
    cudaMalloc(&d_distances, NUM_POLYTOPES * sizeof(gkFloat));

    gkPolytope* temp = (gkPolytope*)malloc(2 * NUM_POLYTOPES * sizeof(gkPolytope));
    gkFloat*    staging = (gkFloat*)malloc(total_verts * 3 * sizeof(gkFloat));
    int offset = 0;
    for (int i = 0; i < 2 * NUM_POLYTOPES; i++) {
      int n = idx_polytopes[i].numpoints * 3;
      memcpy(staging + offset, idx_polytopes[i].coord, n * sizeof(gkFloat));
      temp[i] = idx_polytopes[i];
      temp[i].coord = d_coords + offset;
      offset += n;
    }
    cudaMemcpy(d_coords, staging, total_verts * 3 * sizeof(gkFloat), cudaMemcpyHostToDevice);
    cudaMemcpy(d_polytopes, temp, 2 * NUM_POLYTOPES * sizeof(gkPolytope), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pairs, idx_pairs, NUM_POLYTOPES * sizeof(gkCollisionPair), cudaMemcpyHostToDevice);
    cudaMemset(d_simplices, 0, NUM_POLYTOPES * sizeof(gkSimplex));
    free(staging);
    free(temp);

    GJK::GPU::timer().startGpuTimer();
    compute_minimum_distance_indexed_device(NUM_POLYTOPES, d_polytopes, d_pairs, d_simplices, d_distances);
    GJK::GPU::timer().endGpuTimer();
    float t_idev_time = GJK::GPU::timer().getGpuElapsedTimeForPreviousOperation();

    cudaMemcpy(idx_simplices, d_simplices, NUM_POLYTOPES * sizeof(gkSimplex), cudaMemcpyDeviceToHost);
    cudaMemcpy(idx_distances, d_distances, NUM_POLYTOPES * sizeof(gkFloat), cudaMemcpyDeviceToHost);

    bool passed = true;
    for (int i = 0; i < test_count; i++) {
      if (fabs(idx_distances[i] - gpu_distances[i]) > tolerance) { passed = false; break; }
    }
    printf("compute_minimum_distance_indexed_device: %s  %.4f ms\n", passed ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m", t_idev_time);

    cudaFree(d_coords);
    cudaFree(d_polytopes);
    cudaFree(d_pairs);
    cudaFree(d_simplices);
    cudaFree(d_distances);
    free(idx_polytopes);
    free(idx_pairs);
    free(idx_simplices);
    free(idx_distances);
  }

  // --- Test 2: computeGJKAndEPA - distances match GJK reference ---
  gkSimplex* gjkepa_simplices = (gkSimplex*)malloc(NUM_POLYTOPES * sizeof(gkSimplex));
  gkFloat*   gjkepa_distances = (gkFloat*)malloc(NUM_POLYTOPES * sizeof(gkFloat));
  gkFloat*   gjkepa_witness1  = (gkFloat*)malloc(NUM_POLYTOPES * 3 * sizeof(gkFloat));
  gkFloat*   gjkepa_witness2  = (gkFloat*)malloc(NUM_POLYTOPES * 3 * sizeof(gkFloat));
  for (int i = 0; i < NUM_POLYTOPES; i++) gjkepa_simplices[i].nvrtx = 0;

  GJK::GPU::computeGJKAndEPA(NUM_POLYTOPES, polytopes1, polytopes2,
                              gjkepa_simplices, gjkepa_distances,
                              gjkepa_witness1, gjkepa_witness2);
  float t2_time = GJK::GPU::timer().getGpuElapsedTimeForPreviousOperation();
  {
    bool passed = true;
    for (int i = 0; i < test_count; i++) {
      if (gpu_distances[i] > tolerance) {
        // Non-colliding: EPA doesn't change distance, should match GJK
        if (fabs(gjkepa_distances[i] - gpu_distances[i]) > tolerance) { passed = false; break; }
      } else {
        // Colliding: EPA returns negative penetration depth
        if (gjkepa_distances[i] > tolerance) { passed = false; break; }
      }
    }
    printf("computeGJKAndEPA (distances vs GJK):   %s  %.4f ms\n", passed ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m", t2_time);
  }

  // --- Test 3: computeCollisionInformation - witnesses match computeGJKAndEPA ---
  {
    gkSimplex* t_simplices = (gkSimplex*)malloc(NUM_POLYTOPES * sizeof(gkSimplex));
    gkFloat*   t_distances = (gkFloat*)malloc(NUM_POLYTOPES * sizeof(gkFloat));
    gkFloat*   t_witness1  = (gkFloat*)malloc(NUM_POLYTOPES * 3 * sizeof(gkFloat));
    gkFloat*   t_witness2  = (gkFloat*)malloc(NUM_POLYTOPES * 3 * sizeof(gkFloat));
    gkFloat*   t_normals   = (gkFloat*)malloc(NUM_POLYTOPES * 3 * sizeof(gkFloat));
    for (int i = 0; i < NUM_POLYTOPES; i++) t_simplices[i].nvrtx = 0;

    GJK::GPU::computeDistances(NUM_POLYTOPES, polytopes1, polytopes2, t_simplices, t_distances);
    GJK::GPU::computeCollisionInformation(NUM_POLYTOPES, polytopes1, polytopes2,
                                          t_simplices, t_distances,
                                          t_witness1, t_witness2, t_normals);
    float t3_time = GJK::GPU::timer().getGpuElapsedTimeForPreviousOperation();

    bool passed = true;
    for (int i = 0; i < test_count && passed; i++) {
      for (int d = 0; d < 3; d++) {
        if (fabs(t_witness1[i*3+d] - gjkepa_witness1[i*3+d]) > tolerance ||
            fabs(t_witness2[i*3+d] - gjkepa_witness2[i*3+d]) > tolerance) {
          passed = false; break;
        }
      }
    }
    printf("computeCollisionInformation (witnesses): %s  %.4f ms\n", passed ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m", t3_time);

    free(t_simplices);
    free(t_distances);
    free(t_witness1);
    free(t_witness2);
    free(t_normals);
  }

  // --- Test 4: computeCollisionInformation with nullptr contact_normals ---
  {
    gkSimplex* t_simplices = (gkSimplex*)malloc(NUM_POLYTOPES * sizeof(gkSimplex));
    gkFloat*   t_distances = (gkFloat*)malloc(NUM_POLYTOPES * sizeof(gkFloat));
    gkFloat*   t_witness1  = (gkFloat*)malloc(NUM_POLYTOPES * 3 * sizeof(gkFloat));
    gkFloat*   t_witness2  = (gkFloat*)malloc(NUM_POLYTOPES * 3 * sizeof(gkFloat));
    for (int i = 0; i < NUM_POLYTOPES; i++) t_simplices[i].nvrtx = 0;

    GJK::GPU::computeDistances(NUM_POLYTOPES, polytopes1, polytopes2, t_simplices, t_distances);
    GJK::GPU::computeCollisionInformation(NUM_POLYTOPES, polytopes1, polytopes2,
                                          t_simplices, t_distances,
                                          t_witness1, t_witness2, nullptr);
    float t4_time = GJK::GPU::timer().getGpuElapsedTimeForPreviousOperation();

    bool passed = true;
    for (int i = 0; i < test_count && passed; i++) {
      for (int d = 0; d < 3; d++) {
        if (fabs(t_witness1[i*3+d] - gjkepa_witness1[i*3+d]) > tolerance ||
            fabs(t_witness2[i*3+d] - gjkepa_witness2[i*3+d]) > tolerance) {
          passed = false; break;
        }
      }
    }
    printf("computeCollisionInformation (nullptr):   %s  %.4f ms\n", passed ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m", t4_time);

    free(t_simplices);
    free(t_distances);
    free(t_witness1);
    free(t_witness2);
  }

  // --- Test 5: compute_gjk_epa_indexed - witnesses match gjkepa reference ---
  {
    gkPolytope*      idx_polytopes = (gkPolytope*)malloc(2 * NUM_POLYTOPES * sizeof(gkPolytope));
    gkCollisionPair* idx_pairs     = (gkCollisionPair*)malloc(NUM_POLYTOPES * sizeof(gkCollisionPair));
    gkSimplex*       idx_simplices = (gkSimplex*)malloc(NUM_POLYTOPES * sizeof(gkSimplex));
    gkFloat*         idx_distances = (gkFloat*)malloc(NUM_POLYTOPES * sizeof(gkFloat));
    gkFloat*         idx_witness1  = (gkFloat*)malloc(NUM_POLYTOPES * 3 * sizeof(gkFloat));
    gkFloat*         idx_witness2  = (gkFloat*)malloc(NUM_POLYTOPES * 3 * sizeof(gkFloat));
    for (int i = 0; i < NUM_POLYTOPES; i++) {
      idx_polytopes[2*i]   = polytopes1[i];
      idx_polytopes[2*i+1] = polytopes2[i];
      idx_pairs[i].idx1 = 2*i;
      idx_pairs[i].idx2 = 2*i+1;
    }

    GJK::GPU::timer().startGpuTimer();
    compute_gjk_epa_indexed(2 * NUM_POLYTOPES, NUM_POLYTOPES,
                            idx_polytopes, idx_pairs,
                            idx_simplices, idx_distances,
                            idx_witness1, idx_witness2);
    GJK::GPU::timer().endGpuTimer();
    float t5_time = GJK::GPU::timer().getGpuElapsedTimeForPreviousOperation();

    bool passed = true;
    for (int i = 0; i < test_count && passed; i++) {
      for (int d = 0; d < 3; d++) {
        if (fabs(idx_witness1[i*3+d] - gjkepa_witness1[i*3+d]) > tolerance ||
            fabs(idx_witness2[i*3+d] - gjkepa_witness2[i*3+d]) > tolerance) {
          passed = false; break;
        }
      }
    }
    printf("compute_gjk_epa_indexed (witnesses):     %s  %.4f ms\n", passed ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m", t5_time);

    free(idx_polytopes);
    free(idx_pairs);
    free(idx_simplices);
    free(idx_distances);
    free(idx_witness1);
    free(idx_witness2);
  }

  // --- Test 6: compute_epa_indexed - witnesses match gjkepa reference ---
  {
    gkPolytope*      idx_polytopes = (gkPolytope*)malloc(2 * NUM_POLYTOPES * sizeof(gkPolytope));
    gkCollisionPair* idx_pairs     = (gkCollisionPair*)malloc(NUM_POLYTOPES * sizeof(gkCollisionPair));
    gkSimplex*       idx_simplices = (gkSimplex*)malloc(NUM_POLYTOPES * sizeof(gkSimplex));
    gkFloat*         idx_distances = (gkFloat*)malloc(NUM_POLYTOPES * sizeof(gkFloat));
    gkFloat*         idx_witness1  = (gkFloat*)malloc(NUM_POLYTOPES * 3 * sizeof(gkFloat));
    gkFloat*         idx_witness2  = (gkFloat*)malloc(NUM_POLYTOPES * 3 * sizeof(gkFloat));
    for (int i = 0; i < NUM_POLYTOPES; i++) {
      idx_polytopes[2*i]   = polytopes1[i];
      idx_polytopes[2*i+1] = polytopes2[i];
      idx_pairs[i].idx1 = 2*i;
      idx_pairs[i].idx2 = 2*i+1;
      idx_simplices[i] = gjkepa_simplices[i];
      idx_distances[i] = gjkepa_distances[i];
    }

    GJK::GPU::timer().startGpuTimer();
    compute_epa_indexed(2 * NUM_POLYTOPES, NUM_POLYTOPES,
                        idx_polytopes, idx_pairs,
                        idx_simplices, idx_distances,
                        idx_witness1, idx_witness2);
    GJK::GPU::timer().endGpuTimer();
    float t6_time = GJK::GPU::timer().getGpuElapsedTimeForPreviousOperation();

    bool passed = true;
    for (int i = 0; i < test_count && passed; i++) {
      for (int d = 0; d < 3; d++) {
        if (fabs(idx_witness1[i*3+d] - gjkepa_witness1[i*3+d]) > tolerance ||
            fabs(idx_witness2[i*3+d] - gjkepa_witness2[i*3+d]) > tolerance) {
          passed = false; break;
        }
      }
    }
    printf("compute_epa_indexed (witnesses):         %s  %.4f ms\n", passed ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m", t6_time);

    free(idx_polytopes);
    free(idx_pairs);
    free(idx_simplices);
    free(idx_distances);
    free(idx_witness1);
    free(idx_witness2);
  }

  printf("================================================================================\n");

  free(gjkepa_simplices);
  free(gjkepa_distances);
  free(gjkepa_witness1);
  free(gjkepa_witness2);
#endif

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
  free(gpu_simplices);
  free(warm_up_gpu_simplices);
  free(distances);
  free(gpu_distances);
  free(warm_up_gpu_distances);

  //printf("\Validation Testing complete!\n");

#if SAVE_PERFORMANCE_DATA_TO_FILE
  // Run spread of performance testing for csv file output
  int polytopeCounts[] = {10, 50, 100, 500, 1000, 5000};
  int vertexCounts[] = {10, 25, 50, 100, 200, 500};
  std::string outputFile = OUTPUT_FILE;

  GJK::GPU::testing(polytopeCounts, sizeof(polytopeCounts) / sizeof(polytopeCounts[0]), vertexCounts, sizeof(vertexCounts) / sizeof(vertexCounts[0]), outputFile.c_str());
#endif

#if EPA_TESTING
  printf("\n");
  GJK::GPU::EPATesting();
#endif
  return (0);
}
