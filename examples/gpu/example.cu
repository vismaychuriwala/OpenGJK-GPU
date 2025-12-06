#include "example.h"
#include "GJK/gpu/warpParallelGJK.h"
#include "../cpu/example.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#ifdef WIN32
#include <errno.h>
#endif

namespace GJK {
    namespace GPU {
        using GJK::Common::PerformanceTimer;

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        void computeDistances(int n,
                            const gkPolytope* bd1,
                            const gkPolytope* bd2,
                            gkSimplex* simplices,
                            gkFloat* distances) {
            if (n <= 0) return;

            // Device pointers
            gkPolytope* d_bd1 = nullptr;
            gkPolytope* d_bd2 = nullptr;
            gkSimplex* d_simplices = nullptr;
            gkFloat* d_distances = nullptr;
            gkFloat* d_coord1 = nullptr;
            gkFloat* d_coord2 = nullptr;

            // Allocate device memory
            cudaMalloc(&d_bd1, n * sizeof(gkPolytope));
            cudaMalloc(&d_bd2, n * sizeof(gkPolytope));
            cudaMalloc(&d_simplices, n * sizeof(gkSimplex));
            cudaMalloc(&d_distances, n * sizeof(gkFloat));

            // Calculate total coordinate size needed
            int total_coords1 = 0;
            int total_coords2 = 0;
            for (int i = 0; i < n; i++) {
                total_coords1 += bd1[i].numpoints * 3;
                total_coords2 += bd2[i].numpoints * 3;
            }

            // Allocate coordinate arrays
            cudaMalloc(&d_coord1, total_coords1 * sizeof(gkFloat));
            cudaMalloc(&d_coord2, total_coords2 * sizeof(gkFloat));

            // Copy coordinate data and update polytope structures
            gkPolytope* temp_bd1 = new gkPolytope[n];
            gkPolytope* temp_bd2 = new gkPolytope[n];

            int offset1 = 0;
            int offset2 = 0;
            for (int i = 0; i < n; i++) {
                // Copy polytope metadata
                temp_bd1[i] = bd1[i];
                temp_bd2[i] = bd2[i];

                // Copy coordinate data to device
                int coord_size1 = bd1[i].numpoints * 3 * sizeof(gkFloat);
                int coord_size2 = bd2[i].numpoints * 3 * sizeof(gkFloat);

                cudaMemcpy(d_coord1 + offset1, bd1[i].coord, coord_size1, cudaMemcpyHostToDevice);
                cudaMemcpy(d_coord2 + offset2, bd2[i].coord, coord_size2, cudaMemcpyHostToDevice);

                // Update pointers to device memory locations
                temp_bd1[i].coord = d_coord1 + offset1;
                temp_bd2[i].coord = d_coord2 + offset2;

                offset1 += bd1[i].numpoints * 3;
                offset2 += bd2[i].numpoints * 3;
            }

            // Copy polytope structures to device
            cudaMemcpy(d_bd1, temp_bd1, n * sizeof(gkPolytope), cudaMemcpyHostToDevice);
            cudaMemcpy(d_bd2, temp_bd2, n * sizeof(gkPolytope), cudaMemcpyHostToDevice);
            cudaMemcpy(d_simplices, simplices, n * sizeof(gkSimplex), cudaMemcpyHostToDevice);

            // Launch kernel with timing
            timer().startGpuTimer();
            int blockSize = 256;
            int numBlocks = (n + blockSize - 1) / blockSize;
            compute_minimum_distance<<<numBlocks, blockSize>>>(d_bd1, d_bd2, d_simplices, d_distances, n);
            timer().endGpuTimer();

            // Copy results back
            cudaMemcpy(distances, d_distances, n * sizeof(gkFloat), cudaMemcpyDeviceToHost);
            cudaMemcpy(simplices, d_simplices, n * sizeof(gkSimplex), cudaMemcpyDeviceToHost);

            // Free memory
            delete[] temp_bd1;
            delete[] temp_bd2;
            cudaFree(d_bd1);
            cudaFree(d_bd2);
            cudaFree(d_simplices);
            cudaFree(d_distances);
            cudaFree(d_coord1);
            cudaFree(d_coord2);
        }

        void computeDistancesWarpParallel(int n,
                            const gkPolytope* bd1,
                            const gkPolytope* bd2,
                            gkSimplex* simplices,
                            gkFloat* distances) {
            if (n <= 0) return;

            // Device pointers
            gkPolytope* d_bd1 = nullptr;
            gkPolytope* d_bd2 = nullptr;
            gkSimplex* d_simplices = nullptr;
            gkFloat* d_distances = nullptr;
            gkFloat* d_coord1 = nullptr;
            gkFloat* d_coord2 = nullptr;

            // Allocate device memory
            cudaMalloc(&d_bd1, n * sizeof(gkPolytope));
            cudaMalloc(&d_bd2, n * sizeof(gkPolytope));
            cudaMalloc(&d_simplices, n * sizeof(gkSimplex));
            cudaMalloc(&d_distances, n * sizeof(gkFloat));

            // Calculate total coordinate size needed
            int total_coords1 = 0;
            int total_coords2 = 0;
            for (int i = 0; i < n; i++) {
                total_coords1 += bd1[i].numpoints * 3;
                total_coords2 += bd2[i].numpoints * 3;
            }

            // Allocate coordinate arrays
            cudaMalloc(&d_coord1, total_coords1 * sizeof(gkFloat));
            cudaMalloc(&d_coord2, total_coords2 * sizeof(gkFloat));

            // Copy coordinate data and update polytope structures
            gkPolytope* temp_bd1 = new gkPolytope[n];
            gkPolytope* temp_bd2 = new gkPolytope[n];

            int offset1 = 0;
            int offset2 = 0;
            for (int i = 0; i < n; i++) {
                // Copy polytope metadata
                temp_bd1[i] = bd1[i];
                temp_bd2[i] = bd2[i];

                // Copy coordinate data to device
                int coord_size1 = bd1[i].numpoints * 3 * sizeof(gkFloat);
                int coord_size2 = bd2[i].numpoints * 3 * sizeof(gkFloat);

                cudaMemcpy(d_coord1 + offset1, bd1[i].coord, coord_size1, cudaMemcpyHostToDevice);
                cudaMemcpy(d_coord2 + offset2, bd2[i].coord, coord_size2, cudaMemcpyHostToDevice);

                // Update pointers to device memory locations
                temp_bd1[i].coord = d_coord1 + offset1;
                temp_bd2[i].coord = d_coord2 + offset2;

                offset1 += bd1[i].numpoints * 3;
                offset2 += bd2[i].numpoints * 3;
            }

            // Copy polytope structures to device
            cudaMemcpy(d_bd1, temp_bd1, n * sizeof(gkPolytope), cudaMemcpyHostToDevice);
            cudaMemcpy(d_bd2, temp_bd2, n * sizeof(gkPolytope), cudaMemcpyHostToDevice);
            cudaMemcpy(d_simplices, simplices, n * sizeof(gkSimplex), cudaMemcpyHostToDevice);
            
            // Launch kernel with timing
            // Each collision uses 16 threads (half-warp)
            // Results in 16 times as many threads as the regular GPU implementationS
            // Block size should be a multiple of 16
            timer().startGpuTimer();
            const int THREADS_PER_COMPUTATION = 16;
            int blockSize = 256;  // 256 threads = 16 collisions per block
            int collisionsPerBlock = blockSize / THREADS_PER_COMPUTATION;
            int numBlocks = (n + collisionsPerBlock - 1) / collisionsPerBlock;
            compute_minimum_distance_warp_parallel<<<numBlocks, blockSize>>>(d_bd1, d_bd2, d_simplices, d_distances, n);
            timer().endGpuTimer();

            // Copy results back
            cudaMemcpy(distances, d_distances, n * sizeof(gkFloat), cudaMemcpyDeviceToHost);
            cudaMemcpy(simplices, d_simplices, n * sizeof(gkSimplex), cudaMemcpyDeviceToHost);

            // Free memory
            delete[] temp_bd1;
            delete[] temp_bd2;
            cudaFree(d_bd1);
            cudaFree(d_bd2);
            cudaFree(d_simplices);
            cudaFree(d_distances);
            cudaFree(d_coord1);
            cudaFree(d_coord2);
        }

        void computeGJKAndEPA(int n,
                            const gkPolytope* bd1,
                            const gkPolytope* bd2,
                            gkSimplex* simplices,
                            gkFloat* distances,
                            gkFloat* witness1,
                            gkFloat* witness2) {
            if (n <= 0) return;

            // Device pointers
            gkPolytope* d_bd1 = nullptr;
            gkPolytope* d_bd2 = nullptr;
            gkSimplex* d_simplices = nullptr;
            gkFloat* d_distances = nullptr;
            gkFloat* d_witness1 = nullptr;
            gkFloat* d_witness2 = nullptr;
            gkFloat* d_coord1 = nullptr;
            gkFloat* d_coord2 = nullptr;

            // Allocate device memory
            cudaMalloc(&d_bd1, n * sizeof(gkPolytope));
            cudaMalloc(&d_bd2, n * sizeof(gkPolytope));
            cudaMalloc(&d_simplices, n * sizeof(gkSimplex));
            cudaMalloc(&d_distances, n * sizeof(gkFloat));
            cudaMalloc(&d_witness1, n * 3 * sizeof(gkFloat));
            cudaMalloc(&d_witness2, n * 3 * sizeof(gkFloat));

            int total_coords1 = 0;
            int total_coords2 = 0;
            for (int i = 0; i < n; i++) {
                total_coords1 += bd1[i].numpoints * 3;
                total_coords2 += bd2[i].numpoints * 3;
            }

            cudaMalloc(&d_coord1, total_coords1 * sizeof(gkFloat));
            cudaMalloc(&d_coord2, total_coords2 * sizeof(gkFloat));

            gkPolytope* temp_bd1 = new gkPolytope[n];
            gkPolytope* temp_bd2 = new gkPolytope[n];

            int offset1 = 0;
            int offset2 = 0;
            for (int i = 0; i < n; i++) {
                temp_bd1[i] = bd1[i];
                temp_bd2[i] = bd2[i];

                int coord_size1 = bd1[i].numpoints * 3 * sizeof(gkFloat);
                int coord_size2 = bd2[i].numpoints * 3 * sizeof(gkFloat);

                cudaMemcpy(d_coord1 + offset1, bd1[i].coord, coord_size1, cudaMemcpyHostToDevice);
                cudaMemcpy(d_coord2 + offset2, bd2[i].coord, coord_size2, cudaMemcpyHostToDevice);

                temp_bd1[i].coord = d_coord1 + offset1;
                temp_bd2[i].coord = d_coord2 + offset2;

                offset1 += bd1[i].numpoints * 3;
                offset2 += bd2[i].numpoints * 3;
            }

            // Copy polytope structures to device
            cudaMemcpy(d_bd1, temp_bd1, n * sizeof(gkPolytope), cudaMemcpyHostToDevice);
            cudaMemcpy(d_bd2, temp_bd2, n * sizeof(gkPolytope), cudaMemcpyHostToDevice);
            cudaMemcpy(d_simplices, simplices, n * sizeof(gkSimplex), cudaMemcpyHostToDevice);
            
            // Launch GJK kernel with timing
            timer().startGpuTimer();
            const int THREADS_PER_COMPUTATION_GJK = 16;
            int blockSizeGJK = 256;
            int collisionsPerBlockGJK = blockSizeGJK / THREADS_PER_COMPUTATION_GJK;
            int numBlocksGJK = (n + collisionsPerBlockGJK - 1) / collisionsPerBlockGJK;
            compute_minimum_distance_warp_parallel<<<numBlocksGJK, blockSizeGJK>>>(d_bd1, d_bd2, d_simplices, d_distances, n);
            
            // Wait for GJK to complete before starting EPA
            cudaDeviceSynchronize();
            
            // Copy GJK results to print them
            gkFloat* gjk_distances = new gkFloat[n];
            gkSimplex* gjk_simplices = new gkSimplex[n];
            gkFloat* gjk_witness1 = new gkFloat[n * 3];
            gkFloat* gjk_witness2 = new gkFloat[n * 3];
            
            cudaMemcpy(gjk_distances, d_distances, n * sizeof(gkFloat), cudaMemcpyDeviceToHost);
            cudaMemcpy(gjk_simplices, d_simplices, n * sizeof(gkSimplex), cudaMemcpyDeviceToHost);
            
            // Copy witness points from simplice
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < 3; j++) {
                    gjk_witness1[i * 3 + j] = gjk_simplices[i].witnesses[0][j];
                    gjk_witness2[i * 3 + j] = gjk_simplices[i].witnesses[1][j];
                }
            }
            
            // Print GJK results
            printf("\n=== GJK Results (before EPA) ===\n");
            for (int i = 0; i < n; i++) {
                printf("  Collision %d:\n", i);
                printf("    Simplex vertices: %d\n", gjk_simplices[i].nvrtx);
                printf("    Distance: %f\n", gjk_distances[i]);
                for (int v = 0; v < gjk_simplices[i].nvrtx; v++) {
                    printf("    Vertex %d: (%.6f, %.6f, %.6f)\n", v,
                           gjk_simplices[i].vrtx[v][0],
                           gjk_simplices[i].vrtx[v][1],
                           gjk_simplices[i].vrtx[v][2]);
                }
                printf("    Witness 1: (%.6f, %.6f, %.6f)\n", 
                       gjk_witness1[i * 3 + 0], gjk_witness1[i * 3 + 1], gjk_witness1[i * 3 + 2]);
                printf("    Witness 2: (%.6f, %.6f, %.6f)\n", 
                       gjk_witness2[i * 3 + 0], gjk_witness2[i * 3 + 1], gjk_witness2[i * 3 + 2]);
            }
            printf("================================\n\n");
            
            // Clean up temporary arrays
            delete[] gjk_distances;
            delete[] gjk_simplices;
            delete[] gjk_witness1;
            delete[] gjk_witness2;

            // Launch EPA kernel
            // Each collision uses 32 threads (one warp)
            const int THREADS_PER_COMPUTATION_EPA = 32;
            int blockSizeEPA = 256;
            int collisionsPerBlockEPA = blockSizeEPA / THREADS_PER_COMPUTATION_EPA;
            int numBlocksEPA = (n + collisionsPerBlockEPA - 1) / collisionsPerBlockEPA;
            compute_epa_warp_parallel<<<numBlocksEPA, blockSizeEPA>>>(d_bd1, d_bd2, d_simplices, d_distances, d_witness1, d_witness2, n);
            timer().endGpuTimer();

            cudaMemcpy(distances, d_distances, n * sizeof(gkFloat), cudaMemcpyDeviceToHost);
            cudaMemcpy(simplices, d_simplices, n * sizeof(gkSimplex), cudaMemcpyDeviceToHost);
            cudaMemcpy(witness1, d_witness1, n * 3 * sizeof(gkFloat), cudaMemcpyDeviceToHost);
            cudaMemcpy(witness2, d_witness2, n * 3 * sizeof(gkFloat), cudaMemcpyDeviceToHost);

            // Free memory
            delete[] temp_bd1;
            delete[] temp_bd2;
            cudaFree(d_bd1);
            cudaFree(d_bd2);
            cudaFree(d_simplices);
            cudaFree(d_distances);
            cudaFree(d_witness1);
            cudaFree(d_witness2);
            cudaFree(d_coord1);
            cudaFree(d_coord2);
        }

        /// @brief Generate a polytope with specified number of vertices and random offset
        static gkFloat* generatePolytope(int numVerts, gkFloat offsetX, gkFloat offsetY, gkFloat offsetZ) {
            gkFloat* verts = (gkFloat*)malloc(numVerts * 3 * sizeof(gkFloat));
            const gkFloat M_PI = 3.14159265358979323846;

            // Generate random points on a sphere (simple polytope generation)
            for (int i = 0; i < numVerts; i++) {
                gkFloat theta = ((gkFloat)rand() / RAND_MAX) * 2.0f * M_PI;
                gkFloat phi = ((gkFloat)rand() / RAND_MAX) * M_PI;
                gkFloat r = 1.0f + ((gkFloat)rand() / RAND_MAX) * 0.5f;

                verts[i*3 + 0] = r * sin(phi) * cos(theta) + offsetX;
                verts[i*3 + 1] = r * sin(phi) * sin(theta) + offsetY;
                verts[i*3 + 2] = r * cos(phi) + offsetZ;
            }

            return verts;
        }

        /// @brief Generate a cube with a grid of vertices on each face
        /// @param gridSize Number of vertices per side of each face (gridSize x gridSize per face)
        /// @param size Half-size of the cube (cube extends from -size to +size in each dimension)
        /// @param offsetX, offsetY, offsetZ Translation offset
        /// @return Pointer to allocated vertex array (6 * gridSize * gridSize vertices)
        static gkFloat* generateCubeWithGrid(int gridSize, gkFloat size, gkFloat offsetX, gkFloat offsetY, gkFloat offsetZ) {
            int verticesPerFace = gridSize * gridSize;
            int totalVertices = 6 * verticesPerFace;
            gkFloat* verts = (gkFloat*)malloc(totalVertices * 3 * sizeof(gkFloat));
            
            int idx = 0;
            
            // Generate vertices for each of the 6 faces
            // Face 1: +X face (x = size)
            for (int i = 0; i < gridSize; i++) {
                for (int j = 0; j < gridSize; j++) {
                    gkFloat y = -size + (2.0f * size * i) / (gridSize - 1);
                    gkFloat z = -size + (2.0f * size * j) / (gridSize - 1);
                    verts[idx * 3 + 0] = size + offsetX;
                    verts[idx * 3 + 1] = y + offsetY;
                    verts[idx * 3 + 2] = z + offsetZ;
                    idx++;
                }
            }
            
            // Face 2: -X face (x = -size)
            for (int i = 0; i < gridSize; i++) {
                for (int j = 0; j < gridSize; j++) {
                    gkFloat y = -size + (2.0f * size * i) / (gridSize - 1);
                    gkFloat z = -size + (2.0f * size * j) / (gridSize - 1);
                    verts[idx * 3 + 0] = -size + offsetX;
                    verts[idx * 3 + 1] = y + offsetY;
                    verts[idx * 3 + 2] = z + offsetZ;
                    idx++;
                }
            }
            
            // Face 3: +Y face (y = size)
            for (int i = 0; i < gridSize; i++) {
                for (int j = 0; j < gridSize; j++) {
                    gkFloat x = -size + (2.0f * size * i) / (gridSize - 1);
                    gkFloat z = -size + (2.0f * size * j) / (gridSize - 1);
                    verts[idx * 3 + 0] = x + offsetX;
                    verts[idx * 3 + 1] = size + offsetY;
                    verts[idx * 3 + 2] = z + offsetZ;
                    idx++;
                }
            }
            
            // Face 4: -Y face (y = -size)
            for (int i = 0; i < gridSize; i++) {
                for (int j = 0; j < gridSize; j++) {
                    gkFloat x = -size + (2.0f * size * i) / (gridSize - 1);
                    gkFloat z = -size + (2.0f * size * j) / (gridSize - 1);
                    verts[idx * 3 + 0] = x + offsetX;
                    verts[idx * 3 + 1] = -size + offsetY;
                    verts[idx * 3 + 2] = z + offsetZ;
                    idx++;
                }
            }
            
            // Face 5: +Z face (z = size)
            for (int i = 0; i < gridSize; i++) {
                for (int j = 0; j < gridSize; j++) {
                    gkFloat x = -size + (2.0f * size * i) / (gridSize - 1);
                    gkFloat y = -size + (2.0f * size * j) / (gridSize - 1);
                    verts[idx * 3 + 0] = x + offsetX;
                    verts[idx * 3 + 1] = y + offsetY;
                    verts[idx * 3 + 2] = size + offsetZ;
                    idx++;
                }
            }
            
            // Face 6: -Z face (z = -size)
            for (int i = 0; i < gridSize; i++) {
                for (int j = 0; j < gridSize; j++) {
                    gkFloat x = -size + (2.0f * size * i) / (gridSize - 1);
                    gkFloat y = -size + (2.0f * size * j) / (gridSize - 1);
                    verts[idx * 3 + 0] = x + offsetX;
                    verts[idx * 3 + 1] = y + offsetY;
                    verts[idx * 3 + 2] = -size + offsetZ;
                    idx++;
                }
            }
            
            return verts;
        }

        /// @brief Generate points randomly distributed on the surface of a sphere
        /// @param numPoints Number of points to generate
        /// @param radius Radius of the sphere
        /// @param offsetX, offsetY, offsetZ Translation offset (center of sphere)
        /// @return Pointer to allocated vertex array (numPoints vertices)
        static gkFloat* generateSphereSurface(int numPoints, gkFloat radius, gkFloat offsetX, gkFloat offsetY, gkFloat offsetZ) {
            gkFloat* verts = (gkFloat*)malloc(numPoints * 3 * sizeof(gkFloat));
            const gkFloat M_PI = 3.14159265358979323846;

            // Generate random points uniformly distributed on sphere surface
            for (int i = 0; i < numPoints; i++) {
                // Generate uniform random points on sphere using spherical coordinates
                // For uniform distribution: theta in [0, 2*PI], phi in [0, PI]
                // But we need to account for the fact that phi should be distributed as cos(phi)
                // to get uniform distribution on the sphere surface
                gkFloat u = ((gkFloat)rand() / RAND_MAX); // [0, 1]
                gkFloat v = ((gkFloat)rand() / RAND_MAX); // [0, 1]
                
                gkFloat theta = 2.0f * M_PI * u; // [0, 2*PI]
                gkFloat phi = acos(2.0f * v - 1.0f); // [0, PI] with proper distribution
                
                // Convert to Cartesian coordinates on sphere surface
                verts[i * 3 + 0] = radius * sin(phi) * cos(theta) + offsetX;
                verts[i * 3 + 1] = radius * sin(phi) * sin(theta) + offsetY;
                verts[i * 3 + 2] = radius * cos(phi) + offsetZ;
            }

            return verts;
        }

        void testing(const int* numPolytopesArray,
                     int numPolytopesArraySize,
                     const int* numVerticesArray,
                     int numVerticesArraySize,
                     const char* outputFile) {
            
            if (numPolytopesArray == nullptr || numVerticesArray == nullptr || outputFile == nullptr) {
                fprintf(stderr, "ERROR: Invalid parameters to testing function\n");
                return;
            }

            // Open CSV file to write results to
            FILE* fp = nullptr;
#ifdef WIN32
            errno_t err;
            if ((err = fopen_s(&fp, outputFile, "w")) != 0) {
#else
            if ((fp = fopen(outputFile, "w")) == NULL) {
#endif
                fprintf(stderr, "ERROR: Could not open output file %s for writing\n", outputFile);
                return;
            }

            // Write CSV header to file
            fprintf(fp, "NumPolytopes,NumVertices,CPU_Time_ms,GPU_Time_ms,WarpParallelGPU_Time_ms\n");

            // Iterate through all  of polytope count and vertex count
            for (int pIdx = 0; pIdx < numPolytopesArraySize; pIdx++) {
                int numPolytopes = numPolytopesArray[pIdx];
                
                for (int vIdx = 0; vIdx < numVerticesArraySize; vIdx++) {
                    int numVertices = numVerticesArray[vIdx];

                    // SUms for averaging over 10 runs
                    float cpu_time_sum = 0.0f;
                    float gpu_time_sum = 0.0f;
                    float warp_gpu_time_sum = 0.0f;
                    const int NUM_RUNS = 10;

                    // Run each test configuration 10 times and accumulate results
                    for (int run = 0; run < NUM_RUNS; run++) {
                        /* Allocate arrays for all polytope vertex data */
                        gkFloat** vrtx1_array = (gkFloat**)malloc(numPolytopes * sizeof(gkFloat*));
                        gkFloat** vrtx2_array = (gkFloat**)malloc(numPolytopes * sizeof(gkFloat*));

                        // Random offsets for each polytope pair
                        for (int i = 0; i < numPolytopes; i++) {
                            gkFloat offset1_x = ((gkFloat)rand() / RAND_MAX - 0.5f) * 10.0f;
                            gkFloat offset1_y = ((gkFloat)rand() / RAND_MAX - 0.5f) * 10.0f;
                            gkFloat offset1_z = ((gkFloat)rand() / RAND_MAX - 0.5f) * 10.0f;

                            gkFloat offset2_x = ((gkFloat)rand() / RAND_MAX - 0.5f) * 10.0f;
                            gkFloat offset2_y = ((gkFloat)rand() / RAND_MAX - 0.5f) * 10.0f;
                            gkFloat offset2_z = ((gkFloat)rand() / RAND_MAX - 0.5f) * 10.0f;

                            vrtx1_array[i] = generatePolytope(numVertices, offset1_x, offset1_y, offset1_z);
                            vrtx2_array[i] = generatePolytope(numVertices, offset2_x, offset2_y, offset2_z);
                        }

                        // Allocate arrays for polytope pairs
                        gkPolytope* polytopes1 = (gkPolytope*)malloc(numPolytopes * sizeof(gkPolytope));
                        gkPolytope* polytopes2 = (gkPolytope*)malloc(numPolytopes * sizeof(gkPolytope));
                        gkSimplex* simplices_cpu = (gkSimplex*)malloc(numPolytopes * sizeof(gkSimplex));
                        gkSimplex* simplices_gpu = (gkSimplex*)malloc(numPolytopes * sizeof(gkSimplex));
                        gkSimplex* simplices_warp = (gkSimplex*)malloc(numPolytopes * sizeof(gkSimplex));
                        gkSimplex* warm_up_simplices = (gkSimplex*)malloc(numPolytopes * sizeof(gkSimplex));
                        gkFloat* distances_cpu = (gkFloat*)malloc(numPolytopes * sizeof(gkFloat));
                        gkFloat* distances_gpu = (gkFloat*)malloc(numPolytopes * sizeof(gkFloat));
                        gkFloat* distances_warp = (gkFloat*)malloc(numPolytopes * sizeof(gkFloat));
                        gkFloat* warm_up_distances = (gkFloat*)malloc(numPolytopes * sizeof(gkFloat));

                        /* Initialize polytope pairs with unique data */
                        for (int i = 0; i < numPolytopes; i++) {
                            polytopes1[i].numpoints = numVertices;
                            polytopes1[i].coord = vrtx1_array[i];

                            polytopes2[i].numpoints = numVertices;
                            polytopes2[i].coord = vrtx2_array[i];

                            simplices_cpu[i].nvrtx = 0;
                            simplices_gpu[i].nvrtx = 0;
                            simplices_warp[i].nvrtx = 0;
                            warm_up_simplices[i].nvrtx = 0;
                        }

                        // Warm up GPU (only on first run to save time)
                        if (run == 0) {
                            GJK::GPU::computeDistances(numPolytopes, polytopes1, polytopes2, warm_up_simplices, warm_up_distances);
                            GJK::GPU::timer().getGpuElapsedTimeForPreviousOperation(); // Discard warm-up time
                        }

                        // Test CPU implementation
                        GJK::CPU::computeDistances(numPolytopes, polytopes1, polytopes2, simplices_cpu, distances_cpu);
                        float cpu_time = GJK::CPU::timer().getCpuElapsedTimeForPreviousOperation();
                        cpu_time_sum += cpu_time;

                        // Test GPU implementation
                        GJK::GPU::computeDistances(numPolytopes, polytopes1, polytopes2, simplices_gpu, distances_gpu);
                        float gpu_time = GJK::GPU::timer().getGpuElapsedTimeForPreviousOperation();
                        gpu_time_sum += gpu_time;

                        // Test Warp-Parallel GPU implementation
                        GJK::GPU::computeDistancesWarpParallel(numPolytopes, polytopes1, polytopes2, simplices_warp, distances_warp);
                        float warp_gpu_time = GJK::GPU::timer().getGpuElapsedTimeForPreviousOperation();
                        warp_gpu_time_sum += warp_gpu_time;

                        // Free memory for this run
                        for (int i = 0; i < numPolytopes; i++) {
                            free(vrtx1_array[i]);
                            free(vrtx2_array[i]);
                        }
                        free(vrtx1_array);
                        free(vrtx2_array);
                        free(polytopes1);
                        free(polytopes2);
                        free(simplices_cpu);
                        free(simplices_gpu);
                        free(simplices_warp);
                        free(warm_up_simplices);
                        free(distances_cpu);
                        free(distances_gpu);
                        free(distances_warp);
                        free(warm_up_distances);
                    }

                    // Calculate averages
                    float cpu_time_avg = cpu_time_sum / NUM_RUNS;
                    float gpu_time_avg = gpu_time_sum / NUM_RUNS;
                    float warp_gpu_time_avg = warp_gpu_time_sum / NUM_RUNS;

                    // Write average results to CSV
                    fprintf(fp, "%d,%d,%.6f,%.6f,%.6f\n", 
                            numPolytopes, numVertices, cpu_time_avg, gpu_time_avg, warp_gpu_time_avg);
                }
            }

            fclose(fp);
            printf("Performance testing complete! Results saved to %s\n", outputFile);
        }

        void EPATesting() {
            printf("\n========================================\n");
            printf("EPA Algorithm Testing\n");
            printf("========================================\n\n");



            // These are test cases for degenerate cases. Still in progress
            // // Test Case 1: Two overlapping cubes
            printf("Test Case 1: Two overlapping cubes\n");
            printf("-----------------------------------\n");
            {
               // Create two cubes that overlap
               // Cube 1: centered at (0, 0, 0), size 2x2x2
               // Cube 2: centered at (1, 0, 0), size 2x2x2 (overlaps by 1 unit)
               const int numVerts = 8;
               gkFloat* cube1_verts = (gkFloat*)malloc(numVerts * 3 * sizeof(gkFloat));
               gkFloat* cube2_verts = (gkFloat*)malloc(numVerts * 3 * sizeof(gkFloat));

               // Cube 1 vertices (centered at origin, size 2)
               int idx = 0;
               for (int x = -1; x <= 1; x += 2) {
                   for (int y = -1; y <= 1; y += 2) {
                       for (int z = -1; z <= 1; z += 2) {
                           cube1_verts[idx * 3 + 0] = x;
                           cube1_verts[idx * 3 + 1] = y;
                           cube1_verts[idx * 3 + 2] = z;
                           idx++;
                       }
                   }
               }

               // Cube 2 vertices (centered at (1, 0, 0), size 2)
               idx = 0;
               for (int x = -1; x <= 1; x += 2) {
                   for (int y = -1; y <= 1; y += 2) {
                       for (int z = -1; z <= 1; z += 2) {
                           cube2_verts[idx * 3 + 0] = x + 1.0f;
                           cube2_verts[idx * 3 + 1] = y;
                           cube2_verts[idx * 3 + 2] = z;
                           idx++;
                       }
                   }
               }

               gkPolytope polytope1, polytope2;
               polytope1.numpoints = numVerts;
               polytope1.coord = cube1_verts;
               polytope2.numpoints = numVerts;
               polytope2.coord = cube2_verts;

               gkSimplex simplex;
               simplex.nvrtx = 0;
               gkFloat distance;
               gkFloat witness1[3], witness2[3];

               computeGJKAndEPA(1, &polytope1, &polytope2, &simplex, &distance, witness1, witness2);

               printf("  Simplex vertices: %d\n", simplex.nvrtx);
               printf("  Distance/Penetration: %.6f\n", distance);
               printf("  Expected: Collision (distance should be small/negative)\n");
               printf("  Witness 1: (%.6f, %.6f, %.6f)\n", witness1[0], witness1[1], witness1[2]);
               printf("  Witness 2: (%.6f, %.6f, %.6f)\n", witness2[0], witness2[1], witness2[2]);
               
               // Verify witness points are within bounds
               bool valid1 = (witness1[0] >= -1.0f && witness1[0] <= 1.0f) &&
                            (witness1[1] >= -1.0f && witness1[1] <= 1.0f) &&
                            (witness1[2] >= -1.0f && witness1[2] <= 1.0f);
               bool valid2 = (witness2[0] >= 0.0f && witness2[0] <= 2.0f) &&
                            (witness2[1] >= -1.0f && witness2[1] <= 1.0f) &&
                            (witness2[2] >= -1.0f && witness2[2] <= 1.0f);
               
               if (simplex.nvrtx == 4 && valid1 && valid2) {
                   printf("  PASS: Collision detected, witness points valid\n");
               } else {
                   printf("  FAIL: Invalid results\n");
               }
               printf("\n");

               free(cube1_verts);
               free(cube2_verts);
            }

            // Test Case 2: Two touching cubes (just touching, no penetration)
            printf("Test Case 2: Two touching cubes\n");
            printf("-----------------------------------\n");
            {
               const int numVerts = 8;
               gkFloat* cube1_verts = (gkFloat*)malloc(numVerts * 3 * sizeof(gkFloat));
               gkFloat* cube2_verts = (gkFloat*)malloc(numVerts * 3 * sizeof(gkFloat));

               // Cube 1: centered at (0, 0, 0), size 2x2x2
               int idx = 0;
               for (int x = -1; x <= 1; x += 2) {
                   for (int y = -1; y <= 1; y += 2) {
                       for (int z = -1; z <= 1; z += 2) {
                           cube1_verts[idx * 3 + 0] = x;
                           cube1_verts[idx * 3 + 1] = y;
                           cube1_verts[idx * 3 + 2] = z;
                           idx++;
                       }
                   }
               }

               // Cube 2: centered at (2, 0, 0), size 2x2x2 (touching at x=1)
               idx = 0;
               for (int x = -1; x <= 1; x += 2) {
                   for (int y = -1; y <= 1; y += 2) {
                       for (int z = -1; z <= 1; z += 2) {
                           cube2_verts[idx * 3 + 0] = x + 2.0f;
                           cube2_verts[idx * 3 + 1] = y;
                           cube2_verts[idx * 3 + 2] = z;
                           idx++;
                       }
                   }
               }

               gkPolytope polytope1, polytope2;
               polytope1.numpoints = numVerts;
               polytope1.coord = cube1_verts;
               polytope2.numpoints = numVerts;
               polytope2.coord = cube2_verts;

               gkSimplex simplex;
               simplex.nvrtx = 0;
               gkFloat distance;
               gkFloat witness1[3], witness2[3];

               computeGJKAndEPA(1, &polytope1, &polytope2, &simplex, &distance, witness1, witness2);

               printf("  Simplex vertices: %d\n", simplex.nvrtx);
               printf("  Distance: %.6f\n", distance);
               printf("  Expected: Very small distance (near zero)\n");
               printf("  Witness 1: (%.6f, %.6f, %.6f)\n", witness1[0], witness1[1], witness1[2]);
               printf("  Witness 2: (%.6f, %.6f, %.6f)\n", witness2[0], witness2[1], witness2[2]);
               
               if (distance >= 0 && distance < 0.01f) {
                   printf(" PASS: Distance near zero as expected\n");
               } else {
                   printf(" WARNING: Distance may indicate collision or separation\n");
               }
               printf("\n");

               free(cube1_verts);
               free(cube2_verts);
            }

            // Test Case 3: Two separated cubes
            printf("Test Case 3: Two separated cubes\n");
            printf("-----------------------------------\n");
            {
               const int numVerts = 8;
               gkFloat* cube1_verts = (gkFloat*)malloc(numVerts * 3 * sizeof(gkFloat));
               gkFloat* cube2_verts = (gkFloat*)malloc(numVerts * 3 * sizeof(gkFloat));

               // Cube 1: centered at (0, 0, 0), size 2x2x2
               int idx = 0;
               for (int x = -1; x <= 1; x += 2) {
                   for (int y = -1; y <= 1; y += 2) {
                       for (int z = -1; z <= 1; z += 2) {
                           cube1_verts[idx * 3 + 0] = x;
                           cube1_verts[idx * 3 + 1] = y;
                           cube1_verts[idx * 3 + 2] = z;
                           idx++;
                       }
                   }
               }

              // Cube 2: centered at (5, 0, 0), size 2x2x2 (separated by 3 units)
              idx = 0;
              for (int x = -1; x <= 1; x += 2) {
                  for (int y = -1; y <= 1; y += 2) {
                      for (int z = -1; z <= 1; z += 2) {
                          cube2_verts[idx * 3 + 0] = x + 5.0f;
                          cube2_verts[idx * 3 + 1] = y;
                          cube2_verts[idx * 3 + 2] = z;
                          idx++;
                      }
                  }
              }

              gkPolytope polytope1, polytope2;
              polytope1.numpoints = numVerts;
              polytope1.coord = cube1_verts;
              polytope2.numpoints = numVerts;
              polytope2.coord = cube2_verts;

              gkSimplex simplex;
              simplex.nvrtx = 0;
              gkFloat distance;
              gkFloat witness1[3], witness2[3];

              computeGJKAndEPA(1, &polytope1, &polytope2, &simplex, &distance, witness1, witness2);

              printf("  Simplex vertices: %d\n", simplex.nvrtx);
              printf("  Distance: %.6f\n", distance);
              printf("  Expected: Distance ≈ 3.0 (separation between cubes)\n");
              printf("  Witness 1: (%.6f, %.6f, %.6f)\n", witness1[0], witness1[1], witness1[2]);
              printf("  Witness 2: (%.6f, %.6f, %.6f)\n", witness2[0], witness2[1], witness2[2]);
              
              if (simplex.nvrtx < 4 && distance > 2.9f && distance < 3.1f) {
                  printf(" PASS: Correct separation distance\n");
              } else if (simplex.nvrtx < 4) {
                  printf(" WARNING: Distance may be incorrect\n");
              } else {
                  printf(" FAIL: Should not detect collision\n");
              }
              printf("\n");

              free(cube1_verts);
              free(cube2_verts);
            }

            //Test Case 5: Overlapping polytopes with many vertices
            printf("Test Case 5: Overlapping polytopes (~50 vertices each)\n");
            printf("-----------------------------------\n");
            {
                const int numVerts = 50;
                gkPolytope polytope1, polytope2;
                
                // Generate polytopes that overlap
                // Polytope 1: centered at (0, 0, 0)
                gkFloat* verts1 = generatePolytope(numVerts, 0.0f, 0.0f, 0.0f);
                
                // Polytope 2: centered at (0.5, 0, 0) - overlaps with polytope 1
                gkFloat* verts2 = generatePolytope(numVerts, 0.5f, 0.0f, 0.0f);
                
                polytope1.numpoints = numVerts;
                polytope1.coord = verts1;
                polytope2.numpoints = numVerts;
                polytope2.coord = verts2;
                
                gkSimplex simplex;
                simplex.nvrtx = 0;
                gkFloat distance;
                gkFloat witness1[3], witness2[3];
                
                computeGJKAndEPA(1, &polytope1, &polytope2, &simplex, &distance, witness1, witness2);
                
                printf("  Simplex vertices: %d\n", simplex.nvrtx);
                printf("  Distance/Penetration: %.6f\n", distance);
                printf("  Witness 1: (%.6f, %.6f, %.6f)\n", witness1[0], witness1[1], witness1[2]);
                printf("  Witness 2: (%.6f, %.6f, %.6f)\n", witness2[0], witness2[1], witness2[2]);
                
                // Check penetration distance
                // For overlapping polytopes, distance should be negative (penetration depth)
                // or very small positive (just touching)
                if (simplex.nvrtx == 4) {
                    if (distance < 0.0f) {
                        printf("  PASS: Collision detected with penetration depth of %.6f\n", -distance);
                    } else if (distance < 0.1f) {
                        printf("  PASS: Collision detected (very small distance/penetration)\n");
                    } else {
                        printf("  WARNING: Collision detected but distance seems large: %.6f\n", distance);
                    }
                } else {
                    printf("  WARNING: Simplex has %d vertices (expected 4 for collision)\n", simplex.nvrtx);
                    if (distance > 0.0f) {
                        printf("  INFO: Polytopes are separated by distance: %.6f\n", distance);
                    }
                }
                printf("\n");
                
                free(verts1);
                free(verts2);
            }

            // Test Case 6: Cube and rotated cube (diamond shape) with high-resolution vertices
            printf("Test Case 6: Cube and rotated cube (45° around all axes, x+1) - High Resolution\n");
            printf("-----------------------------------\n");
            {
                const int gridSize = 40;
                const gkFloat cubeSize = 1.0f; // Half-size of cube
                const int numVerts = 6 * gridSize * gridSize; // 6 faces * gridSize^2
                
                printf("  Generating cubes with %d vertices each (40x40 grid per face)...\n", numVerts);
                
                // Cube 1: centered at (0, 0, 0), size 2x2x2
                gkFloat* cube1_verts = generateCubeWithGrid(gridSize, cubeSize, 0.0f, 0.0f, 0.0f);
                
                // Cube 2: generate at origin first, then rotate and translate
                gkFloat* cube2_verts = generateCubeWithGrid(gridSize, cubeSize, 0.0f, 0.0f, 0.0f);
                
                // Rotate all vertices of cube2 by 45° around all axes and translate by (1, 0, 0)
                const gkFloat M_PI = 3.14159265358979323846;
                const gkFloat angle = 45.0f * M_PI / 180.0f; // 45 degrees in radians
                const gkFloat cos_a = cos(angle);
                const gkFloat sin_a = sin(angle);
                
                // Apply rotation and translation to all vertices
                for (int i = 0; i < numVerts; i++) {
                    gkFloat px = cube2_verts[i * 3 + 0];
                    gkFloat py = cube2_verts[i * 3 + 1];
                    gkFloat pz = cube2_verts[i * 3 + 2];
                    
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
                    cube2_verts[i * 3 + 0] = px + 1.0f;
                    cube2_verts[i * 3 + 1] = py;
                    cube2_verts[i * 3 + 2] = pz;
                }
                
                gkPolytope polytope1, polytope2;
                polytope1.numpoints = numVerts;
                polytope1.coord = cube1_verts;
                polytope2.numpoints = numVerts;
                polytope2.coord = cube2_verts;

                gkSimplex simplex;
                simplex.nvrtx = 0;
                gkFloat distance;
                gkFloat witness1[3], witness2[3];

                printf("  Running GJK and EPA...\n");
                computeGJKAndEPA(1, &polytope1, &polytope2, &simplex, &distance, witness1, witness2);

                printf("  Simplex vertices: %d\n", simplex.nvrtx);
                printf("  Distance/Penetration: %.6f\n", distance);
                printf("  Expected: May overlap or be close depending on rotation\n");
                printf("  Witness 1: (%.6f, %.6f, %.6f)\n", witness1[0], witness1[1], witness1[2]);
                printf("  Witness 2: (%.6f, %.6f, %.6f)\n", witness2[0], witness2[1], witness2[2]);
                
                // Verify witness points are reasonable (expanded bounds for rotated cube)
                bool valid1 = (witness1[0] >= -2.0f && witness1[0] <= 2.0f) &&
                             (witness1[1] >= -2.0f && witness1[1] <= 2.0f) &&
                             (witness1[2] >= -2.0f && witness1[2] <= 2.0f);
                bool valid2 = (witness2[0] >= -1.0f && witness2[0] <= 3.0f) &&
                             (witness2[1] >= -2.0f && witness2[1] <= 2.0f) &&
                             (witness2[2] >= -2.0f && witness2[2] <= 2.0f);
                
                if (simplex.nvrtx == 4 && valid1 && valid2) {
                    if (distance < 0.0f) {
                        printf("  PASS: Collision detected with penetration depth of %.6f\n", -distance);
                    } else {
                        printf("  PASS: Collision detected, witness points valid\n");
                    }
                } else if (simplex.nvrtx < 4 && distance >= 0.0f) {
                    printf("  PASS: No collision, separation distance: %.6f\n", distance);
                } else {
                    printf("  WARNING: Unexpected results\n");
                }
                printf("\n");

                free(cube1_verts);
                free(cube2_verts);
            }

            // Test Case 7: Two overlapping spheres
            printf("Test Case 7: Two overlapping spheres (radius 2, 1000 points each)\n");
            printf("-----------------------------------\n");
            {
                const int numPoints = 1000;
                const gkFloat radius = 2.0f;
                
                printf("  Generating spheres with %d points each, radius %.2f...\n", numPoints, radius);
                
                // Sphere 1: centered at (0, 0, 0)
                gkFloat* sphere1_verts = generateSphereSurface(numPoints, radius, 0.0f, 0.0f, 0.0f);
                
                // Sphere 2: centered at (1, 0, 0) - shifted 1 unit in x direction
                // With radius 2, they overlap by 3 units (2+2-1=3), but centers are 1 unit apart
                gkFloat* sphere2_verts = generateSphereSurface(numPoints, radius, 1.0f, 0.0f, 0.0f);
                
                gkPolytope polytope1, polytope2;
                polytope1.numpoints = numPoints;
                polytope1.coord = sphere1_verts;
                polytope2.numpoints = numPoints;
                polytope2.coord = sphere2_verts;

                gkSimplex simplex;
                simplex.nvrtx = 0;
                gkFloat distance;
                gkFloat witness1[3], witness2[3];

                printf("  Running GJK and EPA...\n");
                computeGJKAndEPA(1, &polytope1, &polytope2, &simplex, &distance, witness1, witness2);

                printf("  Simplex vertices: %d\n", simplex.nvrtx);
                printf("  Distance/Penetration: %.6f\n", distance);
                printf("  Expected: Collision (spheres overlap, centers 1 unit apart, each radius 2)\n");
                printf("  Expected overlap: ~3 units (2+2-1=3)\n");
                printf("  Witness 1: (%.6f, %.6f, %.6f)\n", witness1[0], witness1[1], witness1[2]);
                printf("  Witness 2: (%.6f, %.6f, %.6f)\n", witness2[0], witness2[1], witness2[2]);
                
                // Verify witness points are within sphere bounds
                // Sphere 1: centered at (0,0,0), radius 2
                // Sphere 2: centered at (1,0,0), radius 2
                gkFloat dist1 = sqrt(witness1[0]*witness1[0] + witness1[1]*witness1[1] + witness1[2]*witness1[2]);
                gkFloat dist2 = sqrt((witness2[0]-1.0f)*(witness2[0]-1.0f) + witness2[1]*witness2[1] + witness2[2]*witness2[2]);
                
                bool valid1 = dist1 <= radius + 0.1f; // Allow small tolerance
                bool valid2 = dist2 <= radius + 0.1f;
                
                if (simplex.nvrtx == 4 && valid1 && valid2) {
                    if (distance < 0.0f) {
                        printf("  PASS: Collision detected with penetration depth of %.6f\n", -distance);
                        printf("  Expected penetration: ~3.0 units\n");
                    } else if (distance < 0.1f) {
                        printf("  PASS: Collision detected (very small distance/penetration)\n");
                    } else {
                        printf("  WARNING: Collision detected but distance seems large: %.6f\n", distance);
                    }
                } else if (simplex.nvrtx < 4 && distance >= 0.0f) {
                    printf("  WARNING: No collision detected, but spheres should overlap\n");
                    printf("  Separation distance: %.6f\n", distance);
                } else {
                    printf("  WARNING: Unexpected results\n");
                    if (!valid1) printf("    Witness 1 distance from sphere 1 center: %.6f (expected <= %.2f)\n", dist1, radius);
                    if (!valid2) printf("    Witness 2 distance from sphere 2 center: %.6f (expected <= %.2f)\n", dist2, radius);
                }
                printf("\n");

                free(sphere1_verts);
                free(sphere2_verts);
            }

            printf("========================================\n");
            printf("EPA Testing Complete\n");
            printf("========================================\n\n");
        }
    }
}
