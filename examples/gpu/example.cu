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

                    // Warm up GPU
                    GJK::GPU::computeDistances(numPolytopes, polytopes1, polytopes2, warm_up_simplices, warm_up_distances);
                    GJK::GPU::timer().getGpuElapsedTimeForPreviousOperation(); // Discard warm-up time

                    // Test CPU implementation
                    GJK::CPU::computeDistances(numPolytopes, polytopes1, polytopes2, simplices_cpu, distances_cpu);
                    float cpu_time = GJK::CPU::timer().getCpuElapsedTimeForPreviousOperation();

                    // Test GPU implementation
                    GJK::GPU::computeDistances(numPolytopes, polytopes1, polytopes2, simplices_gpu, distances_gpu);
                    float gpu_time = GJK::GPU::timer().getGpuElapsedTimeForPreviousOperation();

                    // Test Warp-Parallel GPU implementation
                    GJK::GPU::computeDistancesWarpParallel(numPolytopes, polytopes1, polytopes2, simplices_warp, distances_warp);
                    float warp_gpu_time = GJK::GPU::timer().getGpuElapsedTimeForPreviousOperation();

                    // Write results to CSV
                    fprintf(fp, "%d,%d,%.6f,%.6f,%.6f\n", 
                            numPolytopes, numVertices, cpu_time, gpu_time, warp_gpu_time);

                    // Free memory
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
            }

            fclose(fp);
            printf("Performance testing complete! Results saved to %s\n", outputFile);
        }
    }
}
