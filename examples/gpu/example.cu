#include "example.h"
#include "GJK/gpu/warpParallelGJK.h"
#include <cuda_runtime.h>
#include <stdio.h>

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
    }
}
