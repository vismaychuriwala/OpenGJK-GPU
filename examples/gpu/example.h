#ifndef EXAMPLE_H
#define EXAMPLE_H

#include "GJK/gpu/openGJK.h"
#include "../common/timer.h"

namespace GJK {
    namespace GPU {
        GJK::Common::PerformanceTimer& timer();

        /**
         * Computes minimum distance between polytopes on GPU.
         * Handles all GPU memory allocation and transfers internally.
         *
         * @param n         Number of polytope pairs
         * @param bd1       Array of first polytopes (host memory)
         * @param bd2       Array of second polytopes (host memory)
         * @param simplices Array to store resulting simplices (host memory)
         * @param distances Array to store distances (host memory)
         */
        void computeDistances(int n,
                            const gkPolytope* bd1,
                            const gkPolytope* bd2,
                            gkSimplex* simplices,
                            gkFloat* distances);

        /**
         * Computes minimum distance between polytopes on GPU using warp-parallel implementation.
         * Uses 16 threads per collision for improved performance.
         * Handles all GPU memory allocation and transfers internally.
         *
         * @param n         Number of polytope pairs
         * @param bd1       Array of first polytopes (host memory)
         * @param bd2       Array of second polytopes (host memory)
         * @param simplices Array to store resulting simplices (host memory)
         * @param distances Array to store distances (host memory)
         */
        void computeDistancesWarpParallel(int n,
                            const gkPolytope* bd1,
                            const gkPolytope* bd2,
                            gkSimplex* simplices,
                            gkFloat* distances);

        /**
         * Runs performance tests for all combinations of polytope counts and vertex counts.
         * Tests CPU, GPU, and warp-parallel GPU implementations and saves results to CSV.
         *
         * @param numPolytopesArray     Array of polytope counts to test
         * @param numPolytopesArraySize Size of numPolytopesArray
         * @param numVerticesArray       Array of vertex counts per polytope to test
         * @param numVerticesArraySize  Size of numVerticesArray
         * @param outputFile            Path to output CSV file
         */
        void testing(const int* numPolytopesArray,
                     int numPolytopesArraySize,
                     const int* numVerticesArray,
                     int numVerticesArraySize,
                     const char* outputFile);
    }
}

#endif // EXAMPLE_H
