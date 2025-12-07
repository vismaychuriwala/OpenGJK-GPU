#ifndef EXAMPLE_H
#define EXAMPLE_H

#include "GJK/gpu/openGJK.h"
#include "../common/timer.h"

namespace GJK {
    namespace GPU {
        GJK::Common::PerformanceTimer& timer();

        /**
         * Computes minimum distance between polytopes using GJK algorithm on GPU.
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
         * Computes collision information (penetration depth, witness points, and contact normal) for colliding polytopes using EPA algorithm only.
         * Takes pre-computed simplices and distances from GJK as input and runs EPA to compute
         * detailed collision information.
         * Handles all GPU memory allocation and transfers internally.
         *
         * @param n         Number of polytope pairs
         * @param bd1       Array of first polytopes (host memory)
         * @param bd2       Array of second polytopes (host memory)
         * @param simplices Array of input simplices from GJK (host memory, will be updated with results)
         * @param distances Array of input distances from GJK (host memory, will be updated with negative penetration depths for colliding objects)
         * @param witness1   Array to store witness points on first polytope (n*3 floats, host memory)
         * @param witness2   Array to store witness points on second polytope (n*3 floats, host memory)
         * @param contact_normals Optional array to store contact normals from bd1 to bd2 (n*3 floats, host memory, can be nullptr)
         */
        void computeCollisionInformation(int n,
                            const gkPolytope* bd1,
                            const gkPolytope* bd2,
                            gkSimplex* simplices,
                            gkFloat* distances,
                            gkFloat* witness1,
                            gkFloat* witness2,
                            gkFloat* contact_normals = nullptr);

        
        /**
         * Computes minimum distance and witness points between polytopes using GJK and EPA algorithms.
         * First runs GJK (warp-parallel, 16 threads per collision) to detect collisions,
         * then runs EPA (warp-parallel, 32 threads per collision) to compute penetration depth
         * and witness points for colliding polytopes.
         * Handles all GPU memory allocation and transfers internally.
         *
         * @param n         Number of polytope pairs
         * @param bd1       Array of first polytopes (host memory)
         * @param bd2       Array of second polytopes (host memory)
         * @param simplices Array to store resulting simplices (host memory)
         * @param distances Array to store distances/penetration depths (host memory)
         * @param witness1   Array to store witness points on first polytope (n*3 floats, host memory)
         * @param witness2   Array to store witness points on second polytope (n*3 floats, host memory)
         */
        void computeGJKAndEPA(int n,
                            const gkPolytope* bd1,
                            const gkPolytope* bd2,
                            gkSimplex* simplices,
                            gkFloat* distances,
                            gkFloat* witness1,
                            gkFloat* witness2);



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

        /**
         * Tests the GJK and EPA implementation with basic colliding polytopes.
         * Creates test cases with known collisions and validates the results.
         * Prints detailed output for verification.
         */
        void EPATesting();
    }
}

#endif // EXAMPLE_H
