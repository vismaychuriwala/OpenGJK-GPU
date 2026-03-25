#ifndef EXAMPLE_CPU_H
#define EXAMPLE_CPU_H

#include "GJK/cpu/openGJK.h"
#include "GJK/cpu/EPA.h"
#include "../common/timer.h"

namespace GJK {
    namespace CPU {
        GJK::Common::PerformanceTimer& timer();

        /**
         * Computes minimum distance between polytopes on CPU.
         *
         * @param n         Number of polytope pairs
         * @param bd1       Array of first polytopes
         * @param bd2       Array of second polytopes
         * @param simplices Array to store resulting simplices
         * @param distances Array to store distances
         */
        void computeDistances(int n,
                            const gkPolytope* bd1,
                            const gkPolytope* bd2,
                            gkSimplex* simplices,
                            gkFloat* distances);

        /**
         * Computes collision information (witness points, contact normal) using EPA on CPU.
         * Takes pre-computed simplices and distances from GJK as input.
         *
         * @param n               Number of polytope pairs
         * @param bd1             Array of first polytopes
         * @param bd2             Array of second polytopes
         * @param simplices       Input simplices from GJK (updated with EPA results)
         * @param distances       Input distances from GJK (updated with penetration depths)
         * @param witness1        Output witness points on first polytope (n*3 floats)
         * @param witness2        Output witness points on second polytope (n*3 floats)
         * @param contact_normals Optional output contact normals (n*3 floats, can be nullptr)
         */
        void computeCollisionInformation(int n,
                            const gkPolytope* bd1,
                            const gkPolytope* bd2,
                            gkSimplex* simplices,
                            gkFloat* distances,
                            gkFloat* witness1,
                            gkFloat* witness2,
                            gkFloat* contact_normals = nullptr);
    }
}

#endif // EXAMPLE_CPU_H
