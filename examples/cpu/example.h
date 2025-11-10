#ifndef EXAMPLE_CPU_H
#define EXAMPLE_CPU_H

#include "GJK/cpu/openGJK.h"
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
    }
}

#endif // EXAMPLE_CPU_H
