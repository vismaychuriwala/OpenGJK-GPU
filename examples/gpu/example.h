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
    }
}

#endif // EXAMPLE_H
