#include "example.h"

namespace GJK {
    namespace CPU {
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

            timer().startCpuTimer();
            for (int i = 0; i < n; i++) {
                distances[i] = compute_minimum_distance(bd1[i], bd2[i], &simplices[i]);
            }
            timer().endCpuTimer();
        }
    }
}
