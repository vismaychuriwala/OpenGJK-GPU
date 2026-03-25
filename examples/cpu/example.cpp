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

        void computeCollisionInformation(int n,
                            const gkPolytope* bd1,
                            const gkPolytope* bd2,
                            gkSimplex* simplices,
                            gkFloat* distances,
                            gkFloat* witness1,
                            gkFloat* witness2,
                            gkFloat* contact_normals) {
            if (n <= 0) return;

            gkFloat dummy_normal[3];
            timer().startCpuTimer();
            for (int i = 0; i < n; i++) {
                gkFloat* cn = contact_normals ? &contact_normals[i*3] : dummy_normal;
                compute_epa(&bd1[i], &bd2[i], &simplices[i], &distances[i],
                                   &witness1[i*3], &witness2[i*3], cn);
            }
            timer().endCpuTimer();
        }
    }
}
