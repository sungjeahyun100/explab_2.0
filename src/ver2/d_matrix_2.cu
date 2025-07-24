#include "d_matrix_2.hpp"

namespace d_matrix_ver2 {
    // cuRAND state initialization kernel implementation
    __global__ void initCurandStates(curandState *states, unsigned long long seed, int total) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total) return;
        curand_init(seed, idx, 0, &states[idx]);
    }
}
