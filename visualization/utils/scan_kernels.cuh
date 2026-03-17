// scan_kernels.cuh - Efficient parallel scan kernels
// Adapted from GPU Gems 3 / CUDA stream compaction
#pragma once

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> (LOG_NUM_BANKS))

#define SCAN_BLOCK_SIZE 256
#define SCAN_B (2 * SCAN_BLOCK_SIZE)  // 512 elements per block

// Single-block scan kernel for small arrays
// Performs exclusive prefix sum on input array
// Block size must be power of 2, and 2*blockDim.x >= n
__global__ void block_scan_kernel(int n, int* g_odata, const int* g_idata, int* block_sum)
{
    int thid = threadIdx.x;
    int offset = 1;
    extern __shared__ int temp[];

    int ai = thid;
    int bi = thid + blockDim.x;
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    // Load input into shared memory
    temp[ai + bankOffsetA] = (ai < n) ? g_idata[ai] : 0;
    temp[bi + bankOffsetB] = (bi < n) ? g_idata[bi] : 0;

    // Up-sweep (reduce) phase
    for (int d = blockDim.x; d > 0; d >>= 1)
    {
        __syncthreads();
        if (thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    // Store total sum and clear last element
    if (thid == 0)
    {
        if (block_sum != nullptr) {
            *block_sum = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
        }
        temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
    }

    // Down-sweep phase
    for (int d = 1; d < n; d <<= 1)
    {
        offset >>= 1;
        __syncthreads();
        if (thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();

    // Write results to global memory
    if (ai < n) g_odata[ai] = temp[ai + bankOffsetA];
    if (bi < n) g_odata[bi] = temp[bi + bankOffsetB];
}

// Multi-block scan kernel
// Each block scans B elements and outputs a block sum
__global__ void multi_block_scan_kernel(int n, int B, int* g_odata, const int* g_idata, int* blockSums)
{
    int thid = threadIdx.x;
    int base = B * blockIdx.x;
    int offset = 1;
    extern __shared__ int temp[];

    int ai = thid;
    int bi = thid + (B / 2);
    int ga = base + ai;
    int gb = base + bi;
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    // Load input
    temp[ai + bankOffsetA] = (ga < n) ? g_idata[ga] : 0;
    temp[bi + bankOffsetB] = (gb < n) ? g_idata[gb] : 0;

    // Up-sweep
    for (int d = B >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    // Store block sum and clear last element
    if (thid == 0)
    {
        blockSums[blockIdx.x] = temp[B - 1 + CONFLICT_FREE_OFFSET(B - 1)];
        temp[B - 1 + CONFLICT_FREE_OFFSET(B - 1)] = 0;
    }

    // Down-sweep
    for (int d = 1; d < B; d <<= 1)
    {
        offset >>= 1;
        __syncthreads();
        if (thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();

    // Write results
    if (ga < n) g_odata[ga] = temp[ai + bankOffsetA];
    if (gb < n) g_odata[gb] = temp[bi + bankOffsetB];
}

// Add scanned block sums to each block's elements
__global__ void uniform_add_kernel(int n, int* odata, const int* blockIncr, int B)
{
    int base = blockIdx.x * B;
    int offset = blockIncr[blockIdx.x];

    int i = base + threadIdx.x;
    int j = base + threadIdx.x + (B / 2);

    if (i < n) odata[i] += offset;
    if (j < n) odata[j] += offset;
}

// Helper: Compute next power of 2
__host__ inline int next_power_of_2(int n) {
    int power = 1;
    while (power < n) power <<= 1;
    return power;
}

// Helper: Compute log2 ceiling
__host__ inline int ilog2ceil(int n) {
    int log = 0;
    int pow = 1;
    while (pow < n) {
        pow <<= 1;
        log++;
    }
    return log;
}

// ============================================================================
// PRE-ALLOCATED RECURSIVE MULTI-BLOCK SCAN
// ============================================================================

// Compute total auxiliary ints needed for recursive scan of n elements.
// Each level needs 2 * next_pow2(num_blocks) ints (blockSums + blockIncr).
__host__ inline int scan_aux_size(int n) {
    int total = 0;
    while (n > SCAN_B) {
        int nb = (n + SCAN_B - 1) / SCAN_B;
        int nb2 = next_power_of_2(nb);
        total += 2 * nb2;
        n = nb2;
    }
    return total;
}

// Recursive multi-block exclusive scan using pre-allocated auxiliary buffer.
// aux must point to at least scan_aux_size(n) ints of device memory.
void recursive_scan(int n, int* d_out, const int* d_in, int* aux) {
    int numBlocks = (n + SCAN_B - 1) / SCAN_B;
    int nb2 = next_power_of_2(numBlocks);

    int smem = (SCAN_B + CONFLICT_FREE_OFFSET(SCAN_B)) * sizeof(int);

    int* blockSums = aux;
    int* blockIncr = aux + nb2;
    int* next_aux  = aux + 2 * nb2;

    if (nb2 > numBlocks)
        cudaMemset(blockSums + numBlocks, 0, (nb2 - numBlocks) * sizeof(int));

    multi_block_scan_kernel<<<numBlocks, SCAN_BLOCK_SIZE, smem>>>(
        n, SCAN_B, d_out, d_in, blockSums);

    if (numBlocks > 1) {
        if (nb2 <= SCAN_B) {
            int bsz = nb2 / 2;
            int smem2 = (nb2 + CONFLICT_FREE_OFFSET(nb2)) * sizeof(int);
            block_scan_kernel<<<1, bsz, smem2>>>(nb2, blockIncr, blockSums, nullptr);
        } else {
            recursive_scan(nb2, blockIncr, blockSums, next_aux);
        }
        uniform_add_kernel<<<numBlocks, SCAN_BLOCK_SIZE>>>(n, d_out, blockIncr, SCAN_B);
    }
}
