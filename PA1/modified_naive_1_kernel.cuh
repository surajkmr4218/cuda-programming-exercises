#include <cuda_fp16.h>

// ---------------------------------------------------------------------------
// Naive-1 coalesced
// naive1_coalesced<<<blockNum, threadNum>>>(AT, B, C, N, p)    AT = A transposed
// Thread tid = (r, c) owns C rows r, r+p, r+2p, ... and cols c, c+p, c+2p, ... (cyclic).
// Neighboring threads read neighboring elements of AT and B.
// ---------------------------------------------------------------------------
__global__ void naive1_coalesced(const __half *AT, const __half *B, float *C, int N, int p) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= p * p) return;

    int r = tid / p;    // first C row
    int c = tid % p;    // first C col

    for (int i = r; i < N; i += p)
        for (int j = c; j < N; j += p) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
                sum += __half2float(AT[k * N + i]) * __half2float(B[k * N + j]);
            C[i * N + j] = sum;
        }
}