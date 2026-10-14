#include <cuda_fp16.h>

// Naive-1: p x p threads, each owns an (N/P) x (N/P) tile of C, global memory only
__global__ void naive_1(const __half *A, const __half *B, float *C, int N, int P) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;   // tile column index
    int gy = blockIdx.y * blockDim.y + threadIdx.y;   // tile row index

    for (int i = 0; i < N / P; i++) {
        for (int j = 0; j < N / P; j++) {
            int row = gy * (N / P) + i;
            int col = gx * (N / P) + j;

            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += __half2float(A[row * N + k]) * __half2float(B[k * N + col]);
            }
            C[row * N + col] = sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Naive-1 (global memory only)
// naive1<<<blockNum, threadNum>>>(A, B, C, N, p)      blockNum * threadNum = p*p
// Thread tid owns the T x T tile of C at tile-row tid/p, tile-col tid%p (T = N/p).
// For each element of its tile it reads a full row of A and column of B from
// global memory, does the dot product, and writes the result to C.
// ---------------------------------------------------------------------------

__global__ void naive1(const __half *A, const __half *B, float *C, int N, int p) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= p * p) return;

    int T = N / p;
    int row0 = (tid / p) * T;    // first C row of this tile 
    int col0 = (tid % p) * T;    // first C col of this tile

    for (int i = row0; i < row0 + T; i++)
        for (int j = col0; j < col0 + T; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
                sum += __half2float(A[i * N + k]) * __half2float(B[k * N + j]);
            C[i * N + j] = sum;
        }
}

// ---------------------------------------------------------------------------
// Naive-2 (shared-memory staging)
// Same thread -> tile ownership as Naive-1. The threads of a block together copy
// the rows of A and columns of B that the block needs into shared memory, kc
// elements along k at a time, then each thread computes from shared memory.
//
// naive2<<<blockNum, threadNum, smemBytes>>>(A, B, C, N, p, kc)
//   blockNum * threadNum = p*p, p and threadNum powers of 2, kc divides N
//   smemBytes = (rows + cols) * kc * sizeof(__half), where
//     rows = (threadNum >= p ? threadNum / p : 1) * (N / p)   // A rows a block needs
//     cols = (threadNum >= p ? p : threadNum)     * (N / p)   // B cols a block needs
//   > 48 KB needs cudaFuncSetAttribute(naive2, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes)
// ---------------------------------------------------------------------------
__global__ void naive2(const __half *A, const __half *B, float *C, int N, int p, int kc) {
    extern __shared__ __half smem[];

    int T = N / p;
    int first = blockIdx.x * blockDim.x;                    // first tid in this block
    int rows = (blockDim.x >= p ? blockDim.x / p : 1) * T;  // A rows this block needs
    int cols = (blockDim.x >= p ? p : blockDim.x) * T;      // B cols this block needs
    int rowBase = (first / p) * T;                          // first of those rows / cols
    int colBase = (first % p) * T;
    __half *sA = smem;                                      // rows x kc slice of A
    __half *sB = smem + rows * kc;                          // kc x cols slice of B

    int tid = first + threadIdx.x;
    int row0 = (tid / p) * T;                               // first C row of this tile 
    int col0 = (tid % p) * T;                               // first C col of this tile

    for (int k0 = 0; k0 < N; k0 += kc) {
        // 1) whole block loads A[its rows][k0 .. k0+kc-1] and B[k0 .. k0+kc-1][its cols]
        for (int x = threadIdx.x; x < rows * kc; x += blockDim.x)
            sA[x] = A[(rowBase + x / kc) * N + k0 + x % kc];
        for (int x = threadIdx.x; x < kc * cols; x += blockDim.x)
            sB[x] = B[(k0 + x / cols) * N + colBase + x % cols];
        __syncthreads();

        // 2) each thread adds this slice's contribution to its T x T outputs
        for (int i = row0; i < row0 + T; i++)
            for (int j = col0; j < col0 + T; j++) {
                float sum = (k0 == 0) ? 0.0f : C[i * N + j];   // first slice overwrites C, so no memset needed
                for (int k = 0; k < kc; k++)
                    sum += __half2float(sA[(i - rowBase) * kc + k]) * __half2float(sB[k * cols + (j - colBase)]);
                C[i * N + j] = sum;
            }
        __syncthreads();   // all threads done with this slice before the next load overwrites it
    }
}

// ---------------------------------------------------------------------------
// Block MM (tiled, as in lecture)
// blockMM<<<dim3(N/p, N/p), dim3(p, p)>>>(A, B, C, N, p)      p <= 32 (p*p <= 1024 threads/block)
// Block (bx, by) computes the p x p tile of C starting at (by*p, bx*p); thread (tx, ty)
// computes one element of it. Each step loads a p x p tile of A and of B into shared memory.
// ---------------------------------------------------------------------------
#define MAXP 32
__global__ void blockMM(const __half *A, const __half *B, float *C, int N, int p) {
    __shared__ __half sA[MAXP][MAXP];
    __shared__ __half sB[MAXP][MAXP];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * p + ty;
    int col = blockIdx.x * p + tx;

    float sum = 0.0f;
    for (int k0 = 0; k0 < N; k0 += p) {
        sA[ty][tx] = A[row * N + k0 + tx];     // A tile: this block's rows, cols k0 .. k0+p-1
        sB[ty][tx] = B[(k0 + ty) * N + col];   // B tile: rows k0 .. k0+p-1, this block's cols
        __syncthreads();
        for (int k = 0; k < p; k++)
            sum += __half2float(sA[ty][k]) * __half2float(sB[k][tx]);
        __syncthreads();
    }
    C[row * N + col] = sum;
}