#include <cuda_fp16.h>

// CPU Reference (for myself to look back on)
void matmul_cpu(__half *A, __half *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += __half2float(A[i * n + k]) * __half2float(B[k * n + j]);
            }
            C[i * n + j] = sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Naive-1 (global memory only)
// naive1<<<blockNum, threadNum>>>(A, B, C, N, p)   
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
// Same thread to tile ownership as Naive-1. The threads of a block together copy
// the FULL rows of A and FULL columns of B that the block needs into shared memory.
//
// naive2<<<blockNum, threadNum, smemBytes>>>(A, B, C, N, p)
//   blockNum * threadNum = p*p, p and threadNum powers of 2
//   smemBytes = (rows + cols) * N * sizeof(__half), where
//     rows = (threadNum >= p ? threadNum / p : 1) * (N / p)   // A rows a block needs
//     cols = (threadNum >= p ? p : threadNum)     * (N / p)   // B cols a block needs
//   If smemBytes > the GPU's per-block limit, skip the experiment.
// ---------------------------------------------------------------------------
__global__ void naive2(const __half *A, const __half *B, float *C, int N, int p) {
    extern __shared__ __half smem[];

    int T = N / p;
    int first = blockIdx.x * blockDim.x;      // first tid in this block
    int last    = first + blockDim.x - 1;     // last tid in this block

    int rowBase = (first / p) * T;            // top row of the FIRST thread's tile
    int colBase = (first % p) * T;            // left col of the FIRST thread's tile
    int rowEnd  = (last / p) * T + T;         // one past the bottom row of the LAST thread's tile
    int colEnd  = (last % p) * T + T;         // one past the right col of the LAST thread's tile
    int rows = rowEnd - rowBase;              // how many A rows the block needs
    int cols = colEnd - colBase;              // how many B cols the block needs
    __half *sA = smem;                        // full rows of A
    __half *sB = smem + rows * N;             // full columns of B

    // 1) whole block loads its A rows and B columns (each thread copies every blockDim.x-th element)
    for (int x = threadIdx.x; x < rows * N; x += blockDim.x)
        sA[x] = A[rowBase * N + x];                         
    for (int x = threadIdx.x; x < N * cols; x += blockDim.x)
        sB[x] = B[(x / cols) * N + colBase + x % cols];     
    __syncthreads();                           // wait until shared memory is full

    // 2) each thread computes its T x T tile of C entirely from shared memory
    int tid = first + threadIdx.x;
    int row0 = (tid / p) * T;                  // first C row of this tile
    int col0 = (tid % p) * T;                  // first C col of this tile

    for (int i = row0; i < row0 + T; i++)
        for (int j = col0; j < col0 + T; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
                sum += __half2float(sA[(i - rowBase) * N + k]) * __half2float(sB[k * cols + (j - colBase)]);
            C[i * N + j] = sum;
        }
}

// ---------------------------------------------------------------------------
// Block MM (tiled, as in lecture)
// blockMM<<<dim3(N/p, N/p), dim3(p, p)>>>(A, B, C, N, p)    
// p <= 32 (p*p <= 1024 threads/block)
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