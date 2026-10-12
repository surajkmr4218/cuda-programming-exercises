#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define N  512            // A, B, C are N x N
#define P  32             // p x p total threads
#define BD 8              // threads per block side 
#define T  (N / P)        // each thread computes a T x T tile of C
#define BS (BD * T)       // rows/cols of C covered by one block
#define KC 32             // chunk of the shared dimension staged in shared memory (Naive-2)

// CPU Reference (for myself to look back on)
void matmul_cpu(__half *A, __half *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++){
                sum += __half2float(A[i * n + k]) * __half2float(B[k * n + j]);
            }
            C[i * n + j] = sum;
        }
    }
}

// Naive-1: p x p threads, each owns a T x T tile of C, global memory only
__global__ void naive_1(const __half *A, const __half *B, float *C) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;   // tile column index
    int gy = blockIdx.y * blockDim.y + threadIdx.y;   // tile row index

    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {
            int row = gy * T + i;
            int col = gx * T + j;

            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += __half2float(A[row * N + k]) * __half2float(B[k * N + col]);
            }
            C[row * N + col] = sum;
        }
    }
}

void init_matrix(__half *mat, int n) {
    for (int i = 0; i < n * n; i++) {
        mat[i] = __float2half((float)rand() / RAND_MAX * 2.0f - 1.0f);
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    size_t size_h = (size_t)N * N * sizeof(__half);
    size_t size_f = (size_t)N * N * sizeof(float);

    __half *h_A = (__half *)malloc(size_h);
    __half *h_B = (__half *)malloc(size_h);
    float *h_C_cpu = (float *)malloc(size_f);
    float *h_C_gpu = (float *)malloc(size_f);

    srand(42);
    init_matrix(h_A, N);
    init_matrix(h_B, N);

    __half *d_A, *d_B;
    float  *d_C;
    cudaMalloc(&d_A, size_h);
    cudaMalloc(&d_B, size_h);
    cudaMalloc(&d_C, size_f);
    cudaMemcpy(d_A, h_A, size_h, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_h, cudaMemcpyHostToDevice);

    dim3 blockDim(BD, BD);
    dim3 gridDim(P / BD, P / BD);
    printf("N=%d P=%d T=%d | blockNum=%d threadNum=%d\n",
           N, P, T, (P / BD) * (P / BD), BD * BD);

    // matmul_cpu(h_A, h_B, h_C_cpu, N);

    // ---------- Naive-1 ----------
    printf("Benchmarking Naive-1...\n");
    for (int i = 0; i < 3; i++) {                       // warm-up
        naive_1<<<gridDim, blockDim>>>(d_A, d_B, d_C);
        cudaDeviceSynchronize();
    }
    double total1 = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        naive_1<<<gridDim, blockDim>>>(d_A, d_B, d_C);
        cudaDeviceSynchronize();
        double end_time = get_time();
        total1 += end_time - start_time;
    }
    cudaMemcpy(h_C_gpu, d_C, size_f, cudaMemcpyDeviceToHost);
    printf("Naive-1 average time: %f microseconds\n", total1 / 20.0 * 1e6);
}