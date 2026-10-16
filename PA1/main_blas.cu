#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

void init_matrix(__half *mat, int n) {
    for (int i = 0; i < n * n; i++) {
        mat[i] = __float2half((float)rand() / RAND_MAX * 2.0f - 1.0f);
    }
}

// GPU timing with CUDA events: tic() before a launch, toc() after it returns ms
cudaEvent_t evStart, evStop;
void tic() {
    cudaEventRecord(evStart);
}
float toc() {
    cudaEventRecord(evStop);
    cudaEventSynchronize(evStop);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  CUDA error: %s\n", cudaGetErrorString(err));
        return -1.0f;
    }
    float ms;
    cudaEventElapsedTime(&ms, evStart, evStop);
    return ms;
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);   // unbuffered to see progress live in the Slurm log

    int Ns[] = {64, 128, 512, 1024, 4096, 16384};
    int numN = sizeof(Ns) / sizeof(Ns[0]);

    cudaEventCreate(&evStart);
    cudaEventCreate(&evStop);

    cublasHandle_t handle;
    cublasCreate(&handle);
    __half alpha = __float2half(1.0f);   // C = alpha * A * B + beta * C
    __half beta  = __float2half(0.0f);

    for (int ni = 0; ni < numN; ni++) {
        int N = Ns[ni];
        size_t bytesH = (size_t)N * N * sizeof(__half);

        // Host matrices (random), copied to device. cublasHgemm outputs half, so C is half here.
        __half *hA = (__half *)malloc(bytesH);
        __half *hB = (__half *)malloc(bytesH);
        init_matrix(hA, N);
        init_matrix(hB, N);

        __half *dA, *dB, *dC;
        cudaMalloc(&dA, bytesH);
        cudaMalloc(&dB, bytesH);
        cudaMalloc(&dC, bytesH);
        cudaMemcpy(dA, hA, bytesH, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB, bytesH, cudaMemcpyHostToDevice);

        // cuBLAS is column-major, but matrices in C are row-major. cuBLAS reads A as A^T and B as B^T.
        // Computing B^T * A^T = (A * B)^T in column-major is exactly A * B in row-major,

        // Warm-up for every N: cuBLAS picks (and loads) a different kernel for each size
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dB, N, dA, N, &beta, dC, N);
        cudaDeviceSynchronize();

        tic();
        cublasStatus_t st = cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                        &alpha, dB, N, dA, N, &beta, dC, N);
        float ms = toc();
        if (st != CUBLAS_STATUS_SUCCESS)
            printf("cublas, N=%d, error %d\n", N, st);
        else
            printf("cublas, N=%d, time=%.3f\n", N, ms);

        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        free(hA);
        free(hB);
    }

    cublasDestroy(handle);
    cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
    return 0;
}