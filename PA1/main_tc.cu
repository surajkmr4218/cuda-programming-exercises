#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "kernels_tc.cuh"

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

    for (int ni = 0; ni < numN; ni++) {
        int N = Ns[ni];
        size_t bytesH = (size_t)N * N * sizeof(__half);
        size_t bytesF = (size_t)N * N * sizeof(float);

        // Host matrices (random), copied to device
        __half *hA = (__half *)malloc(bytesH);
        __half *hB = (__half *)malloc(bytesH);
        init_matrix(hA, N);
        init_matrix(hB, N);

        __half *dA, *dB;
        float *dC;
        cudaMalloc(&dA, bytesH);
        cudaMalloc(&dB, bytesH);
        cudaMalloc(&dC, bytesF);
        cudaMemcpy(dA, hA, bytesH, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB, bytesH, cudaMemcpyHostToDevice);

        // Warm-up once: the first launch includes module-load overhead
        if (ni == 0) {
            tensorMM<<<1, 32>>>(dA, dB, dC, N);
            cudaDeviceSynchronize();
        }

        // One warp per 16x16 tile of C. Sweep warps per block: 1, 2, 4, ..., 32
        int tiles = (N / 16) * (N / 16);
        for (int wpb = 1; wpb <= 32; wpb *= 2) {
            if (wpb > tiles) break;          // N = 64 only has 16 tiles
            int tn = 32 * wpb;               // threadNum
            int bn = tiles / wpb;            // blockNum
            tic();
            tensorMM<<<bn, tn>>>(dA, dB, dC, N);
            printf("tensor, N=%d, bn=%d, tn=%d, time=%.3f\n", N, bn, tn, toc());
        }

        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        free(hA);
        free(hB);
    }

    cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
    return 0;
}