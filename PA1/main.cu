#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "kernels.cuh"

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

// Shared memory (bytes) one Naive-2 block needs: its full A rows + full B columns.
// Must match the rows/cols formulas inside the naive2 kernel.
size_t naive2_smem(int N, int p, int tn) {
    int T = N / p;
    int rows = (tn >= p ? tn / p : 1) * T;   // A rows the block needs
    int cols = (tn >= p ? p : tn) * T;       // B cols the block needs
    return (size_t)(rows + cols) * N * sizeof(__half);
}

int main() {
    int Ns[] = {64, 128, 512, 1024, 4096, 16384};
    int Ps[] = {1, 16, 32, 64};
    int numN = sizeof(Ns) / sizeof(Ns[0]);
    int numP = sizeof(Ps) / sizeof(Ps[0]);

    // Naive-1/2 experiments from the write-up table, same for every N: {p, blockNum, threadNum}
    int configs[][3] = {
        {1,  1,   1},      // single
        {16, 1,   256},    // A: few big blocks
        {16, 8,   32},     // B: many small blocks
        {32, 1,   1024},   // A
        {32, 32,  32},     // B
        {64, 4,   1024},   // A
        {64, 128, 32},     // B
    };
    int numConfigs = sizeof(configs) / sizeof(configs[0]);

    cudaEventCreate(&evStart);
    cudaEventCreate(&evStop);

    // Let naive2 use the device's max (opt-in) shared memory per block
    int maxSmem;
    cudaDeviceGetAttribute(&maxSmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    cudaFuncSetAttribute(naive2, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSmem);

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

        // Warm-up once: the first launch of each kernel includes module-load overhead
        if (ni == 0) {
            naive1<<<1, 1>>>(dA, dB, dC, N, N);
            naive2<<<1, 1, naive2_smem(N, N, 1)>>>(dA, dB, dC, N, N);
            blockMM<<<dim3(1, 1), dim3(1, 1)>>>(dA, dB, dC, N, 1);
            cudaDeviceSynchronize();
        }

        // ---- Naive-1 and Naive-2: only the configs in the table ----
        for (int c = 0; c < numConfigs; c++) {
            int p  = configs[c][0];
            int bn = configs[c][1];
            int tn = configs[c][2];

            // Naive-1: no shared memory, runs for every config
            tic();
            naive1<<<bn, tn>>>(dA, dB, dC, N, p);
            printf("naive1, N=%d, p=%d, bn=%d, tn=%d, time=%.3f\n", N, p, bn, tn, toc());

            // Naive-2: skip if the block's full A rows + B columns don't fit in shared memory
            size_t smem = naive2_smem(N, p, tn);
            if (smem > (size_t)maxSmem) {
                printf("naive2, N=%d, p=%d, bn=%d, tn=%d, skipped (needs %zu KB smem > %d KB)\n",
                       N, p, bn, tn, smem / 1024, maxSmem / 1024);
                continue;
            }
            tic();
            naive2<<<bn, tn, smem>>>(dA, dB, dC, N, p);
            printf("naive2, N=%d, p=%d, bn=%d, tn=%d, time=%.3f\n", N, p, bn, tn, toc());
        }

        // ---- Block MM: fixed layout, (N/p)^2 blocks of p x p threads ----
        // p x p threads per block must be <= 1024, so p = 64 is not possible
        for (int pi = 0; pi < numP; pi++) {
            int p = Ps[pi];
            if (p * p > 1024) {
                printf("blockMM, N=%d, p=%d,-,-,skipped (p*p > 1024)\n", N, p);
                continue;
            }
            tic();
            blockMM<<<dim3(N / p, N / p), dim3(p, p)>>>(dA, dB, dC, N, p);
            printf("blockMM, N=%d, p=%d, (N/p)*(N/p)=%d, p*p=%d, time=%.3f\n",
                   N, p, (N / p) * (N / p), p * p, toc());
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