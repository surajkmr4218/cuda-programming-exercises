#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "kernels.cuh"

#define MAX_THREAD_WORK 2e9

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
    int Ns[] = {64, 128, 512, 1024, 4096, 16384};
    int Ps[] = {1, 16, 32, 64};
    int numN = sizeof(Ns) / sizeof(Ns[0]);
    int numP = sizeof(Ps) / sizeof(Ps[0]);

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
            naive2<<<1, 1, 2 * N * sizeof(__half)>>>(dA, dB, dC, N, N, N);
            blockMM<<<dim3(1, 1), dim3(1, 1)>>>(dA, dB, dC, N, 1);
            cudaDeviceSynchronize();
        }

        for (int pi = 0; pi < numP; pi++) {
            int p = Ps[pi];
            int total = p * p;   // total threads for Naive-1/2
            bool tooSlow = (double)N * N * N / total > MAX_THREAD_WORK;

            // ---- Naive-1 and Naive-2: blockNum * threadNum = p*p ----
            // threadNum = 1, 2, 4, ... up to min(p*p, 1024)
            for (int tn = 1; tn <= total && tn <= 1024; tn *= 2) {
                int bn = total / tn;

                if (tooSlow) {
                    printf("naive1,%d,%d,%d,%d,skipped\n", N, p, bn, tn);
                    printf("naive2,%d,%d,%d,%d,skipped\n", N, p, bn, tn);
                    continue;
                }

                // Naive-1
                tic();
                naive1<<<bn, tn>>>(dA, dB, dC, N, p);
                printf("naive1,%d,%d,%d,%d,%.3f\n", N, p, bn, tn, toc());

                // Naive-2: pick the largest kc (power of 2) whose slice fits in shared memory
                int T = N / p;
                int rows = (tn >= p ? tn / p : 1) * T;
                int cols = (tn >= p ? p : tn) * T;
                int kc = N;
                while (kc > 1 && (size_t)(rows + cols) * kc * sizeof(__half) > (size_t)maxSmem)
                    kc /= 2;
                size_t smem = (size_t)(rows + cols) * kc * sizeof(__half);
                if (smem > (size_t)maxSmem) {
                    printf("naive2,%d,%d,%d,%d,skipped\n", N, p, bn, tn);
                    continue;
                }
                tic();
                naive2<<<bn, tn, smem>>>(dA, dB, dC, N, p, kc);
                printf("naive2,%d,%d,%d,%d,%.3f\n", N, p, bn, tn, toc());
            }

            // ---- Block MM: fixed layout, (N/p)^2 blocks of p x p threads ----
            // p x p threads per block must be <= 1024, so p = 64 is not possible
            if (p <= 32) {
                tic();
                blockMM<<<dim3(N / p, N / p), dim3(p, p)>>>(dA, dB, dC, N, p);
                printf("blockMM,%d,%d,%d,%d,%.3f\n", N, p, (N / p) * (N / p), p * p, toc());
            } else {
                printf("blockMM,%d,%d,-,-,skipped (p*p > 1024)\n", N, p);
            }
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