#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

// ---------------------------------------------------------------------------
// Tensor core MM (WMMA)
// tensorMM<<<blockNum, threadNum>>>(A, B, C, N)
// threadNum = 32 * warpsPerBlock, blockNum = (N/16)^2 / warpsPerBlock
// One warp computes one 16x16 tile of C. Each step loads a 16x16 piece of A
// and of B and the tensor core does c += a * b in one instruction.
// ---------------------------------------------------------------------------
__global__ void tensorMM(const __half *A, const __half *B, float *C, int N) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;   // global warp id = tile id
    int tilesPerRow = N / 16;
    if (warp >= tilesPerRow * tilesPerRow) return;
    int row = (warp / tilesPerRow) * 16;    // first C row of this tile
    int col = (warp % tilesPerRow) * 16;    // first C col of this tile

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);

    for (int k = 0; k < N; k += 16) {
        wmma::load_matrix_sync(a, A + row * N + k, N);   // 16 x 16 tile of a
        wmma::load_matrix_sync(b, B + k * N + col, N);   // 16 x 16 tile of b
        wmma::mma_sync(c, a, b, c);                      // c += a * b on the tensor core
    }
    wmma::store_matrix_sync(C + row * N + col, c, N, wmma::mem_row_major);
}