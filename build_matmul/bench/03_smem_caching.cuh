#pragma once
// Kernel 3 -- Shared-memory cache blocking. The block cooperatively stages a
// BLOCKSIZE x BLOCKSIZE tile of A and B into on-chip shared memory, syncs, then
// every thread computes partial dot products from that fast cache.
//
// __shared__ declares per-block on-chip memory. __syncthreads() is a block-wide
// barrier -- both barriers below are mandatory (one after filling the cache, one
// before overwriting it) or fast threads race slow ones.

template <const uint BLOCKSIZE>
__global__ void sgemm_smem(int M, int N, int K, float alpha, const float *A,
                           const float *B, float beta, float *C) {
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    const uint threadRow = threadIdx.x / BLOCKSIZE;
    const uint threadCol = threadIdx.x % BLOCKSIZE;

    A += cRow * BLOCKSIZE * K;
    B += cCol * BLOCKSIZE;
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

    float tmp = 0.0f;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];
        __syncthreads();

        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx)
            tmp += As[threadRow * BLOCKSIZE + dotIdx] *
                   Bs[dotIdx * BLOCKSIZE + threadCol];
        __syncthreads();
    }
    C[threadRow * N + threadCol] =
        alpha * tmp + beta * C[threadRow * N + threadCol];
}
