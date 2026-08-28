#pragma once
// Kernel 2 -- Global memory coalescing. Same work as kernel 1.
// element map is changed so a warp's 32 threads touch consecutive addresses.
// BLOCKSIZE is a compile-time template constant baked into the index math.

template <const uint BLOCKSIZE>
__global__ void sgemm_coalescing(int M, int N, int K, float alpha,
                                 const float *A, const float *B, float beta,
                                 float *C) {
    const uint x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);  // row
    const uint y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);  // col
    if (x < (uint)M && y < (uint)N) {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i)
            tmp += A[x * K + i] * B[i * N + y];
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}
