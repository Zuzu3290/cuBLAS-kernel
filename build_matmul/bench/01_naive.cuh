#pragma once
// Kernel 1 -- Naive. One thread per element of C computes a full dot product.

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;  // row
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;  // col
    if (x < (uint)M && y < (uint)N) {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i)
            tmp += A[x * K + i] * B[i * N + y];
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}
