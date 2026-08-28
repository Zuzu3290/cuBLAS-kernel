#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

typedef unsigned int uint;
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define BLOCKSIZE 32

// Kernel 3 -- shared memory caching.
// Shift a block chunk of A and B in the SMEM. 
__global__ void sgemm_smem(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
    // the computation point of operation moves from a column or uncontrolled matrix calcualktion
    // The operation particularily excutes in a row, wrapper,. based computation. 
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    const uint threadRow = threadIdx.x / BLOCKSIZE;
    const uint threadCol = threadIdx.x % BLOCKSIZE;
    // advanced pointers 
    A += cRow * BLOCKSIZE * K;
    B += cCol * BLOCKSIZE;
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

    float tmp = 0.0f;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];
        __syncthreads();  // <- the "controlled" operation: every thread in the block
                          //    waits here, so none of them races ahead independently

        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx)
            tmp += As[threadRow * BLOCKSIZE + dotIdx] * Bs[dotIdx * BLOCKSIZE + threadCol];
        __syncthreads();
    }
    C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}
// cRow, cCol: which BLOCKSIZE x BLOCKSIZE tile of C this block owns,
// out of the CEIL_DIV(M,BLOCKSIZE) x CEIL_DIV(N,BLOCKSIZE) grid of tiles.

int main() {
    // BLOCKSIZE=32 tiles both M/N and K with no boundary check, so the sizes
    // below must be exact multiples of it. Using exactly one tile (32) keeps
    // the k-loop to a single iteration -- smallest size that actually runs.
    int M = 32, N = 32, K = 32;
    float alpha = 2.0f, beta = 1.0f;

    float *A, *B, *C;
    cudaMallocManaged(&A, M * K * sizeof(float));
    cudaMallocManaged(&B, K * N * sizeof(float));
    cudaMallocManaged(&C, M * N * sizeof(float));

    for (int i = 0; i < M * K; i++) A[i] = rand() % 5;
    for (int i = 0; i < K * N; i++) B[i] = rand() % 5;
    for (int i = 0; i < M * N; i++) C[i] = 1.0f;

    dim3 gridDim(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE), 1);
    dim3 blockDim(BLOCKSIZE * BLOCKSIZE, 1, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sgemm_smem<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    double flops = 2.0 * M * N * K;
    double gflops = (flops / 1e9) / (ms / 1000.0);
    printf("kernel 3 -- SMEM caching\n");
    printf("time = %.4f ms, %.6f GFLOP/s\n", ms, gflops);
    printf("C[0] = %.1f\n", C[0]);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}
