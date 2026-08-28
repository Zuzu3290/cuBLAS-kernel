#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

typedef unsigned int uint;
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

// Kernel 5 -- 2D block tiling. Each thread now computes a whole TM x TN
// square of results instead of a single column, reusing loaded values even
// more than kernel 4 did.
__global__ void sgemm_2d(int M, int N, int K, float alpha, const float *A,
                          const float *B, float beta, float *C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint numThreads = (BM * BN) / (TM * TN);

    const uint threadCol = threadIdx.x % (BN / TN);
    const uint threadRow = threadIdx.x / (BN / TN);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const uint innerRowA = threadIdx.x / BK;
    const uint innerColA = threadIdx.x % BK;
    const uint strideA = numThreads / BK; // rows of As filled per pass (256/8 = 32); each thread loads BM/strideA = 4 elements total, one per pass
    const uint innerRowB = threadIdx.x / BN;
    const uint innerColB = threadIdx.x % BN;
    const uint strideB = numThreads / BN; // rows of Bs filled per pass (256/128 = 2); each thread loads BK/strideB = 4 elements total, one per pass

    float threadResults[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for (uint bkIdx = 0; bkIdx < (uint)K; bkIdx += BK) {
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA)
            As[(innerRowA + loadOffset) * BK + innerColA] =
                A[(innerRowA + loadOffset) * K + innerColA];
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB)
            Bs[(innerRowB + loadOffset) * BN + innerColB] =
                B[(innerRowB + loadOffset) * N + innerColB];
        __syncthreads();

        A += BK;
        B += BK * N;

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (uint i = 0; i < TM; ++i)
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            for (uint i = 0; i < TN; ++i)
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM)
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN)
                    threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
        }
        __syncthreads();
    }

    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM)
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN)
            C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
                alpha * threadResults[resIdxM * TN + resIdxN] +
                beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
}

int main() {
    int M = BM, N = BN, K = BK;
    float alpha = 2.0f, beta = 1.0f;

    float *A, *B, *C;
    cudaMallocManaged(&A, M * K * sizeof(float));
    cudaMallocManaged(&B, K * N * sizeof(float));
    cudaMallocManaged(&C, M * N * sizeof(float));

    for (int i = 0; i < M * K; i++) A[i] = rand() % 5;
    for (int i = 0; i < K * N; i++) B[i] = rand() % 5;
    for (int i = 0; i < M * N; i++) C[i] = 1.0f;

    dim3 blockDim((BM * BN) / (TM * TN));   // 256
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sgemm_2d<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    double flops = 2.0 * M * N * K;
    double gflops = (flops / 1e9) / (ms / 1000.0);
    printf("kernel 5 -- 2D block tiling (TM=%d, TN=%d)\n", TM, TN);
    printf("time = %.4f ms, %.6f GFLOP/s\n", ms, gflops);
    printf("C[0] = %.1f\n", C[0]);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}
