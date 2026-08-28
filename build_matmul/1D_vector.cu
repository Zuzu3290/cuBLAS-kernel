#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

typedef unsigned int uint;
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define BM 64
#define BN 64
#define BK 8
#define TM 8

// Kernel 4 1D block tiling. Each thread now computes a whole column of TM
// results instead of one, so a value loaded from B is cached in a register
// and reused across TM multiply-adds. That cuts shared-memory traffic per
// result and raises arithmetic intensity, the real lever from here.
__global__ void sgemm_1d(int M, int N, int K, float alpha, const float *A,
                          const float *B, float beta, float *C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint threadCol = threadIdx.x % BN;
    const uint threadRow = threadIdx.x / BN;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const uint innerColA = threadIdx.x % BK;
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColB = threadIdx.x % BN;
    const uint innerRowB = threadIdx.x / BN;

    float threadResults[TM] = {0.0f};

    for (uint bkIdx = 0; bkIdx < (uint)K; bkIdx += BK) {
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
        __syncthreads();

        A += BK;
        B += BK * N;

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            float Btmp = Bs[dotIdx * BN + threadCol];
            for (uint resIdx = 0; resIdx < TM; ++resIdx)
                threadResults[resIdx] += As[(threadRow * TM + resIdx) * BK + dotIdx] * Btmp;
        }
        __syncthreads();
    }

    for (uint resIdx = 0; resIdx < TM; ++resIdx)
        C[(threadRow * TM + resIdx) * N + threadCol] =
            alpha * threadResults[resIdx] + beta * C[(threadRow * TM + resIdx) * N + threadCol];
}

int main() {
    // one BM x BN block, one BK-wide k-tile: smallest size this kernel's
    // (boundary-check-free) indexing supports.
    int M = BM, N = BN, K = BK;
    float alpha = 2.0f, beta = 1.0f;

    float *A, *B, *C;
    cudaMallocManaged(&A, M * K * sizeof(float));
    cudaMallocManaged(&B, K * N * sizeof(float));
    cudaMallocManaged(&C, M * N * sizeof(float));

    for (int i = 0; i < M * K; i++) A[i] = rand() % 5;
    for (int i = 0; i < K * N; i++) B[i] = rand() % 5;
    for (int i = 0; i < M * N; i++) C[i] = 1.0f;

    dim3 blockDim((BM * BN) / TM);   // 512
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sgemm_1d<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    double flops = 2.0 * M * N * K;
    double gflops = (flops / 1e9) / (ms / 1000.0);
    printf("kernel 4 1D block tiling (TM=%d)\n", TM);
    printf("time = %.4f ms, %.6f GFLOP/s\n", ms, gflops);
    printf("C[0] = %.1f\n", C[0]);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}

// a conclusion was presneted that we acheived more action by utilllizing more threads hence higher arithemtic computation efficiency. 
// keeping in mind the memeory bound and the reduced access to the GMEM global memory. 
