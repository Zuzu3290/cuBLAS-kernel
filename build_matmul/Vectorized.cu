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

// Kernel 6 -- vectorized loads. Same logic as kernel 5,
// but loads from global memory are now vectorized (float4) instead of scalar (float).
// That cuts the number of load instructions to a quarter, since one float4 load moves
// four floats at once instead of one.
__global__ void sgemm_vectorized(int M, int N, int K, float alpha, float *A,
                                  float *B, float beta, float *C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint threadCol = threadIdx.x % (BN / TN);
    const uint threadRow = threadIdx.x / (BN / TN);

    __shared__ float As[BM * BK];   // transposed: As[k*BM + m]
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);

    float threadResults[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for (uint bkIdx = 0; bkIdx < (uint)K; bkIdx += BK) {
        float4 tmp = reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

        reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
            reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
        __syncthreads();

        A += BK;
        B += BK * N;

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (uint i = 0; i < TM; ++i)
                regM[i] = As[dotIdx * BM + threadRow * TM + i];
            for (uint i = 0; i < TN; ++i)
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM)
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN)
                    threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
        }
        __syncthreads();
    }

    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
            float4 tmp = reinterpret_cast<float4 *>(
                &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
            tmp.x = alpha * threadResults[resIdxM * TN + resIdxN + 0] + beta * tmp.x;
            tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
            tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
            tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
            reinterpret_cast<float4 *>(
                &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] = tmp;
        }
    }
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
    sgemm_vectorized<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    double flops = 2.0 * M * N * K;
    double gflops = (flops / 1e9) / (ms / 1000.0);
    printf("kernel 6 -- vectorized (float4)\n");
    printf("time = %.4f ms, %.6f GFLOP/s\n", ms, gflops);
    printf("C[0] = %.1f\n", C[0]);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}
