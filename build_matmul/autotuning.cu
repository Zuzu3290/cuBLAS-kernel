#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

typedef unsigned int uint;
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

// Kernel 9 -- autotuning. 
// BM, BN and BK, which specify how much data we cache from GMEM into SMEM.
// TM and TN, which specify how much data we cache from SMEM into the registers.
template <uint BM, uint BN, uint BK, uint TM, uint TN>
__global__ void sgemm_vectorized(int M, int N, int K, float alpha, float *A,
                                  float *B, float beta, float *C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint threadCol = threadIdx.x % (BN / TN);
    const uint threadRow = threadIdx.x / (BN / TN);

    __shared__ float As[BM * BK];
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

// BM,BN are forced equal (= BK*TM*TN/4) by this kernel's own load-index math
// -- each thread loads one float4, and the loop coverage only works out if
// BM*BK/4 == BN*BK/4 == blockDim.x. So a "candidate" here is really just a
// (BK,TM,TN) triple; BM/BN fall out of it.
template <uint BK, uint TM, uint TN>
float run_config(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    constexpr uint BM = BK * TM * TN / 4;
    constexpr uint BN = BM;
    dim3 blockDim((BM * BN) / (TM * TN));
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warmup (first launch pays JIT-compile cost on this GPU/toolkit combo)
    sgemm_vectorized<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    cudaDeviceSynchronize();

    float best = 1e30f;
    for (int rep = 0; rep < 5; rep++) {
        cudaEventRecord(start);
        sgemm_vectorized<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        if (ms < best) best = ms;
    }

    double flops = 2.0 * M * N * K;
    double gflops = (flops / 1e9) / (best / 1000.0);
    printf("  BM=BN=%3u BK=%2u TM=%u TN=%u  ->  %8.3f ms  %10.1f GFLOP/s\n",
           BM, BK, TM, TN, best, gflops);
    return gflops;
}

int main() {
    // large enough that tile-size choice actually matters, and divisible by
    // every candidate BM/BN below (32, 64, 128).
    int M = 1024, N = 1024, K = 1024;
    float alpha = 1.0f, beta = 0.0f;

    float *A, *B, *C;
    cudaMallocManaged(&A, (size_t)M * K * sizeof(float));
    cudaMallocManaged(&B, (size_t)K * N * sizeof(float));
    cudaMallocManaged(&C, (size_t)M * N * sizeof(float));

    for (int i = 0; i < M * K; i++) A[i] = rand() % 5;
    for (int i = 0; i < K * N; i++) B[i] = rand() % 5;
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;

    printf("kernel 9 -- autotuning: sweeping (BK,TM,TN); BM=BN=BK*TM*TN/4\n");
    printf("M=N=K=%d\n\n", M);

    double best_gflops = 0.0;
    const char *best_name = "";

    double g;
    g = run_config<8, 4, 4>(M, N, K, alpha, A, B, beta, C);
    if (g > best_gflops) { best_gflops = g; best_name = "BK=8 TM=4 TN=4 (BM=BN=32)"; }

    g = run_config<8, 8, 4>(M, N, K, alpha, A, B, beta, C);
    if (g > best_gflops) { best_gflops = g; best_name = "BK=8 TM=8 TN=4 (BM=BN=64)"; }

    g = run_config<8, 4, 8>(M, N, K, alpha, A, B, beta, C);
    if (g > best_gflops) { best_gflops = g; best_name = "BK=8 TM=4 TN=8 (BM=BN=64)"; }

    g = run_config<8, 8, 8>(M, N, K, alpha, A, B, beta, C);
    if (g > best_gflops) { best_gflops = g; best_name = "BK=8 TM=8 TN=8 (BM=BN=128)"; }

    printf("\nbest: %s  ->  %.1f GFLOP/s\n", best_name, best_gflops);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}
