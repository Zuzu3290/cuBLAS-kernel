#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

typedef unsigned int uint;
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define WARPSIZE 32
#define NUM_THREADS 128
#define BM 64
#define BN 64
#define BK 16
#define WM 32 // these vairbales are fixed sized arrays that the compiler would have to allocate. quicl referencing 
#define WN 32
#define WNITER 1
#define TM 4
#define TN 4

// Kernel 10 -- warptiling. Adds a third tiling level between the block and
// the thread.
constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
constexpr uint WSUBM = WM / WMITER;
constexpr uint WSUBN = WN / WNITER;

__global__ __launch_bounds__(NUM_THREADS) void sgemm_warptiling(
    int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // which warp this thread belongs to, and where that warp sits in the block tile
    const uint warpIdx = threadIdx.x / WARPSIZE;
    const uint warpCol = warpIdx % (BN / WN);
    const uint warpRow = warpIdx / (BN / WN);

    // where this thread sits inside its warp's WSUBM x WSUBN sub-tile
    const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

    __shared__ float As[BM * BK];   // transposed: As[k*BM + m]
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    // move C to this warp's output tile, not just this block's
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

    float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
    float regM[WMITER * TM] = {0.0f};
    float regN[WNITER * TN] = {0.0f};

    for (uint bkIdx = 0; bkIdx < (uint)K; bkIdx += BK) {
        for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            float4 tmp = reinterpret_cast<float4 *>(
                &A[(innerRowA + offset) * K + innerColA * 4])[0];
            As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }
        for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
            reinterpret_cast<float4 *>(
                &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
                reinterpret_cast<float4 *>(
                    &B[(innerRowB + offset) * N + innerColB * 4])[0];
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
                for (uint i = 0; i < TM; ++i)
                    regM[wSubRowIdx * TM + i] =
                        As[dotIdx * BM + warpRow * WM + wSubRowIdx * WSUBM +
                           threadRowInWarp * TM + i];

            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
                for (uint i = 0; i < TN; ++i)
                    regN[wSubColIdx * TN + i] =
                        Bs[dotIdx * BN + warpCol * WN + wSubColIdx * WSUBN +
                           threadColInWarp * TN + i];

            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
                for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
                    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM)
                        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN)
                            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                          wSubColIdx * TN + resIdxN] +=
                                regM[wSubRowIdx * TM + resIdxM] *
                                regN[wSubColIdx * TN + resIdxN];
        }
        __syncthreads();
    }

    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            float *C_interim = C + wSubRowIdx * WSUBM * N + wSubColIdx * WSUBN;
            for (uint resIdxM = 0; resIdxM < TM; resIdxM++) {
                for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    float4 tmp = reinterpret_cast<float4 *>(
                        &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                                   threadColInWarp * TN + resIdxN])[0];
                    const uint i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                   wSubColIdx * TN + resIdxN;
                    tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
                    tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
                    tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
                    tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
                    reinterpret_cast<float4 *>(
                        &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                                   threadColInWarp * TN + resIdxN])[0] = tmp;
                }
            }
        }
    }
}

int main() {
    printf("WMITER=%u WSUBM=%u WSUBN=%u\n", WMITER, WSUBM, WSUBN);

    int M = BM, N = BN, K = BK;   // one block tile, one k-iteration
    float alpha = 2.0f, beta = 1.0f;

    float *A, *B, *C;
    cudaMallocManaged(&A, M * K * sizeof(float));
    cudaMallocManaged(&B, K * N * sizeof(float));
    cudaMallocManaged(&C, M * N * sizeof(float));

    for (int i = 0; i < M * K; i++) A[i] = rand() % 5;
    for (int i = 0; i < K * N; i++) B[i] = rand() % 5;
    for (int i = 0; i < M * N; i++) C[i] = 1.0f;

    dim3 blockDim(NUM_THREADS);
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sgemm_warptiling<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    double flops = 2.0 * M * N * K;
    double gflops = (flops / 1e9) / (ms / 1000.0);
    printf("kernel 10 -- warptiling\n");
    printf("time = %.4f ms, %.6f GFLOP/s\n", ms, gflops);
    printf("C[0] = %.1f\n", C[0]);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}
