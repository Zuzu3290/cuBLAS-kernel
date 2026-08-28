#pragma once
// Kernel 6 -- Vectorized SMEM/GMEM.

template <const uint BM, const uint BN, const uint BK, const uint TM,
          const uint TN>
__global__ void sgemm_vectorized(int M, int N, int K, float alpha, float *A,
                                 float *B, float beta, float *C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint threadCol = threadIdx.x % (BN / TN);
    const uint threadRow = threadIdx.x / (BN / TN);

    __shared__ float As[BM * BK];  // stored TRANSPOSED: As[k*BM + m]
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
        // vectorized load of A, transposed into As on the way in
        float4 tmp =
            reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

        // vectorized load of B, normal layout
        reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
            reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
        __syncthreads();

        A += BK;
        B += BK * N;

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (uint i = 0; i < TM; ++i)
                regM[i] = As[dotIdx * BM + threadRow * TM + i];  // contiguous
            for (uint i = 0; i < TN; ++i)
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM)
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN)
                    threadResults[resIdxM * TN + resIdxN] +=
                        regM[resIdxM] * regN[resIdxN];
        }
        __syncthreads();
    }

    // vectorized write-back, 4 columns at a time
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
            float4 tmp = reinterpret_cast<float4 *>(
                &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
            tmp.x = alpha * threadResults[resIdxM * TN + resIdxN + 0] + beta * tmp.x;
            tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
            tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
            tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
            reinterpret_cast<float4 *>(
                &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
                tmp;
        }
    }
}
