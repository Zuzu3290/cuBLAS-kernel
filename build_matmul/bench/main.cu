// main.cu -- the proper benchmark harness.
//
// Run:    ./bench <kernel 1-6,10> [size ...]
//   e.g.  ./bench 6             # kernel 6 on the default sizes
//         ./bench 3 1024 2048   # kernel 3 on two sizes
//         ./bench 0             # cuBLAS-only baseline

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <string>
#include "utils.cuh"
#include "01_naive.cuh"
#include "02_coalescing.cuh"
#include "03_smem_caching.cuh"
#include "04_1d_blocktiling.cuh"
#include "05_2d_blocktiling.cuh"
#include "06_vectorized.cuh"
#include "10_warptiling.cuh"

void run_kernel(int kernel, int M, int N, int K, float alpha, float *A,
                float *B, float beta, float *C) {
    switch (kernel) {
        case 1: {
            dim3 block(32, 32);
            dim3 grid(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
            sgemm_naive<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }
        case 2: {
            dim3 block(32 * 32);
            dim3 grid(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
            sgemm_coalescing<32><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }
        case 3: {
            dim3 block(32 * 32);
            dim3 grid(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
            sgemm_smem<32><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }
        case 4: {
            constexpr uint BM = 64, BN = 64, BK = 8, TM = 8;
            dim3 block((BM * BN) / TM);            // 512
            dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
            sgemm_1d<BM, BN, BK, TM>
                <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }
        case 5: {
            constexpr uint BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
            dim3 block((BM * BN) / (TM * TN));     // 256
            dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
            sgemm_2d<BM, BN, BK, TM, TN>
                <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }
        case 6: {
            constexpr uint BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
            dim3 block((BM * BN) / (TM * TN));     // 256
            dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
            sgemm_vectorized<BM, BN, BK, TM, TN>
                <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }
        case 10: {

            constexpr uint NUM_THREADS = 128;
            constexpr uint BM = 64, BN = 64, BK = 16;
            constexpr uint WM = 32, WN = 32, WNITER = 1;
            constexpr uint TM = 4, TN = 4;
            dim3 block(NUM_THREADS);
            dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
            sgemm_warptiling<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
                <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }
        default:
            fprintf(stderr, "unknown kernel %d (use 1-6, 10, or 0 for cuBLAS)\n",
                    kernel);
            exit(EXIT_FAILURE);
    }
}

void run_cublas(cublasHandle_t handle, int M, int N, int K, float alpha,
                float *A, float *B, float beta, float *C) {
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                             B, N, A, K, &beta, C, N));
}

double gflops(int M, int N, int K, float ms) {
    return (2.0 * M * N * K) / (ms * 1e-3) / 1e9;
}

int main(int argc, char **argv) {
    const int kernel = (argc > 1) ? atoi(argv[1]) : 1;
    std::vector<int> sizes;
    for (int i = 2; i < argc; ++i) sizes.push_back(atoi(argv[i]));
    if (sizes.empty()) sizes = {1024, 2048, 4096};

    const float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  (compute capability %d.%d)\n\n", prop.name, prop.major,
           prop.minor);

    printf("kernel %d\n", kernel);
    printf("%6s | %4s | %10s | %10s | %10s | %8s\n", "size", "ok", "rel_err",
           "GFLOP/s", "cuBLAS", "% cuBLAS");
    printf("---------------------------------------------------------------\n");

    std::mt19937 rng(0);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int n : sizes) {
        const int M = n, N = n, K = n;
        const size_t bytes = (size_t)M * N * sizeof(float);

        std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
        for (auto &v : hA) v = dist(rng);
        for (auto &v : hB) v = dist(rng);
        std::vector<float> hC(M * N), hRef(M * N);

        float *dA, *dB, *dC, *dRef;
        CUDA_CHECK(cudaMalloc(&dA, (size_t)M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, (size_t)K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, bytes));
        CUDA_CHECK(cudaMalloc(&dRef, bytes));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), (size_t)M * K * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), (size_t)K * N * sizeof(float),
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemset(dRef, 0, bytes));
        run_cublas(handle, M, N, K, alpha, dA, dB, beta, dRef);
        CUDA_CHECK(cudaDeviceSynchronize());

        bool ok = true;
        double relerr = 0.0;
        if (kernel != 0) {
            CUDA_CHECK(cudaMemset(dC, 0, bytes));
            run_kernel(kernel, M, N, K, alpha, dA, dB, beta, dC);
            CUDA_CHECK_LAST();
            CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hRef.data(), dRef, bytes,
                                  cudaMemcpyDeviceToHost));
            double num = 0.0, den = 0.0;
            for (size_t i = 0; i < hC.size(); ++i) {
                double d = (double)hC[i] - (double)hRef[i];
                num += d * d;
                den += (double)hRef[i] * (double)hRef[i];
            }
            relerr = std::sqrt(num) / (std::sqrt(den) + 1e-30);
            ok = relerr < 1e-2;
        }

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        auto time_it = [&](auto launch) {
            for (int i = 0; i < 5; ++i) launch();  // warmup
            CUDA_CHECK(cudaDeviceSynchronize());
            float best = 1e30f;
            for (int i = 0; i < 20; ++i) {
                CUDA_CHECK(cudaEventRecord(start));
                launch();
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));
                float ms;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                if (ms < best) best = ms;
            }
            return best;
        };

        float ms_cublas = time_it(
            [&] { run_cublas(handle, M, N, K, alpha, dA, dB, beta, dRef); });
        float ms_kernel =
            (kernel == 0)
                ? ms_cublas
                : time_it([&] {
                      run_kernel(kernel, M, N, K, alpha, dA, dB, beta, dC);
                  });

        double g = gflops(M, N, K, ms_kernel);
        double gc = gflops(M, N, K, ms_cublas);
        printf("%6d | %4s | %10.2e | %10.1f | %10.1f | %7.1f%%\n", n,
               (kernel == 0 ? "-" : (ok ? "yes" : "NO")), relerr, g, gc,
               100.0 * g / gc);
        if (!ok)
            printf("       ^ rel_err %.2e exceeds 1e-2: kernel likely buggy at "
                   "this size\n", relerr);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef);
    }

    cublasDestroy(handle);
    return 0;
}
