#pragma once
// utils.cuh -- toolchain plumbing you will reuse in every CUDA project.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(err__));                                 \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Use after a kernel launch: launches are async and errors surface later, so we
// force a sync and check. In hot benchmark loops you skip this; in correctness
// paths you keep it.
#define CUDA_CHECK_LAST()                                                       \
    do {                                                                        \
        CUDA_CHECK(cudaGetLastError());                                         \
        CUDA_CHECK(cudaDeviceSynchronize());                                    \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t st__ = (call);                                           \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuBLAS error %s:%d: status %d\n", __FILE__,        \
                    __LINE__, (int)st__);                                       \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)
