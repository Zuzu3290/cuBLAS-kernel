#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                             const float *B, float beta, float *C) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

int main() {
    // M: number of rows in A and C
    // N: number of columns in B and C
    // K: number of columns in A / rows in B
    // alpha: scalar multiplied into the A*B product, C = alpha*(A@B) + beta*C
    // beta: scalar multiplied into the existing value of C before adding alpha*(A@B)

    int M = 4, N = 4, K = 4;
    float alpha = 2.0f, beta = 1.0f;   // change these freely and re-run

    float *A, *B, *C;
    // A is matrix of size M x K and a pointer allocates memory for it on the GPU
    // B is matrix of size K x N and a pointer allocates memory for it on the GPU
    // C is matrix of size M x N and a pointer allocates memory for it on the GPU

    cudaMallocManaged(&A, M * K * sizeof(float));
    cudaMallocManaged(&B, K * N * sizeof(float));
    cudaMallocManaged(&C, M * N * sizeof(float));

    for (int i = 0; i < M * K; i++) A[i] = rand() % 5;
    for (int i = 0; i < K * N; i++) B[i] = rand() % 5;
    // C being initialized before the kernel in that freshly allocated memory, not a real value.
    for (int i = 0; i < M * N; i++) C[i] = 1.0f;

    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
    dim3 blockDim(32, 32, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    double flops = 2.0 * M * N * K;   // one multiply + one add per inner-loop step
    double gflops = (flops / 1e9) / (ms / 1000.0);
    printf("time = %.4f ms, %.6f GFLOP/s\n", ms, gflops);
    printf("A row 0    = ");
    for (int i = 0; i < K; i++) printf("%.1f ", A[0 * K + i]);   // row 0: contiguous
    printf("\n");

    printf("B column 0 = ");
    for (int i = 0; i < K; i++) printf("%.1f ", B[i * N + 0]);  // column 0: stride N
    printf("\n");

    printf("C[0] = %.1f\n", C[0]);

    cudaFree(A); // we free the storage but still have a dangling pointer.
    cudaFree(B);
    cudaFree(C);
    return 0;
}
