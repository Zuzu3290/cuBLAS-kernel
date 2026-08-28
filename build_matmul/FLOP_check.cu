#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// A simple kernel used to check how GFLOP/s gets measured at runtime.
__global__ void max_flops_kernel(float *result, unsigned long long iterations) {
    float a = 1.001f;
    float b = 1.002f;
    float c = 1.003f;
    float d = 1.004f;

    // #pragma unroll 8 tells the compiler to copy-paste 8 copies of this
    // loop's body back to back instead of jumping back to the top 8 times,
    // so more of the GPU's time is spent on
    // the actual math we're trying to measure.
    #pragma unroll 8
    for (unsigned long long i = 0; i < iterations; ++i) {
        // fmaf(x, y, z) = one "fused multiply-add": computes x*y + z in a
        a = fmaf(a, b, c);
        b = fmaf(b, c, d);
        c = fmaf(c, d, a);
        d = fmaf(d, a, b);

        a = fmaf(a, c, d);
        b = fmaf(b, d, a);
        c = fmaf(c, a, b);
        d = fmaf(d, b, c);
    }

    // Prevent the compiler from optimizing the whole loop away.
    if (blockIdx.x == 0 && threadIdx.x == 0)
        result[0] = a + b + c + d;
}

int main() {
    int blocks = 1024;
    int threads = 256;
    unsigned long long iterations = 1000000ULL;

    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s\n", prop.name);

    float *dResult = nullptr;
    cudaMalloc(&dResult, sizeof(float));

    max_flops_kernel<<<blocks, threads>>>(dResult, iterations);   // warmup
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    max_flops_kernel<<<blocks, threads>>>(dResult, iterations);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    double total_flops = (double)blocks * threads * iterations * 16.0;
    double gflops = total_flops / (ms / 1000.0) / 1e9;

    float result = 0.0f;
    cudaMemcpy(&result, dResult, sizeof(float), cudaMemcpyDeviceToHost);

    printf("time = %.3f ms, %.2f GFLOP/s\n", ms, gflops);
    printf("result = %f\n", result);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dResult);
    return 0;
}
