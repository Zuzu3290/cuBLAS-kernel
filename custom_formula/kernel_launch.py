"""
kernel_launch.py -_ the same custom formula D = (alpha*A - B) @ C, but now as
a real hand-written CUDA kernel launched through cupy.RawKernel . 
Validated against CuPy's own (alpha*A - B) @ C as ground truth 
"""
import cupy as cp

SOURCE = r'''
// extern "C" stops C++ name mangling, so the compiled symbol is literally
// named "custom_formula"
extern "C" __global__
void custom_formula(int M, int N, int K, float alpha,
                     const float* A, const float* B, const float* C, float* D) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;  // row, over M
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;  // col, over N

    if (x < (unsigned)M && y < (unsigned)N) {
        float acc = 0.0f;
        for (int i = 0; i < K; ++i) {
            float combined = alpha * A[x * K + i] - B[x * K + i];
            acc += combined * C[i * N + y];
        }
        D[x * N + y] = acc;
    }
}
'''

kernel = cp.RawKernel(SOURCE, "custom_formula", options=("--std=c++17",))


def run(A, B, C, alpha=1.0):
    M, K = A.shape
    K2, N = C.shape
    assert A.shape == B.shape, "A and B must be the same shape (elementwise combined)"
    assert K == K2, "inner dimension of (A,B) must match C's rows"
    D = cp.zeros((M, N), dtype=cp.float32)
    block = (32, 32, 1)
    grid = ((M + 31) // 32, (N + 31) // 32, 1)
    kernel(
        grid, block,
        (cp.int32(M), cp.int32(N), cp.int32(K), cp.float32(alpha), A, B, C, D),
    )
    return D


if __name__ == "__main__":
    rng = cp.random.default_rng(0)
    M, K, N = 256, 128, 64
    alpha = 2.0

    A = rng.standard_normal((M, K), dtype=cp.float32)
    B = rng.standard_normal((M, K), dtype=cp.float32)
    C = rng.standard_normal((K, N), dtype=cp.float32)

    D = run(A, B, C, alpha)
    D_ref = (alpha * A - B) @ C

    err = float(cp.linalg.norm(D - D_ref) / (cp.linalg.norm(D_ref) + 1e-30))
    print(f"relative error vs CuPy reference: {err:.2e}  ({'OK' if err < 1e-3 else 'MISMATCH'})")
    print("D =", float(D[0, 0]), " ref D =", float(D_ref[0, 0]))
