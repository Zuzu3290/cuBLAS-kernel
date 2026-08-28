"""
explore.py: playing with a different formula than the SGEMM 

    D = (alpha * A - B) @ C

instead of the standard  C = alpha*(A@B) + beta*C.
"""
import numpy as np
import cupy as cp
import torch

def custom_formula(A, B, C, alpha=1.0):
    combined = alpha * A - B
    return combined @ C


if __name__ == "__main__":

    rng = np.random.default_rng(0)
    M, K, N = 4, 5, 3
    alpha = 2.0

    A_np = rng.standard_normal((M, K)).astype(np.float32)
    B_np = rng.standard_normal((M, K)).astype(np.float32)
    C_np = rng.standard_normal((K, N)).astype(np.float32)

    # matching alpha*(A@B2)+beta*C2's actual shape requirements.
    beta = 0.5
    B2_np = rng.standard_normal((K, N)).astype(np.float32)
    C2_np = rng.standard_normal((M, N)).astype(np.float32)

    A_cp, B_cp, C_cp = cp.asarray(A_np), cp.asarray(B_np), cp.asarray(C_np)
    B2_cp, C2_cp = cp.asarray(B2_np), cp.asarray(C2_np)

    normal_cp = alpha * (A_cp @ B2_cp) + beta * C2_cp
    print("normal alpha*(A@B2)+beta*C2 result:\n", cp.asnumpy(normal_cp))
    print()

    D_cp = custom_formula(A_cp, B_cp, C_cp, alpha)

    A_t = torch.from_numpy(A_np).cuda()
    B_t = torch.from_numpy(B_np).cuda()
    C_t = torch.from_numpy(C_np).cuda()
    D_t = custom_formula(A_t, B_t, C_t, alpha)

    print("CuPy result:\n", cp.asnumpy(D_cp))
    print("\nPyTorch result:\n", D_t.cpu().numpy())
    print("\nmax abs diff between the two:",
          np.abs(cp.asnumpy(D_cp) - D_t.cpu().numpy()).max())

    # PyTorch has a "fast but slightly less precise" mode for float32
    # matmul called TF32, which it can turn on automatically on newer
    # NVIDIA GPUs. CuPy does not have this mode it always does the
    # precise version. So even though both sides are "float32" and get
    # the exact same input numbers, PyTorch can return a slightly
    # different answer than CuPy, depending on this one setting.
    #
    # The difference is too tiny to notice on small matrices, so this
    # part uses much bigger ones (K=4096).
    print("\n--- TF32 vs strict FP32 ---")
    Mb, Kb, Nb = 512, 4096, 512
    A_big = torch.randn(Mb, Kb, device="cuda", dtype=torch.float32)
    B_big = torch.randn(Kb, Nb, device="cuda", dtype=torch.float32)
    A_big_cp = cp.from_dlpack(A_big)
    B_big_cp = cp.from_dlpack(B_big)

    strict_cp = A_big_cp @ B_big_cp   # CuPy: always strict fp32

    torch.backends.cuda.matmul.allow_tf32 = False
    strict_t = A_big @ B_big
    torch.backends.cuda.matmul.allow_tf32 = True
    tf32_t = A_big @ B_big

    strict_diff = (torch.from_dlpack(strict_cp) - strict_t).abs().max().item()
    tf32_diff = (torch.from_dlpack(strict_cp) - tf32_t).abs().max().item()
    print(f"CuPy vs torch (TF32 off): max abs diff = {strict_diff:.6f}")
    print(f"CuPy vs torch (TF32 on):  max abs diff = {tf32_diff:.6f}")

# performance results :--- TF32 vs strict FP32 ---
# CuPy vs torch (TF32 off): max abs diff = 0.000000
# CuPy vs torch (TF32 on):  max abs diff = 0.090332

