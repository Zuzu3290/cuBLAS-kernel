import os
import sys
import torch
import cupy as cp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernel_launch


def custom_formula(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, alpha: float) -> torch.Tensor:
    # our kernel launcher needs CuPy arrays, not torch tensors. from_dlpack
    # relabels the same GPU memory as CuPy with no copy; .contiguous() just
    # makes sure that memory is laid out the simple way the kernel expects.
    a_cp = cp.from_dlpack(a.contiguous())
    b_cp = cp.from_dlpack(b.contiguous())
    c_cp = cp.from_dlpack(c.contiguous())
    d_cp = kernel_launch.run(a_cp, b_cp, c_cp, alpha)
    cp.cuda.get_current_stream().synchronize()   # kernel must finish before torch touches the buffer
    return torch.from_dlpack(d_cp)


if __name__ == "__main__":
    torch.manual_seed(0)
    M, K, N = 256, 128, 64
    alpha = 2.0

    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(M, K, device="cuda", dtype=torch.float32)
    C = torch.randn(K, N, device="cuda", dtype=torch.float32)

    D = custom_formula(A, B, C, alpha)
    D_ref = (alpha * A - B) @ C
    fwd_err = (D - D_ref).norm().item() / (D_ref.norm().item() + 1e-30)
    print(f"forward relative error vs torch reference: {fwd_err:.2e}")

    # a random "label" the same shape as D, so loss means something.
    # mean squared error between what the forward pass produced 
    target = torch.randn(M, N, device="cuda", dtype=torch.float32)
    loss = torch.mean((D - target) ** 2)
    print(f"loss = mean((D - target)^2) = {loss.item():.4f}")
