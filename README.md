# Reconstructing the CUDA SGEMM Ladder

*Hand-building Simon Boehm's matmul optimization sequence on hardware his article never saw, then testing whether the same skills carry over to a formula of our own.*

Hardware: NVIDIA GeForce RTX 5060 Laptop GPU (compute capability 12.0, Blackwell)
---

## Page 1 — Topic

### Objective

Reproduce, kernel by kernel, the optimization sequence from Simon Boehm's *"How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance"* — naive → global-memory coalescing → shared-memory tiling → 1D and 2D register blocking → vectorized `float4` loads → autotuning → warp-level tiling — verifying each kernel's correctness independently before benchmarking the full ladder against cuBLAS on an RTX 5060 Laptop GPU, a Blackwell-generation.

A second, smaller objective runs alongside it: exercise the same three skills: write a CUDA kernel, launch it from Python, wire it into autograd, on a formula that isn't standard SGEMM. 

### Motivation

My motivation isn't just to work through this as a study exercise to complete it. What I actually want is to properly encapsulate the fundamentals of the core operative mechanisms that make deep learning applications fast on a GPU, going through them step by step using Simon Boehm's article as the guide and tying each one of those steps back to the fundamental mathematical and hardware mechanisms that cuBLAS itself is built on, the ones that let a GPU extensively exploit matrix computation and launch that work at scale to actually do the computation deep learning depends on.


## Page 2 — What Each File Does

### `build_matmul/` — teaching files

Nine standalone programs, one per idea. Each compiles and runs on its own, at whatever tiny matrix size makes that kernel's tiling correct. 

| File | What it does |
|---|---|
| `Naive_1.cu` | **Kernel 1.** One thread per output element; showing what a first CUDA program actually looks like. |
| `memory_2.cu` | **Kernel 2.** Same math as kernel 1, but remaps threads so a warp's 32 threads read consecutive addresses;  global memory coalescing. |
| `SMEM.cu` | **Kernel 3.** Stages a tile of A and B into on-chip shared memory before computing. |
| `1D_vector.cu` | **Kernel 4.** Each thread computes a column of 8 results cached in registers, reusing one loaded B value across all eight. |
| `2D_vector.cu` | **Kernel 5.** Each thread's registers now hold TM×TN = 64 values (up from TM = 8 in kernel 4), computed as an outer product of two small register arrays. |
| `Vectorized.cu` | **Kernel 6.** Same compute and the same 64-value register footprint as kernel 5, but memory moves in 128-bit `float4` chunks through a transposed shared-memory layout instead. |
| `autotuning.cu` | **Kernel 9.** Templated tile sizes; sweeps four (BM,BN,BK,TM,TN) configurations of the vectorized kernel at a real 1024³ problem and reports the fastest. |
| `warptiling.cu` | **Kernel 10.** Inserts a warp-sized tiling level between the block and thread levels; hand-verified index math, one correctness-sized test case. |
| `FLOP_check.cu` | Not part of the artcile — a simple check to see how GFLOP/s is actually measured at runtime, timed the same way as every kernel above. |

### `build_matmul/bench/` — the harness

The teaching files above answer *"does this idea work?"* one at a time. They can't answer *"which kernel is actually fastest, on the same matrix, against the same baseline?"* — different sizes, no shared reference, no warmup. `bench/` exists solely to answer that second question, the one the article's whole results table is built on.

Getting there required three changes the standalone files deliberately skip: tile sizes as C++ **template** parameters rather than `#define`, so kernels 2 through 10 can share one binary without their macros colliding; a single **cuBLAS** call per matrix size as both the correctness check and the speed baseline; and **warmup + best-of-20** timing so the numbers reflect steady-state throughput, not one-time JIT cost. Claude guided. 

| File | What it does |
|---|---|
| `utils.cuh` | The error-handling this harness uses: `CUDA_CHECK` / `CUBLAS_CHECK` macros that abort with file+line on any failed call. |
| `01_naive.cuh` … `06_vectorized.cuh` | Kernels 1–6, converted from the standalone files into template functions so their tile sizes can vary per instantiation. |
| `10_warptiling.cuh` | Kernel 10 as a template, with `WMITER`/`WSUBM`/`WSUBN` derived at compile time from the other tile parameters. |
| `main.cu` | The harness itself: allocates once per size, runs cuBLAS as reference, dispatches the requested kernel, and prints the `% cuBLAS` table. |

---