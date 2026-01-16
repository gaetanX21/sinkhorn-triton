from typing import Callable

import torch
import triton
import triton.language as tl

# "high" uses TF32 (faster, slight precision loss, uses Tensor Cores on Ampere+ GPUs)
# "highest" uses standard FP32 (slower, full precision, uses CUDA cores) -> default
torch.set_float32_matmul_precision("high")  # we want to leverage Tensor Cores!


def sinkhorn_pytorch(
    log_A: torch.Tensor,  # logits
    n_iter: int,  # increase for better convergence
    epsilon: float,  # numerical stability
) -> torch.Tensor:
    """PyTorch baseline for comparison with Triton kernels."""
    A = torch.exp(log_A)  # ensure positive values
    A_T = A.transpose(-1, -2)  # free transpose (view trick)
    B, N, _ = A.shape

    # initialize scalers
    r = torch.ones(B, N, 1, device=A.device)
    c = torch.ones(B, N, 1, device=A.device)

    # loop
    for _ in range(n_iter):
        r = 1.0 / (A @ c + epsilon)  # row normalization
        c = 1.0 / (A_T @ r + epsilon)  # column normalization

    # final scaled matrix
    return r * A * c.transpose(-1, -2)


# auto-fused & forward-only kernel
sinkhorn_pytorch_compiled = torch.inference_mode()(torch.compile(sinkhorn_pytorch))


@triton.jit
def sinkhorn_A_in_global_memory_kernel(
    A_ptr,
    Out_ptr,
    stride_b,
    stride_h,
    stride_w,
    N: tl.constexpr,  # loop unrolling (okay for N<=16, otherwise compile-time bloat)
    n_iter: tl.constexpr,  # loop unrolling
    N_next: tl.constexpr,
    epsilon: tl.constexpr,
):
    """First attempt: keep A in global memory but r, c now live in registers and we fuse the loop."""
    # 1D grid (B,) -> each block handles one full matrix A of size (N,N)
    pid = tl.program_id(0)

    # initialize scalers in registers
    r = tl.full((N_next,), 1.0, dtype=tl.float32)
    c = tl.full((N_next,), 1.0, dtype=tl.float32)

    # base pointer for matrix A[pid]
    A_base = A_ptr + pid * stride_b
    offsets = tl.arange(0, N_next)
    mask = offsets < N  # safety (for last block if N not multiple of N_next)

    # loop is now inside kernel -> fused memory access for scalers
    for _ in range(n_iter):
        # 1. update r
        # we need to iterate over ROWS of A to set r[i] = 1 / sum(A[i,:]*c)
        for i in range(N):
            row_start_ptr = A_base + i * stride_h
            row_ptrs = row_start_ptr + offsets * stride_w
            row_vals = tl.load(row_ptrs, mask=mask, other=0.0)
            row_sum = tl.sum(row_vals * c)
            new_r_scalar = 1.0 / (row_sum + epsilon)
            r = tl.where(offsets == i, new_r_scalar, r)

        # 2. update c
        # we need to iterate over COLS of A to set c[j] = 1 / sum(A[:,j]*r)
        # BUT, to avoid slow strided column reads, we read ROWS again and accumulate into a temporary buffer
        c_new_accum = tl.zeros((N_next,), dtype=tl.float32)
        for i in range(N):
            row_start_ptr = A_base + i * stride_h
            row_ptrs = row_start_ptr + offsets * stride_w
            row_vals = tl.load(row_ptrs, mask=mask, other=0.0)
            # get r[i] (scalar) from our register vector
            r_val = tl.sum(tl.where(offsets == i, r, 0.0))
            # accumulate: c_new_accum += A[i,:]*r[i]
            c_new_accum += row_vals * r_val
        # finalize c update
        c = 1.0 / (c_new_accum + epsilon)

    # final write -> write result back to global memory
    Out_base = Out_ptr + pid * stride_b
    for i in range(N):
        # re-load A
        row_ptr_start = A_base + i * stride_h
        row_ptrs = row_ptr_start + offsets * stride_w
        row_vals = tl.load(row_ptrs, mask=mask, other=0.0)
        # get r[i]
        r_val = tl.sum(tl.where(offsets == i, r, 0.0))
        # apply scaling
        out_val = row_vals * r_val * c
        # store
        out_row_start_ptr = Out_base + i * stride_h
        out_row_ptrs = out_row_start_ptr + offsets * stride_w
        tl.store(out_row_ptrs, out_val, mask=mask)


def sinkhorn_A_in_global_memory(
    log_A: torch.Tensor,
    n_iter: int,
    epsilon: float,
) -> torch.Tensor:
    A = torch.exp(log_A).contiguous()  # forces memory layout to be Row Major
    Out = torch.empty_like(A)  # A contiguous -> Out guaranteed to have same stride as A
    B, N, _ = A.shape
    N_next = triton.next_power_of_2(N)  # heuristic
    grid = (B,)

    # launch kernel
    sinkhorn_A_in_global_memory_kernel[grid](
        A,
        Out,
        stride_b=A.stride(0),
        stride_h=A.stride(1),
        stride_w=A.stride(2),
        N=N,  # type: ignore
        n_iter=n_iter,  # type: ignore
        N_next=N_next,  # type: ignore
        epsilon=epsilon,  # type: ignore
    )

    return Out


@triton.jit
def sinkhorn_A_in_registers_kernel(
    log_A_ptr,
    Out_ptr,
    stride_b,
    stride_h,
    stride_w,
    N: tl.constexpr,
    n_iter: tl.constexpr,
    N_next: tl.constexpr,
    epsilon: tl.constexpr,
):
    """Second attempt: load full matrix A in registers to minimize global memory access.
    Note: this only works for small N (e.g. N<=128) because of register pressure
    i.e. we don't want registers spilling over into "local memory" (VRAM)."""
    # 1D grid (B,) -> each block handles one full matrix A of size (N,N)
    pid = tl.program_id(0)

    log_A_base = log_A_ptr + pid * stride_b
    offsets = tl.arange(0, N_next)
    log_A_ptrs = log_A_base + offsets[:, None] * stride_h + offsets[None, :] * stride_w
    mask = (offsets[:, None] < N) & (offsets[None, :] < N)
    # load full matrix log_A[pid] of size (N,N) in registers
    log_A_vals = tl.load(log_A_ptrs, mask=mask, other=-float("inf"))
    A_vals = tl.exp(log_A_vals)  # on-the-fly exponentiation

    r = tl.full((N_next,), 1.0, dtype=tl.float32)
    c = tl.full((N_next,), 1.0, dtype=tl.float32)

    # loop -> everything in registers now, no more loads!
    for _ in range(n_iter):
        denom_r = tl.sum(A_vals * c[None, :], axis=1)  # sum across cols
        r = 1.0 / (denom_r + epsilon)

        denom_c = tl.sum(A_vals * r[:, None], axis=0)  # sum across rows
        c = 1.0 / (denom_c + epsilon)

    out = A_vals * r[:, None] * c[None, :]
    Out_base = Out_ptr + pid * stride_b
    # NOTE: works because A.strides() == Out.strides() (because A contiguous and Out=torch.empty_like(A))
    out_ptrs = Out_base + offsets[:, None] * stride_h + offsets[None, :] * stride_w
    tl.store(out_ptrs, out, mask=mask)


def sinkhorn_A_in_registers(
    log_A: torch.Tensor,
    n_iter: int,
    epsilon: float,
) -> torch.Tensor:
    log_A = log_A.contiguous()
    Out = torch.empty_like(log_A)
    B, N, _ = log_A.shape
    N_next = triton.next_power_of_2(N)
    grid = (B,)

    sinkhorn_A_in_registers_kernel[grid](
        log_A,
        Out,
        stride_b=log_A.stride(0),
        stride_h=log_A.stride(1),
        stride_w=log_A.stride(2),
        N=N,  # type: ignore
        n_iter=n_iter,  # type: ignore
        N_next=N_next,  # type: ignore
        epsilon=epsilon,  # type: ignore
    )

    return Out


@triton.jit
def sinkhorn_A_in_registers_block_packing_kernel(
    log_A_ptr,
    Out_ptr,
    stride_b,
    stride_h,
    stride_w,
    B: int,
    N: tl.constexpr,
    N_next: tl.constexpr,
    matrix_per_block: tl.constexpr,
    n_iter: tl.constexpr,
    epsilon: tl.constexpr,
):
    """Third attempt: load several matrices A in registers to better utilize GPU threads.
    Each block processes matrix_per_block matrices of size (N,N)."""
    # 1D grid (num_blocks,) -> each block handles matrix_per_block matrices of size (N,N)
    pid = tl.program_id(0)

    offs_b = tl.arange(0, matrix_per_block)
    offs_h = tl.arange(0, N_next)
    offs_w = tl.arange(0, N_next)

    # ensure we don't load/store matrices beyond batch size B
    batch_idxs = pid * matrix_per_block + offs_b
    mask_b = (batch_idxs < B)[:, None, None]
    # ensure we don't load/store rows/cols beyond matrix size N
    mask_h = (offs_h < N)[None, :, None]
    mask_w = (offs_w < N)[None, None, :]
    # final mask
    mask_3d = mask_b & mask_h & mask_w

    # skip matrix_per_block matrices for every previous block
    log_A_base = log_A_ptr + pid * matrix_per_block * stride_b
    # 3D pointers: shape (matrix_per_block, N, N)
    # we use broadcasting to expand dimensions
    offs_3d = (
        offs_b[:, None, None] * stride_b
        + offs_h[None, :, None] * stride_h
        + offs_w[None, None, :] * stride_w
    )
    log_A_ptrs = log_A_base + offs_3d
    # load batch of matrices in registers
    log_A_vals = tl.load(log_A_ptrs, mask=mask_3d, other=-float("inf"))
    A_vals = tl.exp(log_A_vals)

    r = tl.full((matrix_per_block, N_next, 1), 1.0, dtype=tl.float32)
    c = tl.full((matrix_per_block, 1, N_next), 1.0, dtype=tl.float32)

    # Triton takes care of 3D Hadamard product
    for _ in range(n_iter):
        denom_r = tl.sum(A_vals * c, axis=2, keep_dims=True)
        r = 1.0 / (denom_r + epsilon)

        denom_c = tl.sum(A_vals * r, axis=1, keep_dims=True)
        c = 1.0 / (denom_c + epsilon)

    out = A_vals * r * c
    Out_base = Out_ptr + pid * matrix_per_block * stride_b
    out_ptrs = Out_base + offs_3d
    tl.store(out_ptrs, out, mask=mask_3d)


def sinkhorn_A_in_registers_block_packing(
    log_A: torch.Tensor,
    n_iter: int,
    epsilon: float,
    # NOTE: testing on B=2^20, found that block_size=256, num_warps=4 was the best config
    block_size: int = 256,  # proxy for data size per block
    num_warps: int = 4,  # number of warps (32 threads) per block
) -> torch.Tensor:
    log_A = log_A.contiguous()
    Out = torch.empty_like(log_A)
    B, N, _ = log_A.shape
    N_next = triton.next_power_of_2(N)

    # block packing -> blocks now contain batch of matrices instead of single matrix
    wanted_matrix_per_block = min(B, block_size // N)  # heuristic
    actual_matrix_per_block = triton.next_power_of_2(wanted_matrix_per_block)
    num_blocks = triton.cdiv(B, actual_matrix_per_block)  # ceil so we don't drop data
    grid = (num_blocks,)

    sinkhorn_A_in_registers_block_packing_kernel[grid](
        log_A_ptr=log_A,
        Out_ptr=Out,
        stride_b=log_A.stride(0),
        stride_h=log_A.stride(1),
        stride_w=log_A.stride(2),
        B=B,
        N=N,  # type: ignore
        N_next=N_next,  # type: ignore
        matrix_per_block=actual_matrix_per_block,  # type: ignore
        n_iter=n_iter,  # type: ignore
        epsilon=epsilon,  # type: ignore
        num_warps=num_warps,  # type: ignore
    )

    return Out


def verify_correctness(
    func: Callable,
    B: int,
    N: int,
    n_iter: int,
    epsilon: float,
    atol_max: float,
    atol_mean: float,
) -> None:
    log_A = torch.randn(B, N, N, device="cuda")
    out = func(log_A, n_iter=n_iter, epsilon=epsilon)
    max_dist_rows = (out.sum(dim=-1) - 1).abs().max().item()
    max_dist_cols = (out.sum(dim=-2) - 1).abs().max().item()
    if not (max_dist_rows < atol_max and max_dist_cols < atol_max):
        print(f"ACHTUNG: {func.__name__}: {max_dist_cols=:.2g}, {max_dist_rows=:.2g}")
    mean_dist_rows = (out.sum(dim=-1) - 1).abs().mean().item()
    mean_dist_cols = (out.sum(dim=-2) - 1).abs().mean().item()
    if not (mean_dist_rows < atol_mean and mean_dist_cols < atol_mean):
        print(f"ACHTUNG: {func.__name__}: {mean_dist_cols=:.2g}, {mean_dist_rows=:.2g}")


if __name__ == "__main__":
    # sweep, includes non-power of 2 to stress-test masking
    B_list = [1, 2, 3, 4, 5, 255, 423, 1025]
    N_list = [1, 2, 3, 4, 5, 12, 15, 16]

    n_iter = 20  # increase for better convergence
    epsilon = 1e-6  # numerical stability
    atol_max = 1e-4  # ACHTUNG: increase for large B
    atol_mean = 1e-5
    funcs = [
        sinkhorn_pytorch,
        sinkhorn_pytorch_compiled,
        sinkhorn_A_in_global_memory,
        sinkhorn_A_in_registers,
        sinkhorn_A_in_registers_block_packing,
    ]
    for B in B_list:
        for N in N_list:
            print(f"B={B}, N={N}")
            for func in funcs:
                verify_correctness(func, B, N, n_iter, epsilon, atol_max, atol_mean)
