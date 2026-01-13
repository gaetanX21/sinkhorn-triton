from typing import Callable

import torch
import triton
import triton.language as tl


def sinkhorn_pytorch(
    log_A: torch.Tensor,  # logits
    n_iter: int,  # increase for better convergence
    epsilon: float,  # numerical stability
) -> torch.Tensor:
    """PyTorch baseline for comparison with the triton kernels."""
    A = torch.exp(log_A)
    B, N, _ = A.shape

    # initialize scalers
    r = torch.ones(B, N, device=A.device)
    c = torch.ones(B, N, device=A.device)

    # loop
    for _ in range(n_iter):
        r = 1.0 / ((A * c[:, None, :]).sum(dim=-1) + epsilon)  # row normalization
        c = 1.0 / ((A * r[:, :, None]).sum(dim=-2) + epsilon)  # column normalization

    # final scaled matrix
    return A * r[:, :, None] * c[:, None, :]


@triton.jit
def sinkhorn_fused_kernel_A_in_global_memory(
    # fusing the loop over iterations into the kernel to minimize kernel launch overhead and memory access for r and c
    # however, A stays in global memory
    A_ptr,
    Out_ptr,
    stride_b,
    stride_h,
    stride_w,
    N,  # can be made tl.constexpr for small matrices (e.g. N<=16) -> for bigger matrices, not a good idea as it will cause loop unrolling bloat
    n_iter: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    epsilon: tl.constexpr,
):
    # 1D grid (B,) -> each program handles one full matrix A of size (N,N)
    pid = tl.program_id(0)

    # initialize local memory in registers
    r = tl.zeros((BLOCK_SIZE,), dtype=tl.float32) + 1.0
    c = tl.zeros((BLOCK_SIZE,), dtype=tl.float32) + 1.0

    # base pointer for this matrix
    A_base = A_ptr + pid * stride_b
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N  # safety

    # loop is now inside kernel -> FUSED
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
        # to avoid slow strided column reads, we read ROWS again and accumulate into a temporary buffer
        c_new_accum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
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


@triton.jit
def sinkhorn_fused_kernel_A_in_registers(
    # each matrix A[pid] fully loaded in registers to further reduce memory access
    # requires N to be small so we don't run out of registers and spill into "local memory" (VRAM)
    A_ptr,
    Out_ptr,
    stride_b,
    stride_h,
    stride_w,
    N: int,
    n_iter: tl.constexpr,  # set as constant for loop unrolling (okay for small n_iter e.g. 20)
    BLOCK_SIZE: tl.constexpr,
    epsilon: tl.constexpr,
):
    pid = tl.program_id(0)

    A_base = A_ptr + pid * stride_b
    offsets = tl.arange(0, BLOCK_SIZE)
    A_ptrs = A_base + offsets[:, None] * stride_h + offsets[None, :] * stride_w
    mask = (offsets[:, None] < N) & (offsets[None, :] < N)
    # load full matrix A[pid,:,:] of size (N,N)
    A_full = tl.load(A_ptrs, mask=mask, other=0.0)

    # initialize r and c
    r = tl.zeros((BLOCK_SIZE,), dtype=tl.float32) + 1.0
    c = tl.zeros((BLOCK_SIZE,), dtype=tl.float32) + 1.0

    # loop (pure math, no loads!)
    for _ in range(n_iter):
        # 1. r = 1 / (A@c).sum(dim=1)
        # we broadcast c to (1,N) and multiply element-wise, then sum cols
        denom_r = tl.sum(A_full * c[None, :], axis=1)  # sum across cols
        r = 1.0 / (denom_r + epsilon)

        # 2. c = 1 / (A.T@r).sum(dim=0)
        # we broadcast r to (N,1) and mulitply element-wise, then sum rows
        denom_c = tl.sum(A_full * r[:, None], axis=0)  # sum across rows
        c = 1.0 / (denom_c + epsilon)

    out = A_full * r[:, None] * c[None, :]
    Out_base = Out_ptr + pid * stride_b
    # NOTE: works because A.strides() == Out.strides() (because A contiguous and Out=torch.empty_like(A))
    out_ptrs = Out_base + offsets[:, None] * stride_h + offsets[None, :] * stride_w
    tl.store(out_ptrs, out, mask=mask)


@triton.jit
def sinkhorn_fused_kernel_A_in_registers_block_tiling(
    # process several matrices per block to use more threads per block
    A_ptr,
    Out_ptr,
    stride_b,
    stride_h,
    stride_w,
    B: int,
    N: tl.constexpr,  # N fixed e.g. 4 (better compile effiency)
    matrices_per_block: tl.constexpr,
    n_iter: tl.constexpr,  # set as constant for loop unrolling (okay for small n_iter e.g. 20)
    BLOCK_SIZE: tl.constexpr,
    epsilon: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_b = tl.arange(0, matrices_per_block)
    offs_h = tl.arange(0, N)
    offs_w = tl.arange(0, N)

    # 3D pointers: shape (matrices_per_block, N, N)
    # we use broadcasting to expand dimensions
    batch_idxs = pid * matrices_per_block + offs_b
    # ensure we don't load/store matrices beyond batch size B
    mask = (batch_idxs < B)[:, None, None]

    A_base = A_ptr + pid * BLOCK_SIZE
    offs_3d = (
        offs_b[:, None, None] * stride_b
        + offs_h[None, :, None] * stride_h
        + offs_w[None, None, :] * stride_w
    )
    A_ptrs = A_base + offs_3d

    # load chunks of matrices into register
    A_full = tl.load(A_ptrs, mask=mask, other=0.0)

    # initialize r and c
    r = tl.zeros((matrices_per_block, N, 1), dtype=tl.float32) + 1.0
    c = tl.zeros((matrices_per_block, 1, N), dtype=tl.float32) + 1.0

    # loop (pure math, no loads!)
    for _ in range(n_iter):
        # 1. update r
        denom_r = tl.sum(A_full * c, axis=2, keep_dims=True)
        r = 1.0 / (denom_r + epsilon)

        # 2. update c
        denom_c = tl.sum(A_full * r, axis=1, keep_dims=True)
        c = 1.0 / (denom_c + epsilon)

    out = A_full * r * c
    Out_base = Out_ptr + pid * BLOCK_SIZE
    # NOTE: works because A.strides() == Out.strides() (because A contiguous and Out=torch.empty_like(A))
    out_ptrs = Out_base + offs_3d
    tl.store(out_ptrs, out, mask=mask)


@triton.jit
def sinkhorn_fused_coalesced_kernel(
    log_A_ptr,  # logits (B, N, N)
    Out_ptr,  # output: bistochastic matrix (B, N, N)
    total_elements: int,  # for masking (safety)
    N: tl.constexpr,
    n_iter: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    matrices_per_block: tl.constexpr,
    epsilon: tl.constexpr,
):
    pid = tl.program_id(0)

    # treat memory as 1D array of floats
    # block size must be matrices_per_block * N * N

    # load contiguous chunk of data (perfect coalescing)
    block_offset = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    global_idxs = block_offset + offsets
    mask = global_idxs < total_elements  # for masking (safety)

    log_A_ptrs = log_A_ptr + global_idxs
    log_A_vals = tl.load(log_A_ptrs, mask=mask, other=-float("inf"))
    A_vals = tl.exp(log_A_vals)

    # reshape into matrices for sinkhorn math
    # view as (mat_idx, row, cols)
    A_view = tl.reshape(A_vals, (matrices_per_block, N, N))

    # initialize scalers
    r = tl.full((matrices_per_block, N, 1), 1.0, dtype=tl.float32)
    c = tl.full((matrices_per_block, 1, N), 1.0, dtype=tl.float32)

    # loop
    for _ in range(n_iter):
        # r = 1 / (A@c)
        denom_r = tl.sum(A_view * c, axis=2, keep_dims=True)
        r = 1.0 / (denom_r + epsilon)
        # c = 1 / (A@r)
        denom_c = tl.sum(A_view * r, axis=1, keep_dims=True)
        c = 1.0 / (denom_c + epsilon)

    # final write
    out_view = A_view * r * c

    # flatten back to match load structure
    out_vals = tl.reshape(out_view, (BLOCK_SIZE,))

    # write back to global memory
    out_ptrs = Out_ptr + global_idxs
    tl.store(out_ptrs, out_vals, mask=mask)


def sinkhorn_fused_A_in_global_memory(
    log_A: torch.Tensor,
    n_iter: int,
    epsilon: float,
) -> torch.Tensor:
    A = torch.exp(log_A).contiguous()  # forces memory layout of A to be Row Major
    Out = torch.empty_like(A)  # A contiguous -> Out GUARANTEED to have same stride as A
    B, N, _ = A.shape
    grid = (B,)
    BLOCK_SIZE = triton.next_power_of_2(N)

    # launch kernel
    sinkhorn_fused_kernel_A_in_global_memory[grid](
        A,
        Out,
        A.stride(0),
        A.stride(1),
        A.stride(2),
        N,
        n_iter,  # type: ignore
        BLOCK_SIZE,  # type: ignore
        epsilon,  # type: ignore
    )

    return Out


def sinkhorn_fused_A_in_registers(
    log_A: torch.Tensor,
    n_iter: int,
    epsilon: float,
) -> torch.Tensor:
    A = torch.exp(log_A).contiguous()  # forces memory layout of A to be Row Major
    Out = torch.empty_like(A)  # A contiguous -> Out GUARANTEED to have same stride as A
    B, N, _ = A.shape
    grid = (B,)
    BLOCK_SIZE = triton.next_power_of_2(N)

    # launch kernel
    sinkhorn_fused_kernel_A_in_registers[grid](
        A,
        Out,
        A.stride(0),
        A.stride(1),
        A.stride(2),
        N,
        n_iter,  # type: ignore
        BLOCK_SIZE,  # type: ignore
        epsilon,  # type: ignore
    )

    return Out


def sinkhorn_fused_A_in_registers_block_tiling(
    log_A: torch.Tensor,
    n_iter: int,
    epsilon: float,
) -> torch.Tensor:
    A = torch.exp(log_A).contiguous()  # forces memory layout of A to be Row Major
    Out = torch.empty_like(A)  # A contiguous -> Out GUARANTEED to have same stride
    B, N, _ = log_A.shape

    # block tiling
    TARGET_BLOCK_SIZE = 256  # sweep spot
    # block sizing strategy: close to 256, but MUST be multiple of N^2 for reshape to work
    elements_per_matrix = N * N
    assert elements_per_matrix <= TARGET_BLOCK_SIZE  # need small N for coalesced tiling
    # blocks now contain chunks of matrices instead of a single (small) matrix
    matrices_per_block = TARGET_BLOCK_SIZE // elements_per_matrix
    EXACT_BLOCK_SIZE = matrices_per_block * elements_per_matrix
    num_blocks = triton.cdiv(B, matrices_per_block)  # ceil so we don't drop data
    grid = (num_blocks,)

    # launch kernel
    sinkhorn_fused_kernel_A_in_registers_block_tiling[grid](
        A,
        Out,
        A.stride(0),
        A.stride(1),
        A.stride(2),
        B=B,
        N=N,  # type: ignore
        matrices_per_block=matrices_per_block,  # type: ignore
        n_iter=n_iter,  # type: ignore
        BLOCK_SIZE=EXACT_BLOCK_SIZE,  # type: ignore
        epsilon=epsilon,  # type: ignore
    )

    return Out


def sinkhorn_fused_coalesced(
    log_A: torch.Tensor,
    n_iter: int,
    epsilon: float,
) -> torch.Tensor:
    log_A = log_A.contiguous()  # forces memory layout of A to be Row Major
    Out = torch.empty_like(log_A)  # A contiguous -> Out GUARANTEED to have same stride
    B, N, _ = log_A.shape
    total_elements = B * N * N  # needed for masking

    # block tiling
    TARGET_BLOCK_SIZE = 256  # sweep spot
    # block sizing strategy: close to 256, but MUST be multiple of N^2 for reshape to work
    elements_per_matrix = N * N
    assert elements_per_matrix <= TARGET_BLOCK_SIZE  # need small N for coalesced tiling
    # blocks now contain chunks of matrices instead of a single (small) matrix
    matrices_per_block = TARGET_BLOCK_SIZE // elements_per_matrix
    num_blocks = triton.cdiv(B, matrices_per_block)  # ceil so we don't drop data
    grid = (num_blocks,)

    # launch kernel
    sinkhorn_fused_coalesced_kernel[grid](
        log_A,
        Out,
        total_elements=total_elements,
        N=N,  # type: ignore
        matrices_per_block=matrices_per_block,  # type: ignore
        n_iter=n_iter,  # type: ignore
        BLOCK_SIZE=TARGET_BLOCK_SIZE,  # type: ignore
        epsilon=epsilon,  # type: ignore
    )

    return Out


def verify_correctness(
    func: Callable,
    B: int,
    N: int,
    n_iter: int,
    epsilon: float,
    atol: float,
) -> None:
    log_A = torch.randn(B, N, N, device="cuda")
    out = func(log_A, n_iter=n_iter, epsilon=epsilon)
    max_distance_rows = (out.sum(dim=-1) - 1).abs().max().item()
    max_distance_cols = (out.sum(dim=-2) - 1).abs().max().item()
    if not (max_distance_rows < atol and max_distance_cols < atol):
        print(f"{func.__name__=}, {max_distance_cols=:.2g}, {max_distance_rows=:.2g}")


if __name__ == "__main__":
    B_list = [1, 2, 4, 16]
    N_list = [1, 4, 8]
    n_iter = 20
    epsilon = 1e-9
    atol = 1e-5
    funcs = [
        sinkhorn_pytorch,
        sinkhorn_unfused,
        sinkhorn_fused_A_in_global_memory,
        sinkhorn_fused_A_in_registers,
        sinkhorn_fused_A_in_registers_block_tiling,
        sinkhorn_fused_coalesced,
    ]
    for B in B_list:
        for N in N_list:
            print(f"B={B}, N={N}")
            for func in funcs:
                verify_correctness(func, B, N, n_iter, epsilon, atol)
