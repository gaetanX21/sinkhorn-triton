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
    A = torch.exp(log_A)
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
    N,  # can be made tl.constexpr for small matrices (N<=16) ; will cause loop unrolling bloat for bigger matrices
    n_iter: tl.constexpr,  # loop unrolling
    BLOCK_SIZE: tl.constexpr,
    epsilon: tl.constexpr,
):
    """First attempt: keep A in global memory but r, c now live in registers and we fuse the loop."""
    # 1D grid (B,) -> each block handles one full matrix A of size (N,N)
    pid = tl.program_id(0)

    # initialize scalers in registers
    r = tl.full((BLOCK_SIZE,), 1.0, dtype=tl.float32)
    c = tl.full((BLOCK_SIZE,), 1.0, dtype=tl.float32)

    # base pointer for matrix A[pid]
    A_base = A_ptr + pid * stride_b
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N  # safety (for last block if N not multiple of BLOCK_SIZE)

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
def sinkhorn_A_in_registers_kernel(
    A_ptr,
    Out_ptr,
    stride_b,
    stride_h,
    stride_w,
    N: int,
    n_iter: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    epsilon: tl.constexpr,
):
    """Second attempt: load full matrix A in registers to minimize global memory access.
    Note: this only works for small N (e.g. N<=128) because of register pressure
    i.e. we don't want registers spilling over into "local memory" (VRAM)."""
    pid = tl.program_id(0)

    A_base = A_ptr + pid * stride_b
    offsets = tl.arange(0, BLOCK_SIZE)
    A_ptrs = A_base + offsets[:, None] * stride_h + offsets[None, :] * stride_w
    mask = (offsets[:, None] < N) & (offsets[None, :] < N)
    # load full matrix A[pid] of size (N,N) in registers
    A_full = tl.load(A_ptrs, mask=mask, other=0.0)

    r = tl.full((BLOCK_SIZE,), 1.0, dtype=tl.float32)
    c = tl.full((BLOCK_SIZE,), 1.0, dtype=tl.float32)

    # loop -> everything in registers now, no more loads!
    for _ in range(n_iter):
        denom_r = tl.sum(A_full * c[None, :], axis=1)  # sum across cols
        r = 1.0 / (denom_r + epsilon)

        denom_c = tl.sum(A_full * r[:, None], axis=0)  # sum across rows
        c = 1.0 / (denom_c + epsilon)

    out = A_full * r[:, None] * c[None, :]
    Out_base = Out_ptr + pid * stride_b
    # NOTE: works because A.strides() == Out.strides() (because A contiguous and Out=torch.empty_like(A))
    out_ptrs = Out_base + offsets[:, None] * stride_h + offsets[None, :] * stride_w
    tl.store(out_ptrs, out, mask=mask)


@triton.jit
def sinkhorn_A_in_registers_block_packing_kernel(
    A_ptr,
    Out_ptr,
    stride_b,
    stride_h,
    stride_w,
    B: int,
    N: tl.constexpr,  # loop unrolling (we assume small N for this kernel)
    matrices_per_block: tl.constexpr,
    n_iter: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    epsilon: tl.constexpr,
):
    """Third attempt: load several matrices A in registers to better utilize GPU threads.
    Each block processes 'matrices_per_block' matrices of size (N,N)."""
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
    # load batch of matrices in registers
    A_full = tl.load(A_ptrs, mask=mask, other=0.0)

    r = tl.full((matrices_per_block, N, 1), 1.0, dtype=tl.float32)
    c = tl.full((matrices_per_block, 1, N), 1.0, dtype=tl.float32)

    for _ in range(n_iter):
        denom_r = tl.sum(A_full * c, axis=2, keep_dims=True)
        r = 1.0 / (denom_r + epsilon)

        denom_c = tl.sum(A_full * r, axis=1, keep_dims=True)
        c = 1.0 / (denom_c + epsilon)

    out = A_full * r * c
    Out_base = Out_ptr + pid * BLOCK_SIZE
    out_ptrs = Out_base + offs_3d
    tl.store(out_ptrs, out, mask=mask)


@triton.jit
def sinkhorn_coalesced_kernel(
    log_A_ptr,  # read log_A to save memory bandwidth (exp will cost one back and forth in registers)
    Out_ptr,
    total_elements: int,  # for masking (safety)
    N: tl.constexpr,
    n_iter: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    matrices_per_block: tl.constexpr,
    epsilon: tl.constexpr,
):
    """Fourth attempt: coalesced memory access by having each block process several matrices.
    Memory is treated as a 1D array of floats, each block loads a contiguous chunk of data.
    This ensures perfect coalescing when reading from global memory."""
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

    r = tl.full((matrices_per_block, N, 1), 1.0, dtype=tl.float32)
    c = tl.full((matrices_per_block, 1, N), 1.0, dtype=tl.float32)

    for _ in range(n_iter):
        denom_r = tl.sum(A_view * c, axis=2, keep_dims=True)
        r = 1.0 / (denom_r + epsilon)

        denom_c = tl.sum(A_view * r, axis=1, keep_dims=True)
        c = 1.0 / (denom_c + epsilon)

    out_view = A_view * r * c
    # flatten back to match load structure
    out_vals = tl.reshape(out_view, (BLOCK_SIZE,))
    out_ptrs = Out_ptr + global_idxs
    tl.store(out_ptrs, out_vals, mask=mask)


def sinkhorn_A_in_global_memory(
    log_A: torch.Tensor,
    n_iter: int,
    epsilon: float,
) -> torch.Tensor:
    A = torch.exp(log_A).contiguous()  # forces memory layout of A to be Row Major
    Out = torch.empty_like(A)  # A contiguous -> Out guaranteed to have same stride as A
    B, N, _ = A.shape
    grid = (B,)
    BLOCK_SIZE = triton.next_power_of_2(N)

    # launch kernel
    sinkhorn_A_in_global_memory_kernel[grid](
        A,
        Out,
        stride_b=A.stride(0),
        stride_h=A.stride(1),
        stride_w=A.stride(2),
        N=N,
        n_iter=n_iter,  # type: ignore
        BLOCK_SIZE=BLOCK_SIZE,  # type: ignore
        epsilon=epsilon,  # type: ignore
    )

    return Out


def sinkhorn_A_in_registers(
    log_A: torch.Tensor,
    n_iter: int,
    epsilon: float,
) -> torch.Tensor:
    A = torch.exp(log_A).contiguous()
    Out = torch.empty_like(A)
    B, N, _ = A.shape
    grid = (B,)
    BLOCK_SIZE = triton.next_power_of_2(N)

    # launch kernel
    sinkhorn_A_in_registers_kernel[grid](
        A,
        Out,
        stride_b=A.stride(0),
        stride_h=A.stride(1),
        stride_w=A.stride(2),
        N=N,
        n_iter=n_iter,  # type: ignore
        BLOCK_SIZE=BLOCK_SIZE,  # type: ignore
        epsilon=epsilon,  # type: ignore
    )

    return Out


def sinkhorn_A_in_registers_block_packing(
    log_A: torch.Tensor,
    n_iter: int,
    epsilon: float,
) -> torch.Tensor:
    A = torch.exp(log_A).contiguous()
    Out = torch.empty_like(A)
    B, N, _ = log_A.shape

    # block packing
    TARGET_BLOCK_SIZE = 256  # sweep spot
    # block sizing strategy: close to 256, but MUST be multiple of N^2 for reshape to work
    elements_per_matrix = N * N
    assert elements_per_matrix <= TARGET_BLOCK_SIZE  # need small N for coalesced access
    # blocks now contain batch of matrices instead of single matrix
    matrices_per_block = TARGET_BLOCK_SIZE // elements_per_matrix
    EXACT_BLOCK_SIZE = matrices_per_block * elements_per_matrix
    num_blocks = triton.cdiv(B, matrices_per_block)  # ceil so we don't drop data
    grid = (num_blocks,)

    # launch kernel
    sinkhorn_A_in_registers_block_packing_kernel[grid](
        A_ptr=A,
        Out_ptr=Out,
        stride_b=A.stride(0),
        stride_h=A.stride(1),
        stride_w=A.stride(2),
        B=B,
        N=N,  # type: ignore
        matrices_per_block=matrices_per_block,  # type: ignore
        n_iter=n_iter,  # type: ignore
        BLOCK_SIZE=EXACT_BLOCK_SIZE,  # type: ignore
        epsilon=epsilon,  # type: ignore
    )

    return Out


def sinkhorn_coalesced(
    log_A: torch.Tensor,
    n_iter: int,
    epsilon: float,
) -> torch.Tensor:
    log_A = log_A.contiguous()
    Out = torch.empty_like(log_A)
    B, N, _ = log_A.shape
    total_elements = B * N * N  # needed for masking

    # block packing
    TARGET_BLOCK_SIZE = 256  # sweep spot
    # block sizing strategy: close to 256, but MUST be multiple of N^2 for reshape to work
    elements_per_matrix = N * N
    assert elements_per_matrix <= TARGET_BLOCK_SIZE  # need small N for coalesced access
    # blocks now contain chunks of matrices instead of a single (small) matrix
    matrices_per_block = TARGET_BLOCK_SIZE // elements_per_matrix
    num_blocks = triton.cdiv(B, matrices_per_block)  # ceil so we don't drop data
    grid = (num_blocks,)

    # launch kernel
    sinkhorn_coalesced_kernel[grid](
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
    B_list = [1, 4, 16]
    N_list = [2, 4, 8]
    n_iter = 20
    epsilon = 1e-9
    atol = 1e-5
    funcs = [
        sinkhorn_pytorch,
        sinkhorn_pytorch_compiled,
        sinkhorn_A_in_global_memory,
        sinkhorn_A_in_registers,
        sinkhorn_A_in_registers_block_packing,
        sinkhorn_coalesced,
    ]
    for B in B_list:
        for N in N_list:
            print(f"B={B}, N={N}")
            for func in funcs:
                verify_correctness(func, B, N, n_iter, epsilon, atol)
