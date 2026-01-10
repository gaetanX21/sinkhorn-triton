import torch
import triton
import triton.language as tl


def sinkhorn_pytorch(
    log_A: torch.Tensor,  # logits
    n_iter: int,  # increase for better convergence
    epsilon: float = 1e-6,  # numerical stability
) -> torch.Tensor:
    # pytorch baseline for comparison with the triton kernels
    A = torch.exp(log_A)
    B, N, _ = A.shape

    # initialize scalers
    r = torch.ones(B, N, device=A.device)
    c = torch.ones(B, N, device=A.device)

    for _ in range(n_iter):
        r = 1.0 / ((A * c[:, None, :]).sum(dim=-1) + epsilon)  # row normalization
        c = 1.0 / ((A * r[:, :, None]).sum(dim=-2) + epsilon)  # column normalization

    # final scaled matrix
    return A * r[:, :, None] * c[:, None, :]


@triton.jit
def row_normalization_kernel(
    A_ptr,
    r_ptr,
    c_ptr,
    stride_b,
    stride_h,
    stride_w,
    N,
    BLOCK_SIZE: tl.constexpr,
    EPSILON: tl.constexpr,
):
    # we use 2D grid (B, N) where B is batch size
    # pid_0 = which matrix in batch
    # pid_1 = which row in matrix
    pid_batch = tl.program_id(0)
    pid_row = tl.program_id(1)

    # 1. pointers
    row_start_ptr = A_ptr + pid_batch * stride_b + pid_row * stride_h
    c_start_ptr = c_ptr + pid_batch * N

    # 2. loading data
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    # load entire ROW of A -> stride_w (1) -> coalesced memory access (fast)
    a_ptrs = row_start_ptr + offsets * stride_w
    a_vals = tl.load(a_ptrs, mask=mask, other=0.0)
    # load entire vector c
    c_ptrs = c_start_ptr + offsets
    c_vals = tl.load(c_ptrs, mask=mask, other=0.0)

    # 3. math
    total = tl.sum(a_vals * c_vals)

    # 4. write result
    r_val = 1.0 / (total + EPSILON)
    r_out_ptr = r_ptr + pid_batch * N + pid_row
    tl.store(r_out_ptr, r_val)


@triton.jit
def col_normalization_kernel(
    A_ptr,
    r_ptr,
    c_ptr,
    stride_b,
    stride_h,
    stride_w,
    N,
    BLOCK_SIZE: tl.constexpr,
    EPSILON: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_col = tl.program_id(1)

    col_start_ptr = A_ptr + pid_batch * stride_b
    r_start_ptr = r_ptr + pid_batch * N

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N  # safety mask if N is not power of 2

    # XXX: load entire COLUMN of A -> stride_h (N) -> strided memory access (slow)
    a_ptrs = col_start_ptr + offsets * stride_h + pid_col * stride_w
    a_vals = tl.load(a_ptrs, mask=mask, other=0.0)

    r_ptrs = r_start_ptr + offsets
    r_vals = tl.load(r_ptrs, mask=mask, other=0.0)

    total = tl.sum(a_vals * r_vals)

    c_val = 1.0 / (total + EPSILON)
    c_out_ptr = c_ptr + pid_batch * N + pid_col
    tl.store(c_out_ptr, c_val)


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
    EPSILON: tl.constexpr,
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
            new_r_scalar = 1.0 / (row_sum + EPSILON)
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
        c = 1.0 / (c_new_accum + EPSILON)

    # final write -> write result back to global memory
    Out_base = Out_ptr + pid * stride_b
    for i in range(N):
        # re-load A
        row_ptr_start = A_base + i * stride_h
        row_ptrs = row_ptr_start + offsets
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
    N,
    n_iter: tl.constexpr,  # set as constant for loop unrolling (okay for small n_iter e.g. 20)
    BLOCK_SIZE: tl.constexpr,
    EPSILON: tl.constexpr,
):
    pid = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_SIZE)
    A_base = A_ptr + pid * stride_b
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
        r = 1.0 / (denom_r + EPSILON)

        # 2. update c = 1 / (A.T@r).sum(dim=0)
        # we broadcast r to (N,1) and mulitply element-wise, then sum rows
        denom_c = tl.sum(A_full * r[:, None], axis=0)  # sum across rows
        c = 1.0 / (denom_c + EPSILON)

    out = A_full * r[:, None] * c[None, :]
    Out_base = Out_ptr + pid * stride_b
    # NOTE: works because A.strides() == Out.strides() (because A contiguous and Out=torch.empty_like(A))
    out_ptrs = Out_base + offsets[:, None] * stride_h + offsets[None, :] * stride_w
    tl.store(out_ptrs, out, mask=mask)


def sinkhorn_unfused(
    log_A: torch.Tensor,
    n_iter: int = 20,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    # unfused version, iterates between row and column normalization kernels
    # slow because r and c in global memory (VRAM) + kernel launch overhead
    B = log_A.shape[0]
    N = log_A.shape[1]
    A = torch.exp(log_A)
    r = torch.ones(B, N, device=A.device)
    c = torch.ones(B, N, device=A.device)

    grid = (B, N)
    BLOCK_SIZE = triton.next_power_of_2(N)

    for _ in range(n_iter):
        # launch row normalization kernel
        row_normalization_kernel[grid](
            A,
            r,
            c,
            A.stride(0),
            A.stride(1),
            A.stride(2),
            N,
            BLOCK_SIZE,  # type: ignore
            epsilon,  # type: ignore
        )
        # launch column normalization kernel
        col_normalization_kernel[grid](
            A,
            r,
            c,
            A.stride(0),
            A.stride(1),
            A.stride(2),
            N,
            BLOCK_SIZE,  # type: ignore
            epsilon,  # type: ignore
        )

    return A * r[:, :, None] * c[:, None, :]


def sinkhorn_fused_A_in_global_memory(
    log_A: torch.Tensor, n_iter: int = 20, epsilon: float = 1e-6
) -> torch.Tensor:
    A = torch.exp(log_A).contiguous()  # forces memory layout of A to be Row Major
    Out = torch.empty_like(
        A
    )  # A is contiguous so Out is GUARANTEED to have same stride as A
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
    log_A: torch.Tensor, n_iter: int = 20, epsilon: float = 1e-6
) -> torch.Tensor:
    A = torch.exp(log_A).contiguous()  # forces memory layout of A to be Row Major
    Out = torch.empty_like(
        A
    )  # A is contiguous so Out is GUARANTEED to have same stride as A
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


if __name__ == "__main__":
    B = 16
    N = 4
    log_A = torch.randn(B, N, N, device="cuda")
    n_iter = 20
    epsilon = 1e-6

    # test correctness
    for func in [
        sinkhorn_pytorch,
        sinkhorn_unfused,
        sinkhorn_fused_A_in_global_memory,
        sinkhorn_fused_A_in_registers,
    ]:
        out = func(log_A, n_iter=n_iter, epsilon=epsilon)
        max_distance_rows = (out.sum(dim=-1) - 1).abs().max().item()
        max_distance_cols = (out.sum(dim=-2) - 1).abs().max().item()
        assert max_distance_rows < 1e-5 and max_distance_cols < 1e-5
        print(f"kernel {func.__name__} correct")
