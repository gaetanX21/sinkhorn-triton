import os
from typing import Callable

import pandas as pd
import torch
import triton

from plotting import (
    FIG_DIR,
    plot_compute_throughput,
    plot_memory_bandwidth,
    plot_speedup,
    plot_timing,
)
from triton_kernels import (
    sinkhorn_A_in_global_memory,
    sinkhorn_A_in_registers,
    sinkhorn_A_in_registers_block_packing,
    sinkhorn_pytorch,
    sinkhorn_pytorch_compiled,
)


def get_sinkhorn_flops(B: int, N: int, n_iter: int) -> int:
    """Number of FLOPs required to conduct Sinkhorn's algorith with `n_iter` on B (N,N) matrices."""
    ops_per_normalization = 2 * N**2 + N  # N^2 (mult) + N^2 (add) + N (div)
    ops_per_iter = 2 * ops_per_normalization  # row and col
    ops_final = 2 * N**2  # multiply A by row/col normalizer (N^2 each)
    ops_per_matrix = ops_per_iter * n_iter + ops_final
    return B * ops_per_matrix


def get_bytes(B: int, N: int, elem_size: int) -> int:
    """Number of bytes to read from (once) and write to (once) global memory for B (N,N) matrices."""
    # NOTE: valid only for sinkhorn_fused_A_in_registers kernel
    num_elem = N**2
    elem_read_matrix = num_elem  # global memory -> registers
    elem_write_matrix = num_elem  # registers -> global memory
    bytes_per_matrix = (elem_read_matrix + elem_write_matrix) * elem_size
    return B * bytes_per_matrix


def benchmark(
    providers: dict[str, Callable],
    B_list: list[int],
    N: int,
    n_iter: int,
    epsilon: float,
) -> list[tuple]:
    results = []
    for B in B_list:
        print(f"{B=}")
        log_A = torch.randn(B, N, N, device="cuda", dtype=torch.float32)
        total_bytes = get_bytes(B, N, log_A.element_size())
        total_flops = get_sinkhorn_flops(B, N, n_iter)
        for provider, func in providers.items():
            q01, median, q99 = triton.testing.do_bench(
                lambda: func(log_A, n_iter, epsilon),
                # robustness settings
                warmup=25,
                rep=100,
                quantiles=[0.01, 0.5, 0.99],
            )  # type: ignore
            # note: timing is in ms, hence division by 1e6 (and not 1e9) to get GB/s
            memory_bandwidth = total_bytes / median / 1e6  # GB/s
            compute_throughput = total_flops / median / 1e9  # TFLOPS
            results.append(
                (B, N, provider, median, q01, q99, memory_bandwidth, compute_throughput)
            )

    return results


def main():
    # large sweep to observe all hardware regimes
    B_list = [1 << (2 * i) for i in range(13)]
    N = 4  # we stick to mHC's N=4 for simplicity (line search much easier than grid search)

    n_iter = 20  # we stick to 20 iterations
    epsilon = 1e-6  # safety
    providers = {
        "sinkhorn_pytorch": sinkhorn_pytorch,
        "sinkhorn_pytorch_compiled": sinkhorn_pytorch_compiled,
        "sinkhorn_A_in_global_memory": sinkhorn_A_in_global_memory,
        "sinkhorn_A_in_registers": sinkhorn_A_in_registers,
        "sinkhorn_A_in_registers_block_packing": sinkhorn_A_in_registers_block_packing,
    }

    columns = [
        "B",
        "N",
        "provider",
        "timing_median",
        "timing_q01",
        "timing_q99",
        "memory_bandwidth",
        "compute_throughput",
    ]

    print("\n#### Benchmarking ####")
    results = benchmark(providers, B_list, N, n_iter, epsilon)
    print("Done benchmarking.")
    df = pd.DataFrame(results, columns=columns)

    print("\n#### Plotting ###")
    os.makedirs(FIG_DIR, exist_ok=True)
    for plot_fun in [
        plot_speedup,
        plot_timing,
        plot_compute_throughput,
        plot_memory_bandwidth,
    ]:
        plot_fun(df)
    print("Done plotting.")


if __name__ == "__main__":
    main()
