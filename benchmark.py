import os
from typing import Callable

import pandas as pd
import torch
import triton
from fire import Fire

from plotting import (
    PLOT_DIR,
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


def get_bytes(log_A: torch.Tensor, n_iter: int, provider: str) -> int:
    """Number of bytes accessed from global memory for Sinkhorn's algorithm with `provider`."""
    B, N, _ = log_A.shape
    elem_size = log_A.element_size()
    num_elem = N**2
    elem_read_matrix = num_elem  # global memory -> registers
    elem_write_matrix = num_elem  # registers -> global memory
    match provider:
        # A in global memory
        case (
            "sinkhorn_pytorch"
            | "sinkhorn_pytorch_compiled"
            | "sinkhorn_A_in_global_memory"
        ):
            words_exp = elem_read_matrix + elem_write_matrix
            words_iter = 2 * elem_read_matrix * n_iter
            words_final = elem_read_matrix + elem_write_matrix
            words_total = words_exp + words_iter + words_final
        # A in registers
        case "sinkhorn_A_in_registers" | "sinkhorn_A_in_registers_block_packing":
            words_total = elem_read_matrix + elem_write_matrix
        case _:
            raise ValueError(f"Unknown provider: {provider}")
    bytes_per_matrix = words_total * elem_size
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
        total_flops = get_sinkhorn_flops(B, N, n_iter)
        for provider, func in providers.items():
            total_bytes = get_bytes(log_A, n_iter, provider)
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
            operational_intensity = total_flops / total_bytes
            results.append(
                (
                    B,
                    N,
                    provider,
                    median,
                    q01,
                    q99,
                    memory_bandwidth,
                    compute_throughput,
                    operational_intensity,
                )
            )

    return results


def run_benchmark(
    N: int = 4,  # default mHC value (keep N<=16 for perf)
    n_iter: int = 20,  # default mHC value (increase for better convergence)
    epsilon: float = 1e-6,  # numerical stability
    log2_B_min=0,
    log2_B_max=13,
):
    # large sweep to observe all hardware regimes
    B_list = [1 << (2 * i) for i in range(log2_B_min, log2_B_max)]  # 13
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
        "operational_intensity",
    ]

    print("\n#### Benchmarking ####")
    results = benchmark(providers, B_list, N, n_iter, epsilon)
    print("Done benchmarking.")
    df = pd.DataFrame(results, columns=columns)

    print("\n#### Plotting ###")
    os.makedirs(PLOT_DIR, exist_ok=True)
    for plot_fun in [
        plot_speedup,
        plot_timing,
        plot_compute_throughput,
        plot_memory_bandwidth,
    ]:
        plot_fun(df)
    print("Done plotting.")


if __name__ == "__main__":
    Fire(run_benchmark)
