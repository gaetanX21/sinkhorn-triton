from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FIG_DIR = "figures"
VIVID_GREEN = "#00ff22"  # pretty cool imo

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)


def format_func(value: int) -> str:
    if value < 1024:
        return str(value)
    elif value < 1048576:
        return f"{int(value / 1024)}k"
    else:
        return f"{int(value / 1048576)}M"


def standard_plot(filename: str):
    """Wrapper to standardize plots and save them."""

    def wrapper(plot_fun: Callable) -> Callable:
        def inner(df: pd.DataFrame) -> None:
            plt.figure(figsize=(10, 5))

            # plot data
            plot_fun(df)

            # x-axis
            plt.xscale("log", base=2)
            plt.xlabel("Batch Size")
            B_values = df["B"].unique()
            plt.xticks(B_values, [format_func(B) for B in B_values])

            # save
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{FIG_DIR}/{filename}", dpi=200)
            print(f"Saved {FIG_DIR}/{filename}")
            plt.close()

        return inner

    return wrapper


@standard_plot("speedup.png")
def plot_speedup(df: pd.DataFrame) -> None:
    baseline = df[df["provider"] == "sinkhorn_pytorch_compiled"].set_index("B")[
        "timing_median"
    ]
    block_packing = df[
        df["provider"] == "sinkhorn_A_in_registers_block_packing"
    ].set_index("B")["timing_median"]

    # Compute Speedup: Baseline Time / Optimized Time
    speedup_df = (baseline / block_packing).reset_index()
    speedup_df.columns = ["B", "Speedup"]

    # Line plot with markers
    sns.lineplot(
        data=speedup_df,
        x="B",
        y="Speedup",
        marker="o",
        markersize=8,
        linewidth=2.5,
        color=VIVID_GREEN,
    )

    # Annotate values on the plot
    for x, y in zip(speedup_df["B"], speedup_df["Speedup"]):
        plt.text(
            x,
            y + 2,
            f"{y:.0f}x",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="k",
        )

    plt.ylim(0, max(speedup_df["Speedup"]) * 1.2)
    plt.ylabel("Speedup Factor")
    plt.title("Speedup: Triton Block Packing vs. Torch Compiled")


@standard_plot("timing.png")
def plot_timing(df: pd.DataFrame) -> None:
    sns.lineplot(
        data=df,
        x="B",
        y="timing_median",
        hue="provider",
        marker="o",
        zorder=2,
    )

    providers = df["provider"].unique()
    palette = sns.color_palette(n_colors=len(providers))
    provider_color_map = dict(zip(providers, palette))
    for provider in providers:
        subset = df[df["provider"] == provider].sort_values("B")
        color = provider_color_map[provider]
        # Add the shading
        plt.fill_between(
            subset["B"],
            subset["timing_q01"],
            subset["timing_q99"],
            color=color,
            alpha=0.2,
            linewidth=0,
            zorder=1,
        )

    plt.yscale("log", base=10)
    plt.ylabel("Median Execution Time (ms)")
    plt.title("Kernel Execution Time vs Batch Size")


@standard_plot("compute_throughput")
def plot_compute_throughput(df: pd.DataFrame) -> None:
    peak_compute_throughput = df["compute_throughput"].max()
    plt.axhline(
        peak_compute_throughput,
        label=f"peak compute throughput: {peak_compute_throughput:.1f} TFLOPs",
        ls="--",
        c=VIVID_GREEN,
        lw=2,
    )

    # 1. Main Line Plot
    sns.lineplot(
        data=df,
        x="B",
        y="compute_throughput",
        hue="provider",
        marker="o",
    )

    plt.yscale("log", base=10)
    plt.ylabel("Compute Throughput (TFLOPs)")
    plt.title("Compute Throughput vs Batch Size")


@standard_plot("memory_bandwidth")
def plot_memory_bandwidth(df: pd.DataFrame) -> None:
    peak_memory__bandwidth = df["memory_bandwidth"].max()
    plt.axhline(
        peak_memory__bandwidth,
        label=f"peak memory bandwidth: {peak_memory__bandwidth:.0f} GB/s",
        ls="--",
        c=VIVID_GREEN,
        lw=2,
    )

    sns.lineplot(
        data=df,
        x="B",
        y="memory_bandwidth",
        hue="provider",
        marker="o",
    )

    plt.yscale("log", base=10)
    plt.ylabel("Memory Bandwidth (GB/s)")
    plt.title("Memory Bandwidth vs Batch Size")
