#!/usr/bin/env python3
"""Generate paper figures for the AAAI draft."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory where figure files will be written.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(7.2, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 6, height_ratios=[1.05, 1.0])

    ax_rate = fig.add_subplot(gs[0, :3])
    ax_flip = fig.add_subplot(gs[0, 3:])
    ax_shift = fig.add_subplot(gs[1, :])

    draw_pass_rates(ax_rate)
    draw_behavior_flips(ax_flip)
    draw_shift_tradeoff(ax_shift)

    fig.suptitle(
        "Visible tests help only when they change behavior in the right direction",
        x=0.51,
        y=1.02,
        fontsize=10.5,
        fontweight="bold",
    )

    pdf_path = args.output_dir / "results_overview.pdf"
    png_path = args.output_dir / "results_overview.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=240)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


def draw_pass_rates(ax: plt.Axes) -> None:
    datasets = ["MBPP+", "HumanEval+", "LCB\nQwen2.5", "LCB\nQwen3.6"]
    conditions = ["NL-only", "NL+tests", "Shuffled", "Irrelevant"]
    values = np.array(
        [
            [63.8, 71.2, 69.0, 43.9],
            [78.0, 78.0, 79.3, 70.7],
            [13.1, 14.9, 13.7, 15.4],
            [39.4, 42.3, 44.6, 38.9],
        ]
    )
    colors = ["#4C78A8", "#2A9D8F", "#F4A261", "#B56576"]
    x = np.arange(len(datasets))
    width = 0.18

    for idx, (condition, color) in enumerate(zip(conditions, colors)):
        offset = (idx - 1.5) * width
        ax.bar(
            x + offset,
            values[:, idx],
            width,
            color=color,
            label=condition,
            edgecolor="white",
            linewidth=0.6,
        )

    for i, delta in enumerate([7.4, 0.0, 1.7, 2.9]):
        y = max(values[i, 0], values[i, 1]) + 4.0
        label = f"{delta:+.1f}" if delta else "+0.0"
        ax.annotate(
            label,
            xy=(x[i], y),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1B4332" if delta > 0 else "#555555",
            fontweight="bold",
        )

    ax.set_title("(a) Main pass rates")
    ax.set_ylabel("Pass rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 88)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        ncols=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
        columnspacing=0.8,
        handlelength=1.2,
    )


def draw_behavior_flips(ax: plt.Axes) -> None:
    labels = [
        "MB\nrel.",
        "MB\nshuf.",
        "MB\nirr.",
        "HE\nrel.",
        "L7\nrel.",
        "L7\nirr.",
        "L27\nrel.",
        "L27\nsyn.",
    ]
    values = np.array([28, 20, -75, 0, 3, 4, 5, 3])
    colors = ["#2A9D8F" if v >= 0 else "#B56576" for v in values]
    x = np.arange(len(labels))

    ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.6)
    ax.axhline(0, color="#333333", linewidth=0.8)
    for xi, value in zip(x, values):
        va = "bottom" if value >= 0 else "top"
        y = value + (2 if value >= 0 else -2)
        ax.text(
            xi,
            y,
            f"{value:+d}",
            ha="center",
            va=va,
            fontsize=8,
            fontweight="bold",
            color="#333333",
        )

    ax.set_title("(b) Behavior flip net gains")
    ax.set_ylabel("Gains - losses")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylim(-86, 36)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def draw_shift_tradeoff(ax: plt.Axes) -> None:
    points = [
        ("MBPP orig\n+tests", 0.0301, 71.2, "#2A9D8F", "o"),
        ("MBPP synth\nhigh1", 0.0234, 65.9, "#4C78A8", "o"),
        ("MBPP synth\nhigh3", 0.0289, 68.3, "#4C78A8", "o"),
        ("MBPP\nassert-only", 0.0428, 54.8, "#B56576", "X"),
        ("LCB synthetic\nhigh3", 0.0105, 14.3, "#6A4C93", "s"),
        ("LCB synthetic\nhigh5", 0.0120, 14.3, "#6A4C93", "s"),
        ("LCB irrelevant\nhigh5", 0.0130, 15.4, "#B56576", "X"),
        ("Qwen3.6 orig\n+tests", 0.0185, 42.3, "#2A9D8F", "D"),
        ("Qwen3.6 synth\nhigh5", 0.0236, 43.4, "#4C78A8", "D"),
        ("Qwen3.6 orig\nirrelevant", 0.0386, 38.9, "#B56576", "X"),
    ]
    label_positions = {
        "MBPP synth\nhigh1": (0.0240, 66.2),
        "MBPP synth\nhigh3": (0.0294, 62.8),
        "MBPP orig\n+tests": (0.0306, 73.4),
        "MBPP\nassert-only": (0.0433, 49.7),
        "Qwen3.6 orig\n+tests": (0.0130, 46.0),
        "Qwen3.6 synth\nhigh5": (0.0241, 44.0),
        "Qwen3.6 orig\nirrelevant": (0.0391, 35.5),
    }

    for label, shift, pass_rate, color, marker in points:
        ax.scatter(
            shift,
            pass_rate,
            s=70,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        if label in label_positions:
            tx, ty = label_positions[label]
            ax.text(tx, ty, label, fontsize=8, va="center")

    ax.plot(
        [0.0105, 0.0120, 0.0130],
        [14.3, 14.3, 15.4],
        color="#888888",
        linewidth=0.9,
        linestyle="--",
        zorder=2,
    )
    ax.text(
        0.0091,
        18.2,
        "LCB synth high3/high5\nand irrelevant",
        fontsize=8,
        va="bottom",
    )
    ax.plot(
        [0.0185, 0.0236, 0.0386],
        [42.3, 43.4, 38.9],
        color="#888888",
        linewidth=0.9,
        linestyle="--",
        zorder=2,
    )

    ax.annotate(
        "larger shift,\nlower pass rate",
        xy=(0.0428, 54.8),
        xytext=(0.0355, 37.5),
        arrowprops=dict(arrowstyle="->", color="#555555", linewidth=0.9),
        fontsize=8,
        ha="center",
        color="#444444",
    )
    ax.set_title("(c) Hidden-state shift is not the same as useful test utilization")
    ax.set_xlabel("Average cosine distance from NL-only hidden state")
    ax.set_ylabel("Pass rate (%)")
    ax.set_xlim(0.008, 0.047)
    ax.set_ylim(8, 78)
    ax.grid(color="#E5E5E5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


if __name__ == "__main__":
    main()
