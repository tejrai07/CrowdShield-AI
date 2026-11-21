"""G1 CLI: sanity-check analysis of the generated synthetic dataset.

Reports class distribution and per-label means of key features (does NOT
tune the generator to maximize these correlations — see README "Phase G1"
for why forcing artificially clean separability would be a red flag, not
a success). Optionally writes a few lightweight plots. This script is
OUTSIDE the runtime pipeline — it's a one-off analysis tool, not a
dashboard.

Usage:
    python scripts/analyze_synthetic_data.py \
        --dataset sample_data/synthetic/synthetic_dataset.csv \
        --plots-dir sample_data/synthetic/plots
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from data.synthetic_generator import load_dataset  # noqa: E402
from logging_setup import setup_logging  # noqa: E402

LABEL_ORDER = ["SAFE", "LOW", "MODERATE", "HIGH", "CRITICAL"]


def print_class_distribution(df: pd.DataFrame) -> None:
    counts = df["label"].value_counts().reindex(LABEL_ORDER)
    total = len(df)
    print("\n=== Class distribution (rows) ===")
    for label, count in counts.items():
        pct = (count / total) * 100 if total else 0.0
        print(f"  {label:<10} {count:>6}  ({pct:5.1f}%)")

    seq_counts = df.groupby("label")["sequence_id"].nunique().reindex(LABEL_ORDER)
    print("\n=== Class distribution (unique sequences touching this label) ===")
    for label, count in seq_counts.items():
        print(f"  {label:<10} {count:>4}")


def print_feature_correlation_sanity(df: pd.DataFrame) -> None:
    print("\n=== Per-label feature means (sanity check, NOT tuned for separability) ===")
    columns = [
        "density", "avg_speed_px_s", "direction_variance",
        "w5_density_slope", "w5_net_flow_rate", "w5_bottleneck_pressure",
    ]
    summary = df.groupby("label")[columns].mean().reindex(LABEL_ORDER)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 120):
        print(summary)
    print(
        "\nNote: relationships are NOT expected to be perfectly monotonic — e.g. "
        "CONFLICTING_FLOW and BOTTLENECK are both labeled HIGH but reach that "
        "label through different mechanisms (elevated direction_variance vs. "
        "elevated density/bottleneck pressure), by design (see spec section 4: "
        "'danger should not depend solely on density')."
    )


def print_split_summary(df: pd.DataFrame) -> None:
    if "split" not in df.columns:
        print("\n(no 'split' column present — dataset was not split)")
        return
    print("\n=== Split summary ===")
    seq_counts = df.groupby("split")["sequence_id"].nunique()
    row_counts = df.groupby("split").size()
    for split_name in ["train", "val", "test"]:
        if split_name in seq_counts.index:
            print(f"  {split_name:<6} sequences={seq_counts[split_name]:>3}  rows={row_counts[split_name]:>6}")


def maybe_write_plots(df: pd.DataFrame, plots_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available — skipping plots (this is optional).")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    # One representative pure-scenario sequence per scenario, for clarity.
    pure_seqs = df[df["scenario"].isin(LABEL_ORDER + [
        "NORMAL_STABLE", "GRADUAL_BUILDUP", "RAPID_SURGE", "BOTTLENECK",
        "CONFLICTING_FLOW", "CRITICAL_ESCALATION",
    ]) | df["sequence_id"].str.contains("_pure_")]
    first_realization = pure_seqs[pure_seqs["sequence_id"].str.endswith("_000")]

    def _plot(column: str, ylabel: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        for sequence_id, group in first_realization.groupby("sequence_id"):
            ax.plot(group["timestamp_s"], group[column], label=group["scenario"].iloc[0])
        ax.set_xlabel("time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Synthetic scenarios — {ylabel}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plots_dir / filename, dpi=100)
        plt.close(fig)

    _plot("density", "density (people/m²)", "scenario_density.png")
    _plot("avg_speed_px_s", "speed (px/s)", "scenario_speed.png")
    _plot("direction_variance", "direction variance [0,1]", "scenario_direction_variance.png")
    _plot("w5_net_flow_rate", "net flow rate (people/s, 5s window)", "scenario_flow.png")

    print(f"\nPlots written to: {plots_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CrowdShield G1 synthetic dataset analysis")
    parser.add_argument("--dataset", default="sample_data/synthetic/synthetic_dataset.csv")
    parser.add_argument("--plots-dir", default="sample_data/synthetic/plots")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    setup_logging("INFO")

    dataset_path = REPO_ROOT / args.dataset if not Path(args.dataset).is_absolute() else Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}. Run scripts/generate_synthetic_data.py first.")
        sys.exit(1)

    df = load_dataset(dataset_path)
    print(f"Loaded {len(df)} rows, {df['sequence_id'].nunique()} sequences from {dataset_path}")

    print_class_distribution(df)
    print_feature_correlation_sanity(df)
    print_split_summary(df)

    if not args.no_plots:
        plots_dir = REPO_ROOT / args.plots_dir if not Path(args.plots_dir).is_absolute() else Path(args.plots_dir)
        maybe_write_plots(df, plots_dir)


if __name__ == "__main__":
    main()
