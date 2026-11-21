"""G1 CLI: generate the synthetic crowd-risk dataset and save it to disk.

Usage:
    python scripts/generate_synthetic_data.py \
        --scenarios data/scenarios.yaml \
        --output-dir sample_data/synthetic \
        --seed 42
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.loader import load_generator_config  # noqa: E402
from data.synthetic_generator import apply_split, generate_dataset, save_dataset  # noqa: E402
from logging_setup import setup_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="CrowdShield G1 synthetic dataset generator")
    parser.add_argument("--scenarios", default="data/scenarios.yaml")
    parser.add_argument("--output-dir", default="sample_data/synthetic")
    parser.add_argument("--seed", type=int, default=None, help="Override base_seed from scenarios.yaml")
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument(
        "--duration", type=float, default=None,
        help=(
            "Override duration_s from scenarios.yaml. Used by I1 to generate a "
            "longer-duration dataset with non-null future_5min_label/"
            "future_10min_label coverage — the shipped G1 default (120s) has "
            "none, since 120s < the 300s/600s horizons (see G1 README)."
        ),
    )
    parser.add_argument("--gzip", action="store_true", help="write synthetic_dataset.csv.gz (pandas reads it transparently)")
    parser.add_argument("--realizations-per-scenario", type=int, default=None, help="Override realizations_per_scenario")
    parser.add_argument("--realizations-per-transition", type=int, default=None, help="Override realizations_per_transition")
    args = parser.parse_args()

    logger = setup_logging("INFO")

    scenarios_path = REPO_ROOT / args.scenarios if not Path(args.scenarios).is_absolute() else Path(args.scenarios)
    gen_cfg = load_generator_config(scenarios_path)
    overrides = {}
    if args.seed is not None:
        overrides["base_seed"] = args.seed
    if args.duration is not None:
        overrides["duration_s"] = args.duration
    if args.realizations_per_scenario is not None:
        overrides["realizations_per_scenario"] = args.realizations_per_scenario
    if args.realizations_per_transition is not None:
        overrides["realizations_per_transition"] = args.realizations_per_transition
    if overrides:
        gen_cfg = gen_cfg.model_validate({**gen_cfg.model_dump(), **overrides})

    logger.info(
        "Generating synthetic dataset: %d scenarios x %d pure realizations, %d transitions x %d realizations",
        len(gen_cfg.scenarios), gen_cfg.realizations_per_scenario,
        len(gen_cfg.transitions), gen_cfg.realizations_per_transition,
    )

    t0 = time.perf_counter()
    df, metadata = generate_dataset(gen_cfg)
    elapsed = time.perf_counter() - t0

    df = apply_split(df, gen_cfg, seed=args.split_seed)
    metadata["split_seed"] = args.split_seed
    metadata["split_fracs"] = {"train": gen_cfg.train_frac, "val": gen_cfg.val_frac, "test": gen_cfg.test_frac}
    split_counts = df.groupby("split")["sequence_id"].nunique().to_dict()
    metadata["split_sequence_counts"] = split_counts

    output_dir = REPO_ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    csv_path, metadata_path = save_dataset(df, metadata, output_dir, compress=args.gzip)

    logger.info("Generated %d rows across %d sequences in %.2fs", len(df), metadata["n_sequences"], elapsed)
    logger.info("Label counts: %s", metadata["label_counts"])
    logger.info("Split (by sequence): %s", split_counts)
    logger.info("Dataset written to: %s", csv_path)
    logger.info("Metadata written to: %s", metadata_path)


if __name__ == "__main__":
    main()
