"""Synthetic crowd-risk scenario generator (Phase G1).

Generates controlled, reproducible, feature-level temporal sequences for
the future H1 risk model. This module does NOT simulate video, run YOLO,
or run ByteTrack — it synthesizes zone-level aggregate observations AS IF
E1 had already computed them from real tracks, then feeds each synthetic
observation through the REAL F1 `TemporalFeatureBuilder` to derive rolling
window features. This guarantees perfect formula/schema parity between
synthetic and real data — no duplicated slope/rate math.

IMPORTANT — WHAT THIS DATA IS AND ISN'T:
    This module produces `data_source = "synthetic"` engineering data
    only. It is NOT real-world crowd-safety ground truth, and the scenario
    parameters in `data/scenarios.yaml` are explicit engineering
    assumptions, not validated thresholds. See README "Phase G1" for the
    full disclaimer — every generated row also carries a literal
    `data_source` column so this can never be silently forgotten
    downstream.

DESIGN:
    generate_sequence()  -> one sequence's DataFrame (pure scenario or a
                             scripted transition between two scenarios)
    generate_dataset()   -> many sequences, concatenated, per a
                             GeneratorConfig
    split_dataset()      -> adds a "split" column, assigned by whole
                             sequence_id (never splits a sequence's rows
                             across train/val/test — that would leak)
    save_dataset() / load_dataset() -> CSV + JSON metadata round-trip
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data.schemas import GeneratorConfig, ScenarioConfig, TransitionConfig
from features.crowd_metrics import ZoneMetrics
from features.temporal import TemporalFeatureBuilder

GENERATOR_VERSION = "g1.0.0"

# Windows are fixed at exactly [5, 15, 30, 60] throughout the project
# (see config/schema.py::TemporalConfig) — G1 reuses the same constant
# rather than re-deriving it, to guarantee the synthetic feature columns
# match the real pipeline's schema exactly.
WINDOWS_SECONDS = [5, 15, 30, 60]

FUTURE_HORIZONS_S = {"future_5min_label": 300.0, "future_10min_label": 600.0}


def _raw_signal(cfg: ScenarioConfig, t_local: float) -> tuple[float, float, float, float, float, float]:
    """Pre-noise baseline signal for one scenario at t_local seconds since
    that scenario's own regime effectively started (its own local clock —
    see the transition-blending logic in generate_sequence for how this
    is used during a scripted transition)."""
    density = max(0.0, cfg.base_density + cfg.density_trend * t_local)
    if cfg.max_density is not None:
        density = min(density, cfg.max_density)  # plateau instead of unbounded linear extrapolation
    speed = max(0.0, cfg.base_speed_px_s + cfg.speed_trend_px_s2 * t_local)
    if cfg.min_speed_px_s is not None:
        speed = max(speed, cfg.min_speed_px_s)
    direction_variance = min(1.0, max(0.0, cfg.base_direction_variance + cfg.direction_variance_trend * t_local))
    direction = cfg.base_direction_deg % 360.0
    return density, speed, direction_variance, direction, cfg.inflow_rate, cfg.outflow_rate


def generate_sequence(
    scenario_by_name: dict[str, ScenarioConfig],
    sequence_id: str,
    zone_id: str,
    duration_s: float,
    sample_interval_s: float,
    seed: int,
    density_growth_threshold: float,
    scenario_name: str | None = None,
    transition: TransitionConfig | None = None,
    transition_start_s: float | None = None,
) -> pd.DataFrame:
    """Generate one sequence: either a pure single-scenario sequence
    (`scenario_name` given) or a scripted transition between two
    scenarios (`transition` + `transition_start_s` given). Exactly one of
    these two modes must be used.

    LABELING RULE during a transition: the label switches to
    `transition.to_scenario`'s label immediately at `transition_start_s`.
    The underlying feature VALUES continue to blend smoothly from scenario
    A's regime into scenario B's regime over `transition.transition_duration_s`
    seconds after that point — so for a short window after the label
    changes, the features still partially resemble the previous class.
    This is intentional: it is the primary mechanism for the "controlled
    overlap between classes" requirement (labels are not a trivial
    single-feature threshold; feature-value overlap near class boundaries
    is real and expected).

    NO LEAKAGE: every value at timestamp t is computed only from t and
    earlier — the transition's blend weight and scenario B's local clock
    both depend only on t and transition_start_s (both known at time t),
    never on any future timestamp.
    """
    if (scenario_name is None) == (transition is None):
        raise ValueError("Provide exactly one of scenario_name or transition")

    rng = np.random.default_rng(seed)
    n_steps = int(round(duration_s / sample_interval_s))
    timestamps = [round(i * sample_interval_s, 6) for i in range(n_steps)]

    if scenario_name is not None:
        cfg = scenario_by_name[scenario_name]
        area_m2 = cfg.area_m2
        scenario_label_str = scenario_name
    else:
        cfg_a = scenario_by_name[transition.from_scenario]
        area_m2 = cfg_a.area_m2  # zone's physical area is fixed for the whole sequence
        scenario_label_str = f"TRANSITION:{transition.from_scenario}->{transition.to_scenario}"

    builder = TemporalFeatureBuilder(
        zone_ids=[zone_id],
        windows_seconds=WINDOWS_SECONDS,
        density_growth_threshold=density_growth_threshold,
        max_history_seconds=float(max(WINDOWS_SECONDS)),
    )

    rows: list[dict] = []
    w = 1.0
    for t in timestamps:
        if scenario_name is not None:
            density, speed, dvar, direction, inflow_rate, outflow_rate = _raw_signal(cfg, t)
            label = cfg.label
            noise_level = cfg.noise_level
        else:
            cfg_b = scenario_by_name[transition.to_scenario]
            if t < transition_start_s:
                density, speed, dvar, direction, inflow_rate, outflow_rate = _raw_signal(cfg_a, t)
                label = cfg_a.label
                noise_level = cfg_a.noise_level
            else:
                w = min(1.0, (t - transition_start_s) / transition.transition_duration_s)
                a_vals = _raw_signal(cfg_a, t)  # A's own clock keeps running uninterrupted
                b_vals = _raw_signal(cfg_b, t - transition_start_s)  # B's clock starts fresh here
                density = (1 - w) * a_vals[0] + w * b_vals[0]
                speed = (1 - w) * a_vals[1] + w * b_vals[1]
                dvar = (1 - w) * a_vals[2] + w * b_vals[2]
                direction = (1 - w) * a_vals[3] + w * b_vals[3]
                inflow_rate = (1 - w) * a_vals[4] + w * b_vals[4]
                outflow_rate = (1 - w) * a_vals[5] + w * b_vals[5]
                noise_level = (1 - w) * cfg_a.noise_level + w * cfg_b.noise_level
                label = cfg_b.label  # label switches immediately at transition_start_s

        # Controlled stochastic noise — physically bounded afterward.
        density = max(0.0, density + rng.normal(0.0, noise_level * max(density, 0.1)))
        speed = max(0.0, speed + rng.normal(0.0, noise_level * max(speed, 1.0)))
        dvar = min(1.0, max(0.0, dvar + rng.normal(0.0, noise_level * 0.1)))
        direction = (direction + rng.normal(0.0, noise_level * 20.0)) % 360.0

        inflow_count = int(rng.poisson(max(0.0, inflow_rate) * sample_interval_s))
        outflow_count = int(rng.poisson(max(0.0, outflow_rate) * sample_interval_s))
        people_count = max(0, int(round(density * area_m2)))
        speed_variance = (noise_level * max(speed, 1.0) * 0.3) ** 2

        # --- movement semantics, reproducing features/crowd_metrics.py (E1) exactly ---------------------------
        # no people            -> no tracks      -> speed/variance/direction/direction-variance all UNDEFINED (None)
        # people, not moving   -> speed 0.0 and variance 0.0 are VALID zeros; direction UNDEFINED (zero displacement)
        # observation dropout  -> people present but no usable track history this step -> all movement metrics None
        if scenario_name is not None:
            mode, dropout_p = cfg.movement_mode, cfg.metric_dropout_prob
        elif t < transition_start_s or w < 0.5:
            mode, dropout_p = cfg_a.movement_mode, cfg_a.metric_dropout_prob
        else:
            mode, dropout_p = cfg_b.movement_mode, cfg_b.metric_dropout_prob
        if people_count == 0:
            speed = speed_variance = direction = dvar = None
        elif mode == "stationary":
            speed, speed_variance, direction, dvar = 0.0, 0.0, None, None
        if dropout_p > 0.0 and rng.random() < dropout_p and people_count > 0:  # draw ONLY when configured (legacy streams untouched)
            speed = speed_variance = direction = dvar = None

        metrics = ZoneMetrics(
            zone_id=zone_id,
            timestamp_s=t,
            people_count=people_count,
            density=density,
            avg_speed_px_s=speed,
            speed_variance=speed_variance,
            flow_direction_deg=direction,
            direction_variance=dvar,
            inflow_count=inflow_count,
            outflow_count=outflow_count,
        )

        temporal = builder.update({zone_id: metrics})[zone_id]

        row = {
            "sequence_id": sequence_id,
            "scenario": scenario_label_str,
            "zone_id": zone_id,
            "timestamp_s": t,
            "data_source": "synthetic",
            "label": label,
            "people_count": people_count,
            "density": density,
            "avg_speed_px_s": speed,
            "speed_variance": speed_variance,
            "flow_direction_deg": direction,
            "direction_variance": dvar,
            "inflow_count": inflow_count,
            "outflow_count": outflow_count,
            "surge_flag": temporal.surge_flag,
        }
        for window_seconds, w_feat in temporal.windows.items():
            prefix = f"w{window_seconds}_"
            row[prefix + "sample_count"] = w_feat.sample_count
            row[prefix + "coverage_s"] = w_feat.coverage_s
            row[prefix + "is_mature"] = w_feat.is_mature
            row[prefix + "density_mean"] = w_feat.density_mean
            row[prefix + "density_max"] = w_feat.density_max
            row[prefix + "density_min"] = w_feat.density_min
            row[prefix + "speed_mean"] = w_feat.speed_mean
            row[prefix + "speed_min"] = w_feat.speed_min
            row[prefix + "direction_variance_mean"] = w_feat.direction_variance_mean
            row[prefix + "direction_variance_max"] = w_feat.direction_variance_max
            row[prefix + "density_slope"] = w_feat.density_slope
            row[prefix + "speed_slope"] = w_feat.speed_slope
            row[prefix + "direction_variance_slope"] = w_feat.direction_variance_slope
            row[prefix + "inflow_rate"] = w_feat.inflow_rate
            row[prefix + "outflow_rate"] = w_feat.outflow_rate
            row[prefix + "net_flow_rate"] = w_feat.net_flow_rate
            row[prefix + "bottleneck_pressure"] = w_feat.bottleneck_pressure

        rows.append(row)

    df = pd.DataFrame(rows)

    # Future labels: shift(-N) at row i pulls the label from row i+N (a
    # LATER row in this same sequence) into a NEW column at row i. This
    # only ever reads a label that will exist later in time and writes it
    # into a column that didn't exist before — no existing feature column
    # is touched, so this cannot leak future information into the current
    # feature columns. Rows too close to the sequence's end get NaN
    # (explicitly missing, never fabricated).
    for column_name, horizon_s in FUTURE_HORIZONS_S.items():
        shift_steps = int(round(horizon_s / sample_interval_s))
        df[column_name] = df["label"].shift(-shift_steps)

    return df


def generate_dataset(gen_cfg: GeneratorConfig) -> tuple[pd.DataFrame, dict]:
    """Generate the full synthetic dataset: `realizations_per_scenario`
    pure sequences per scenario, plus `realizations_per_transition`
    scripted-transition sequences per configured transition pair.

    Returns (dataframe, metadata_dict). Fully deterministic for a fixed
    `gen_cfg` (including `base_seed`) — every sequence's seed is derived
    from `base_seed` plus a fixed, deterministic counter, never from
    wall-clock time or other non-reproducible state.
    """
    scenario_by_name = {s.name: s for s in gen_cfg.scenarios}
    sequences: list[pd.DataFrame] = []
    seed_counter = 0

    for scenario in gen_cfg.scenarios:
        for i in range(gen_cfg.realizations_per_scenario):
            seed = gen_cfg.base_seed + seed_counter
            seed_counter += 1
            sequence_id = f"{scenario.name}_pure_{i:03d}"
            df = generate_sequence(
                scenario_by_name=scenario_by_name,
                sequence_id=sequence_id,
                zone_id=gen_cfg.zone_id,
                duration_s=gen_cfg.duration_s,
                sample_interval_s=gen_cfg.sample_interval_s,
                seed=seed,
                density_growth_threshold=gen_cfg.density_growth_threshold,
                scenario_name=scenario.name,
            )
            sequences.append(df)

    # A dedicated RNG (seeded, deterministic) picks each transition
    # realization's randomized start time — "different transition
    # timings" per the spec, but reproducible for a fixed base_seed.
    timing_rng = np.random.default_rng(gen_cfg.base_seed + 1_000_000)
    for transition in gen_cfg.transitions:
        for i in range(gen_cfg.realizations_per_transition):
            seed = gen_cfg.base_seed + seed_counter
            seed_counter += 1
            # Keep a margin before/after so both regimes plus the
            # transition ramp are all visible within the sequence.
            earliest = gen_cfg.duration_s * 0.25
            latest = gen_cfg.duration_s * 0.6
            transition_start_s = float(timing_rng.uniform(earliest, latest))

            sequence_id = f"{transition.from_scenario}_to_{transition.to_scenario}_{i:03d}"
            df = generate_sequence(
                scenario_by_name=scenario_by_name,
                sequence_id=sequence_id,
                zone_id=gen_cfg.zone_id,
                duration_s=gen_cfg.duration_s,
                sample_interval_s=gen_cfg.sample_interval_s,
                seed=seed,
                density_growth_threshold=gen_cfg.density_growth_threshold,
                transition=transition,
                transition_start_s=transition_start_s,
            )
            sequences.append(df)

    dataset = pd.concat(sequences, ignore_index=True)
    if gen_cfg.row_stride > 1:
        # subsample AFTER per-sequence future labels were computed at the full 1 Hz resolution, so labels stay exact
        keep = dataset.groupby("sequence_id", sort=False).cumcount() % gen_cfg.row_stride == 0
        dataset = dataset[keep].reset_index(drop=True)

    metadata = {
        "dataset_version": gen_cfg.dataset_version,
        "row_stride": gen_cfg.row_stride,
        "split_strategy": gen_cfg.split_strategy,
        "generator_version": GENERATOR_VERSION,
        "data_source": "synthetic",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_seed": gen_cfg.base_seed,
        "duration_s": gen_cfg.duration_s,
        "sample_interval_s": gen_cfg.sample_interval_s,
        "windows_seconds": WINDOWS_SECONDS,
        "density_growth_threshold": gen_cfg.density_growth_threshold,
        "n_sequences": dataset["sequence_id"].nunique(),
        "n_rows": len(dataset),
        "n_pure_sequences": len(gen_cfg.scenarios) * gen_cfg.realizations_per_scenario,
        "n_transition_sequences": len(gen_cfg.transitions) * gen_cfg.realizations_per_transition,
        "scenarios": [s.name for s in gen_cfg.scenarios],
        "transitions": [f"{t.from_scenario}->{t.to_scenario}" for t in gen_cfg.transitions],
        "label_counts": dataset["label"].value_counts().to_dict(),
        "disclaimer": (
            "This dataset is synthetic engineering data generated to develop and "
            "test the CrowdShield risk model. It is NOT real-world crowd-safety "
            "ground truth and does not establish clinically or operationally "
            "validated thresholds."
        ),
    }

    return dataset, metadata


def split_dataset(
    df: pd.DataFrame,
    seed: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> pd.DataFrame:
    """Assign a 'split' column (train/val/test) by whole sequence_id — a
    sequence's rows are NEVER divided across splits, which would leak
    temporally adjacent information between train and evaluation data.
    """
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    sequence_ids = sorted(df["sequence_id"].unique())  # sort first for determinism
    rng = np.random.default_rng(seed)
    shuffled = list(sequence_ids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    # Remainder goes to test, so the three counts always sum to n exactly.
    n_test = n - n_train - n_val

    split_by_id: dict[str, str] = {}
    for sequence_id in shuffled[:n_train]:
        split_by_id[sequence_id] = "train"
    for sequence_id in shuffled[n_train:n_train + n_val]:
        split_by_id[sequence_id] = "val"
    for sequence_id in shuffled[n_train + n_val:]:
        split_by_id[sequence_id] = "test"

    df = df.copy()
    df["split"] = df["sequence_id"].map(split_by_id)
    return df


def split_dataset_stratified(
    df: pd.DataFrame, seed: int, train_frac: float = 0.7, val_frac: float = 0.15, test_frac: float = 0.15,
) -> pd.DataFrame:
    """SEQUENCE-level split stratified by scenario: every scenario contributes at least one whole sequence to
    train, val AND test (needs >= 3 sequences per scenario), so no class or operating state can silently be
    missing from an evaluation split (the original random split left MODERATE absent from test). A sequence's
    rows are never divided across splits. Deterministic for a fixed seed."""
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")
    rng = np.random.default_rng(seed)
    split_by_id: dict[str, str] = {}
    for scenario in sorted(df["scenario"].unique()):
        ids = sorted(df.loc[df["scenario"] == scenario, "sequence_id"].unique())
        if len(ids) < 3:
            raise ValueError(f"stratified split needs >= 3 sequences per scenario; {scenario!r} has {len(ids)}")
        order = list(ids)
        rng.shuffle(order)
        n_val = max(1, int(round(len(order) * val_frac)))
        n_test = max(1, int(round(len(order) * test_frac)))
        for i, sequence_id in enumerate(order):
            split_by_id[sequence_id] = "val" if i < n_val else "test" if i < n_val + n_test else "train"
    out = df.copy()
    out["split"] = out["sequence_id"].map(split_by_id)
    return out


def apply_split(df: pd.DataFrame, gen_cfg: GeneratorConfig, seed: int) -> pd.DataFrame:
    """Split according to `gen_cfg.split_strategy` (legacy files default to the original random split)."""
    fn = split_dataset_stratified if gen_cfg.split_strategy == "stratified" else split_dataset
    return fn(df, seed=seed, train_frac=gen_cfg.train_frac, val_frac=gen_cfg.val_frac, test_frac=gen_cfg.test_frac)


def save_dataset(df: pd.DataFrame, metadata: dict, output_dir: Path | str, compress: bool = False) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / ("synthetic_dataset.csv.gz" if compress else "synthetic_dataset.csv")
    metadata_path = output_dir / "metadata.json"

    df.to_csv(csv_path, index=False)
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return csv_path, metadata_path


def load_dataset(csv_path: Path | str) -> pd.DataFrame:
    return pd.read_csv(csv_path)
