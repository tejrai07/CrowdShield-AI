"""Typed schemas for Phase G1 synthetic dataset generation.

Mirrors the existing `config/schema.py` pattern (pydantic + YAML) rather
than introducing a second configuration system. These schemas are
G1-specific and deliberately kept separate from `config/schema.py`'s
`AppConfig` — scenario-generation parameters describe how to build a
training dataset, not the real-time pipeline's runtime configuration, so
they don't belong in the same config object.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

RiskLabel = Literal["SAFE", "LOW", "MODERATE", "HIGH", "CRITICAL"]

# Exactly the six scenarios defined for G1. Intentionally a fixed tuple,
# not open-ended — new scenarios are a deliberate future decision, not
# something a config typo should silently introduce.
# The six ORIGINAL G1 scenarios: every scenario file must define these (legacy contract, unchanged).
CORE_SCENARIO_NAMES: tuple[str, ...] = (
    "NORMAL_STABLE",
    "GRADUAL_BUILDUP",
    "RAPID_SURGE",
    "BOTTLENECK",
    "CONFLICTING_FLOW",
    "CRITICAL_ESCALATION",
)
# Reliability-fix scenarios (data/scenarios_v2.yaml): operating states the inference pipeline can produce but the
# original six never covered - empty zones, stationary crowds (valid 0.0 speed, undefined direction), danger driven
# by density rather than speed, slow-but-safe movement, and observation dropout.
EXTENDED_SCENARIO_NAMES: tuple[str, ...] = (
    "EMPTY_ZONE",
    "SINGLE_STATIONARY_PERSON",
    "STATIONARY_LOW_OCCUPANCY",
    "STATIONARY_MODERATE_DENSITY",
    "STATIONARY_ELEVATED_DENSITY",
    "STATIONARY_DENSE_HIGH",
    "STATIONARY_DENSE_CRITICAL",
    "SPARSE_MOVING",
    "MOVING_DENSE_HIGH",
    "MOVING_DENSE_CRITICAL",
    "SLOW_WALK_SAFE",
    "DROPOUT_MOVING_SAFE",
    "DROPOUT_DENSE_HIGH",
)
SCENARIO_NAMES: tuple[str, ...] = CORE_SCENARIO_NAMES + EXTENDED_SCENARIO_NAMES  # all ALLOWED names


class ScenarioConfig(BaseModel):
    """Engineering parameters for one synthetic scenario.

    Every value here is an explicit, documented engineering assumption
    used to generate plausible, controlled, reproducible synthetic
    data — NONE of these are validated real-world crowd-safety
    thresholds. See README "Phase G1" section for the full disclaimer.
    """

    name: str
    label: RiskLabel
    area_m2: float = Field(50.0, gt=0, description="Fixed synthetic zone area for this scenario")

    base_density: float = Field(..., ge=0, description="people/m^2 at t=0 (pre-noise)")
    density_trend: float = Field(..., description="people/m^2 per second (pre-noise linear drift)")

    base_speed_px_s: float = Field(..., ge=0, description="pixels/second at t=0 (pre-noise)")
    speed_trend_px_s2: float = Field(..., description="pixels/s^2 (pre-noise linear drift)")

    base_direction_deg: float = Field(90.0, ge=0, lt=360)
    base_direction_variance: float = Field(..., ge=0, le=1)
    direction_variance_trend: float = Field(..., description="1/s (pre-noise linear drift)")

    inflow_rate: float = Field(..., ge=0, description="people/second, pre-noise baseline")
    outflow_rate: float = Field(..., ge=0, description="people/second, pre-noise baseline")

    noise_level: float = Field(..., ge=0, le=1, description="relative stochastic noise scale")

    # --- reliability-fix fields (all default to the legacy behaviour; legacy scenario files are unaffected) ---
    max_density: float | None = Field(None, gt=0, description="plateau ceiling for density (people/m^2): trends stop here instead of extrapolating forever")
    min_speed_px_s: float | None = Field(None, ge=0, description="plateau floor for speed (px/s) for trends that decay")
    movement_mode: Literal["moving", "stationary"] = Field(
        "moving",
        description="stationary reproduces E1 exactly for people who do not move: speed 0.0 and speed variance 0.0 are VALID zeros, "
                    "direction and direction variance are UNDEFINED (None). With zero people ALL movement metrics are None in either mode.")
    metric_dropout_prob: float = Field(0.0, ge=0, le=1, description="per-step probability that speed/direction metrics are undefined although people are present (observation dropout)")

    @field_validator("name")
    @classmethod
    def _known_scenario(cls, v: str) -> str:
        if v not in SCENARIO_NAMES:
            raise ValueError(f"Unknown scenario name {v!r}; must be one of {SCENARIO_NAMES}")
        return v


class TransitionConfig(BaseModel):
    """A scripted transition from one scenario's regime into another
    within a single synthetic sequence — see README "Scenario
    Transitions" for the exact blending/labeling rule."""

    from_scenario: str
    to_scenario: str
    transition_duration_s: float = Field(15.0, gt=0)

    @field_validator("from_scenario", "to_scenario")
    @classmethod
    def _known_scenario(cls, v: str) -> str:
        if v not in SCENARIO_NAMES:
            raise ValueError(f"Unknown scenario name {v!r}; must be one of {SCENARIO_NAMES}")
        return v


class GeneratorConfig(BaseModel):
    """Top-level synthetic dataset generation configuration."""

    duration_s: float = Field(120.0, gt=0)
    sample_interval_s: float = Field(1.0, gt=0)
    zone_id: str = Field("synthetic_zone")
    realizations_per_scenario: int = Field(8, ge=1)
    realizations_per_transition: int = Field(6, ge=1)
    base_seed: int = Field(42)
    density_growth_threshold: float = Field(
        0.05, gt=0, description="Must match temporal.density_growth_threshold semantics from config/schema.py"
    )

    scenarios: list[ScenarioConfig]
    transitions: list[TransitionConfig] = Field(default_factory=list)

    row_stride: int = Field(1, ge=1, description="keep every Nth 1 Hz row AFTER future labels are computed (shrinks the file, keeps labels correct)")
    split_strategy: Literal["random", "stratified"] = Field("random", description="stratified = every scenario contributes sequences to train, val AND test")
    dataset_version: str = Field("synthetic_v1")
    train_frac: float = Field(0.7, gt=0, lt=1)
    val_frac: float = Field(0.15, gt=0, lt=1)
    test_frac: float = Field(0.15, gt=0, lt=1)

    @field_validator("scenarios")
    @classmethod
    def _all_six_scenarios_present(cls, v: list[ScenarioConfig]) -> list[ScenarioConfig]:
        names = {s.name for s in v}
        missing = set(CORE_SCENARIO_NAMES) - names
        if missing:
            raise ValueError(f"Missing scenario definitions: {sorted(missing)}")
        duplicates = len(v) - len(names)
        if duplicates:
            raise ValueError("Duplicate scenario names are not allowed")
        return v

    @model_validator(mode="after")
    def _fractions_sum_to_one(self) -> "GeneratorConfig":
        total = self.train_frac + self.val_frac + self.test_frac
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train_frac + val_frac + test_frac must sum to 1.0 (got {total})")
        return self
