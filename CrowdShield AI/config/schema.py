"""Typed configuration schema for CrowdShield.

All runtime configuration flows through these models. Nothing else in the
codebase should parse YAML directly — this keeps config validation in one
place and gives every other module IDE-level type safety.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class VideoConfig(BaseModel):
    source: str = Field(..., description="Path or URL to video source")

    # IMPORTANT: target_fps is the PROCESSING/inference target rate, not the
    # physical frame rate of the source video. A source video may be recorded
    # at 30fps while we deliberately process it at a lower target_fps to save
    # compute (frame skipping / sampling). The two are independent settings.
    target_fps: float = Field(
        15.0,
        gt=0,
        le=60,
        description=(
            "Target processing rate in frames/sec. This is NOT the source "
            "video's native FPS — it is the rate at which the pipeline "
            "samples/processes frames, which may be lower than the source."
        ),
    )


class DetectionConfig(BaseModel):
    model: str = Field("yolov8n.pt", description="Ultralytics model name/path")
    confidence: float = Field(0.4, ge=0.0, le=1.0)
    device: Literal["cpu", "cuda", "auto"] = "auto"
    image_size: int = Field(640, gt=0)


class TrackingConfig(BaseModel):
    tracker: Literal["bytetrack", "botsort"] = "bytetrack"
    max_age: int = Field(30, gt=0, description="Frames to keep a lost track alive")


class AnalyticsConfig(BaseModel):
    update_interval_seconds: float = Field(
        1.0, gt=0, description="How often crowd metrics/features are recomputed"
    )
    trajectory_history_seconds: float = Field(15.0, gt=0, le=60)
    feature_history_seconds: float = Field(60.0, gt=0, le=300)

    @field_validator("trajectory_history_seconds")
    @classmethod
    def _bounded_trajectory(cls, v: float) -> float:
        # Hard constraint: never maintain unlimited tracking history.
        if v > 60:
            raise ValueError("trajectory_history_seconds must stay within 5-60s window")
        return v


class RiskConfig(BaseModel):
    model_path: str | None = Field(None, description="Path to trained risk model artifact")
    thresholds: dict[str, float] = Field(
        default_factory=lambda: {"low": 25.0, "moderate": 50.0, "high": 75.0, "critical": 90.0}
    )


class TemporalConfig(BaseModel):
    """Phase F1 rolling-window temporal feature configuration.

    windows_seconds is intentionally constrained to exactly [5, 15, 30, 60]
    for Milestone 1 — the spec is explicit that we do not add more windows
    or arbitrary combinations at this stage. It remains in YAML (rather
    than a hardcoded Python constant) purely for consistency with the
    existing config system, not to invite ad-hoc customization.
    """

    windows_seconds: list[int] = Field(default_factory=lambda: [5, 15, 30, 60])
    density_growth_threshold: float = Field(
        0.05,
        gt=0,
        description=(
            "Engineering trigger for surge_flag: the 5s window's density "
            "regression slope (people/m²/s) must exceed this to flag a "
            "surge. This is a prototype engineering threshold, NOT a "
            "medically or physically validated crowd-safety threshold."
        ),
    )

    @field_validator("windows_seconds")
    @classmethod
    def _exact_windows(cls, v: list[int]) -> list[int]:
        if v != [5, 15, 30, 60]:
            raise ValueError(
                "windows_seconds must be exactly [5, 15, 30, 60] for Milestone 1 (F1) "
                "— no additional or arbitrary window combinations"
            )
        return v


class ZoneConfig(BaseModel):
    id: str
    name: str
    # Coordinate convention (D1): image PIXEL coordinates, origin (0, 0) at
    # the frame's top-left, x increases rightward, y increases downward —
    # matches Detection/Track's bounding box convention (see
    # vision/detector.py, vision/tracker.py). NOT normalized (0-1)
    # coordinates. Perspective/real-world calibration is explicitly
    # deferred to a later phase.
    polygon: list[tuple[float, float]] = Field(
        ..., min_length=3, description="Zone boundary as (x, y) pixel coordinates"
    )
    area_m2: float = Field(..., gt=0, description="Real-world area for density calculation (used from Phase E onward)")
    enabled: bool = Field(True, description="Disabled zones are ignored during zone assignment")

    # --- Phase K1: zone graph fields (decision-support only; unused by
    # A1-J1's detection/tracking/metrics/risk pipeline) ---
    capacity: float | None = Field(
        None, gt=0,
        description=(
            "Maximum safe occupancy (people) for this zone. None means "
            "capacity is not configured for this zone — K1 cannot use "
            "capacity-based rejection/scoring for it without this set."
        ),
    )
    preferred_utilization_threshold: float | None = Field(
        None, gt=0, le=1.0,
        description=(
            "Fraction of capacity (0-1) above which this zone is considered "
            "'above safe utilization' by K1's destination filtering, even if "
            "not yet at hard capacity. E.g. 0.85 rejects a destination once "
            "it's 85% full. None means only hard capacity is enforced."
        ),
    )
    neighbors: list[str] = Field(
        default_factory=list,
        description=(
            "Zone IDs explicitly reachable from this zone, for K1's diversion "
            "decision engine. Deliberately NOT inferred from polygon "
            "adjacency/intersection — a venue operator must explicitly "
            "declare real-world connectivity. This is a PER-ZONE (directed) "
            "list, not a shared undirected edge — a one-way relationship is "
            "supported simply by omitting the reverse direction on the "
            "other zone (see decision/zone_graph.py)."
        ),
    )
    neighbor_costs: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Optional per-neighbor routing cost (e.g. relative distance or "
            "difficulty), keyed by neighbor zone ID. A neighbor not present "
            "here defaults to cost 1.0 (uniform) in K1's scoring — see "
            "decision/diversion_engine.py. Purely an engineering weighting "
            "input, not a real-world distance measurement unless the venue "
            "operator supplies one."
        ),
    )


class VenueConfig(BaseModel):
    name: str
    zones: list[ZoneConfig] = Field(default_factory=list)

    @field_validator("zones")
    @classmethod
    def _unique_zone_ids(cls, zones: list[ZoneConfig]) -> list[ZoneConfig]:
        ids = [z.id for z in zones]
        seen = set()
        duplicates = set()
        for zone_id in ids:
            if zone_id in seen:
                duplicates.add(zone_id)
            seen.add(zone_id)
        if duplicates:
            raise ValueError(f"Duplicate zone id(s) in venue config: {sorted(duplicates)}")
        return zones

    @model_validator(mode="after")
    def _neighbors_reference_valid_zones(self) -> "VenueConfig":
        zone_ids = {z.id for z in self.zones}
        for zone in self.zones:
            unknown = set(zone.neighbors) - zone_ids
            if unknown:
                raise ValueError(
                    f"Zone {zone.id!r} lists unknown neighbor(s) {sorted(unknown)} — "
                    f"neighbors must reference other zone IDs defined in this venue"
                )
            if zone.id in zone.neighbors:
                raise ValueError(f"Zone {zone.id!r} lists itself as its own neighbor")
            unknown_costs = set(zone.neighbor_costs) - set(zone.neighbors)
            if unknown_costs:
                raise ValueError(
                    f"Zone {zone.id!r} has neighbor_costs for {sorted(unknown_costs)}, "
                    f"which are not in its own neighbors list"
                )
        return self


class DiversionConfig(BaseModel):
    """Phase K1 crowd-diversion decision-engine configuration.

    Every threshold/weight here is an explicit ENGINEERING POLICY DECISION
    for this prototype, not a validated real-world evacuation standard —
    see README "Phase K1" for the full disclaimer.
    """

    # --- source-zone trigger thresholds ---
    source_current_risk_threshold: str = Field(
        "HIGH", description="Minimum current risk label (ordinal) that triggers considering a zone a diversion source"
    )
    source_future_risk_threshold: str = Field(
        "HIGH", description="Minimum future (+5m or +10m) risk label (ordinal) that also triggers diversion consideration"
    )
    source_surge_triggers: bool = Field(
        True, description="Whether an active surge_flag alone (regardless of current risk label) can trigger diversion consideration"
    )
    source_bottleneck_pressure_threshold: float | None = Field(
        None, description="If set, a zone's 60s bottleneck_pressure at/above this value also triggers diversion consideration"
    )

    source_capacity_trigger_utilization: float | None = Field(
        1.0, gt=0,
        description=(
            "PROVISIONAL, model-independent rule: a zone whose people_count >= this fraction of its CONFIGURED capacity is a diversion "
            "source regardless of any model output (so a dense stationary crowd cannot be hidden by undefined movement metrics). "
            "1.0 = 'at/over capacity', the same line K1 already uses to refuse a destination - no new emergency threshold is invented. "
            "None disables the rule (previous behaviour). Not validated against real incidents."
        ),
    )
    destination_requires_reliable_inputs: bool = Field(
        True, description="Reject a destination whose model inputs are out-of-distribution (its predicted risk cannot be trusted as evidence of safety)"
    )

    # --- destination rejection thresholds ---
    destination_max_current_risk: str = Field(
        "MODERATE", description="A destination with current risk ABOVE this ordinal label is rejected outright"
    )
    destination_max_future_risk: str = Field(
        "MODERATE", description="A destination with future (+5m or +10m) risk ABOVE this ordinal label is rejected outright"
    )

    # --- scoring weights (normal mode) ---
    # Priority order encoded by weight magnitude, not tuned/optimized —
    # documented engineering assignment: risk >> capacity > congestion >
    # flow >= routing.
    weight_risk: float = Field(10.0, ge=0)
    weight_capacity: float = Field(3.0, ge=0)
    weight_congestion: float = Field(2.0, ge=0)
    weight_flow: float = Field(1.5, ge=0)
    weight_routing: float = Field(1.0, ge=0)

    # --- scoring weights (emergency mode, source is CRITICAL) ---
    # Skews further toward risk avoidance and away from routing
    # convenience once a source zone is CRITICAL — still subject to the
    # SAME hard safety rejection filters above (emergency mode never
    # relaxes safety rejection, only re-prioritizes among already-safe
    # candidates).
    emergency_weight_risk: float = Field(20.0, ge=0)
    emergency_weight_capacity: float = Field(4.0, ge=0)
    emergency_weight_congestion: float = Field(1.0, ge=0)
    emergency_weight_flow: float = Field(1.0, ge=0)
    emergency_weight_routing: float = Field(0.5, ge=0)

    max_rejected_candidates_reported: int = Field(10, ge=0)

    @field_validator(
        "source_current_risk_threshold", "source_future_risk_threshold",
        "destination_max_current_risk", "destination_max_future_risk",
    )
    @classmethod
    def _valid_label(cls, v: str) -> str:
        valid = {"SAFE", "LOW", "MODERATE", "HIGH", "CRITICAL"}
        if v not in valid:
            raise ValueError(f"{v!r} is not a valid risk label; must be one of {sorted(valid)}")
        return v


class InferenceConfig(BaseModel):
    """Phase L1 end-to-end inference pipeline settings (paths are relative
    to the repository root unless absolute)."""

    model_dir: str = Field("models/artifacts", description="Directory holding the I1 future_risk_*.pkl bundle (and H1's risk_model.pkl fallback)")
    reference_dataset: str | None = Field(
        "sample_data/synthetic_long/synthetic_dataset.csv",
        description="Training dataset used ONLY to build J1's reference ('typical value') profile; None or a missing file disables explanations",
    )
    explain_mode: Literal["off", "alerts", "all"] = Field(
        "alerts",
        description="off: no explanations; alerts: only zones that are diversion sources, changed predicted class, or are HIGH/CRITICAL; all: every zone every frame",
    )
    explain_top_k: int = Field(3, ge=1, le=10)
    min_pattern_support_rows: int = Field(
        30, ge=1,
        description="A set of undefined model inputs counts as SUPPORTED only if at least this many training rows had exactly that set undefined",
    )
    max_consecutive_failures: int = Field(
        5, ge=1, description="Consecutive detection/tracking failures tolerated before the pipeline aborts"
    )


class AppConfig(BaseModel):
    """Top-level config object. This is what every module receives."""

    model_config = ConfigDict(frozen=True)  # immutable once loaded — safe in a hot loop

    video: VideoConfig
    detection: DetectionConfig
    tracking: TrackingConfig
    analytics: AnalyticsConfig
    risk: RiskConfig
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    diversion: DiversionConfig = Field(default_factory=DiversionConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    venue: VenueConfig

    random_seed: int = Field(42, description="Global seed for reproducibility")
    config_version: str = Field("0.1.0")

    @model_validator(mode="after")
    def _feature_history_covers_max_window(self) -> "AppConfig":
        # F1's longest rolling window (60s) must fit within the bounded
        # history buffer analytics.feature_history_seconds already
        # defines — this is the same buffer F1's ZoneTemporalTracker uses,
        # not a duplicate concept.
        max_window = max(self.temporal.windows_seconds)
        if self.analytics.feature_history_seconds < max_window:
            raise ValueError(
                f"analytics.feature_history_seconds ({self.analytics.feature_history_seconds}) "
                f"must be >= the largest temporal.windows_seconds value ({max_window})"
            )
        return self
