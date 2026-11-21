"""In-memory state management with thread-safe ring buffers for historical metrics."""
from __future__ import annotations

import collections
from datetime import datetime, timezone, timedelta
import threading
from typing import Any

from backend.schemas import (
    Alert,
    BottleneckPoint,
    DashboardSummary,
    DiversionRecommendation,
    FeatureContribution,
    FlowDirectionSlice,
    OperationalEvent,
    PipelineStatusResponse,
    RiskForecastPoint,
    TimeSeriesPoint,
    Zone,
    ZoneOccupancySlice,
)
from backend.translation import (
    _format_timestamp,
    generate_alerts_and_events,
    translate_bottleneck_pressure,
    translate_dashboard_summary,
    translate_feature_contributions,
    translate_flow_directions,
    translate_forecast,
    translate_zones,
)


def _create_initial_l1_frame() -> dict[str, Any]:
    """Create a valid initial L1 frame so the backend serves sane data immediately on startup."""
    return {
        "schema_version": "l1.0",
        "timestamp_s": 0.0,
        "frame_id": 1,
        "pipeline_status": "ok",
        "model_data_notice": (
            "Risk models were trained ONLY on synthetic data (Phase G1) and have not been validated on real crowd incidents. "
            "Future-risk (+5m/+10m) predictions have low escalation recall (~12.5% at +5m, ~10.7% at +10m). "
            "null means unavailable, never zero or safe."
        ),
        "capabilities": {"future_5m": True, "future_10m": True, "explanations": True},
        "zones": [
            {
                "zone_id": "zone_a",
                "name": "Main Concourse",
                "enabled": True,
                "status": "ok",
                "capacity": 500,
                "people_count": 142,
                "density": 1.42,
                "utilization": 0.28,
                "metrics": {
                    "avg_speed_px_s": 42.5,
                    "speed_variance": 5.2,
                    "flow_direction_deg": 45.0,
                    "direction_variance": 0.12,
                    "inflow_count": 18,
                    "outflow_count": 14,
                },
                "temporal": {
                    "surge_flag": False,
                    "bottleneck_pressure_60s": 0.28,
                    "net_flow_rate_60s": 4.0,
                },
                "risk": {
                    "current": {
                        "status": "ok",
                        "risk_class": "LOW",
                        "risk_score": 24.5,
                        "dangerous_probability": 0.12,
                        "horizon_seconds": 0,
                    },
                    "future_5m": {
                        "status": "ok",
                        "risk_class": "LOW",
                        "risk_score": 28.0,
                        "dangerous_probability": 0.15,
                        "horizon_seconds": 300,
                    },
                    "future_10m": {
                        "status": "ok",
                        "risk_class": "MODERATE",
                        "risk_score": 42.0,
                        "dangerous_probability": 0.26,
                        "horizon_seconds": 600,
                    },
                },
                "input_quality": {"complete": True, "missing_feature_count": 0},
            },
            {
                "zone_id": "zone_b",
                "name": "North Gate Turnstiles",
                "enabled": True,
                "status": "ok",
                "capacity": 300,
                "people_count": 185,
                "density": 2.65,
                "utilization": 0.62,
                "metrics": {
                    "avg_speed_px_s": 22.1,
                    "speed_variance": 8.4,
                    "flow_direction_deg": 80.0,
                    "direction_variance": 0.35,
                    "inflow_count": 29,
                    "outflow_count": 12,
                },
                "temporal": {
                    "surge_flag": True,
                    "bottleneck_pressure_60s": 0.62,
                    "net_flow_rate_60s": 17.0,
                },
                "risk": {
                    "current": {
                        "status": "ok",
                        "risk_class": "MODERATE",
                        "risk_score": 58.2,
                        "dangerous_probability": 0.44,
                        "horizon_seconds": 0,
                    },
                    "future_5m": {
                        "status": "ok",
                        "risk_class": "HIGH",
                        "risk_score": 72.0,
                        "dangerous_probability": 0.68,
                        "horizon_seconds": 300,
                    },
                    "future_10m": {
                        "status": "ok",
                        "risk_class": "HIGH",
                        "risk_score": 75.0,
                        "dangerous_probability": 0.71,
                        "horizon_seconds": 600,
                    },
                },
                "input_quality": {"complete": True, "missing_feature_count": 0},
            },
            {
                "zone_id": "zone_c",
                "name": "Transit Hub Stairwell",
                "enabled": True,
                "status": "ok",
                "capacity": 250,
                "people_count": 65,
                "density": 0.95,
                "utilization": 0.26,
                "metrics": {
                    "avg_speed_px_s": 38.0,
                    "speed_variance": 4.1,
                    "flow_direction_deg": 180.0,
                    "direction_variance": 0.08,
                    "inflow_count": 10,
                    "outflow_count": 12,
                },
                "temporal": {
                    "surge_flag": False,
                    "bottleneck_pressure_60s": 0.18,
                    "net_flow_rate_60s": -2.0,
                },
                "risk": {
                    "current": {
                        "status": "ok",
                        "risk_class": "SAFE",
                        "risk_score": 12.0,
                        "dangerous_probability": 0.05,
                        "horizon_seconds": 0,
                    },
                    "future_5m": {
                        "status": "ok",
                        "risk_class": "SAFE",
                        "risk_score": 15.0,
                        "dangerous_probability": 0.07,
                        "horizon_seconds": 300,
                    },
                    "future_10m": {
                        "status": "ok",
                        "risk_class": "SAFE",
                        "risk_score": 18.0,
                        "dangerous_probability": 0.08,
                        "horizon_seconds": 600,
                    },
                },
                "input_quality": {"complete": True, "missing_feature_count": 0},
            },
        ],
        "explanations": {
            "zone_b": {
                "current": {
                    "status": "ok",
                    "predicted_class": "MODERATE",
                    "predicted_probability": 0.44,
                    "risk_score": 58.2,
                    "top_positive_features": [
                        {
                            "feature": "inflow_outflow_imbalance",
                            "human_name": "Inflow Imbalance",
                            "contribution": 0.32,
                        },
                        {
                            "feature": "bottleneck_pressure",
                            "human_name": "Bottleneck Pressure",
                            "contribution": 0.24,
                        },
                    ],
                    "top_negative_features": [
                        {
                            "feature": "speed_consistency",
                            "human_name": "Speed Consistency",
                            "contribution": -0.10,
                        }
                    ],
                }
            }
        },
        "diversion_recommendations": [
            {
                "source_zone": "zone_b",
                "destination_zone": "zone_c",
                "status": "RECOMMENDED",
                "urgency": "MODERATE",
                "confidence": 0.82,
                "reason": "North Gate Turnstiles bottleneck pressure rising (62%). Transit Hub Stairwell has ample capacity (26%).",
                "source_inputs_complete": True,
            }
        ],
        "warnings": [],
        "incomplete_input_zones": [],
    }


class BackendState:
    """Thread-safe store for current L1 frame and rolling metric histories."""

    def __init__(self, max_history: int = 120) -> None:
        self._lock = threading.RLock()
        self.max_history = max_history
        self.base_time = datetime.now(timezone.utc) - timedelta(minutes=60)
        self.current_frame: dict[str, Any] = _create_initial_l1_frame()

        # Ring buffers for time series: (iso_timestamp, float_value)
        self.history_density: collections.deque[tuple[str, float]] = collections.deque(maxlen=max_history)
        self.history_risk: collections.deque[tuple[str, float]] = collections.deque(maxlen=max_history)
        self.history_speed: collections.deque[tuple[str, float]] = collections.deque(maxlen=max_history)
        self.history_variance: collections.deque[tuple[str, float]] = collections.deque(maxlen=max_history)
        self.history_inflow: collections.deque[tuple[str, float]] = collections.deque(maxlen=max_history)
        self.history_outflow: collections.deque[tuple[str, float]] = collections.deque(maxlen=max_history)

        self.alerts: list[Alert] = []
        self.events: list[OperationalEvent] = []

        # Seed initial history points
        self._seed_history()

    def _seed_history(self) -> None:
        """Seed rolling histories with reasonable initial points so charts render on first load."""
        now = datetime.now(timezone.utc)
        for i in range(24, 0, -1):
            ts = (now - timedelta(minutes=i * 2)).strftime("%Y-%m-%dT%H:%M:%SZ")
            self.history_density.append((ts, round(1.2 + (i % 5) * 0.15, 2)))
            self.history_risk.append((ts, round(25.0 + (i % 7) * 4.5, 1)))
            self.history_speed.append((ts, round(1.1 + (i % 3) * 0.08, 2)))
            self.history_variance.append((ts, round(0.18 + (i % 4) * 0.04, 2)))
            self.history_inflow.append((ts, round(15.0 + (i % 6) * 3.0, 1)))
            self.history_outflow.append((ts, round(12.0 + (i % 5) * 2.5, 1)))

        self.alerts, self.events = generate_alerts_and_events(self.current_frame, self.base_time)

    def update_frame(self, frame: dict[str, Any]) -> None:
        """Push a newly processed frame into state and append to rolling histories."""
        with self._lock:
            self.current_frame = frame
            ts_iso = _format_timestamp(frame.get("timestamp_s"), self.base_time)

            # Calculate venue averages
            raw_zones = frame.get("zones") or []
            densities = [float(z["density"]) for z in raw_zones if z.get("density") is not None]
            avg_density = sum(densities) / len(densities) if densities else 0.0

            risk_scores: list[float] = []
            speeds: list[float] = []
            variances: list[float] = []
            inflows: list[float] = []
            outflows: list[float] = []

            for z in raw_zones:
                curr_risk = ((z.get("risk") or {}).get("current") or {}).get("risk_score")
                if curr_risk is not None:
                    risk_scores.append(float(curr_risk))

                metrics = z.get("metrics") or {}
                if metrics.get("avg_speed_px_s") is not None:
                    speeds.append(float(metrics["avg_speed_px_s"]) / 30.0)  # scale px/s to ~m/s
                if metrics.get("direction_variance") is not None:
                    variances.append(float(metrics["direction_variance"]))
                if metrics.get("inflow_count") is not None:
                    inflows.append(float(metrics["inflow_count"]))
                if metrics.get("outflow_count") is not None:
                    outflows.append(float(metrics["outflow_count"]))

            venue_risk_score = max(risk_scores) if risk_scores else 0.0
            avg_speed = sum(speeds) / len(speeds) if speeds else 1.0
            avg_var = sum(variances) / len(variances) if variances else 0.2
            total_inflow = sum(inflows)
            total_outflow = sum(outflows)

            self.history_density.append((ts_iso, round(avg_density, 2)))
            self.history_risk.append((ts_iso, round(venue_risk_score, 1)))
            self.history_speed.append((ts_iso, round(avg_speed, 2)))
            self.history_variance.append((ts_iso, round(avg_var, 2)))
            self.history_inflow.append((ts_iso, round(total_inflow, 1)))
            self.history_outflow.append((ts_iso, round(total_outflow, 1)))

            # Update alerts & events
            new_alerts, new_events = generate_alerts_and_events(frame, self.base_time)
            self.alerts = new_alerts
            if new_events:
                self.events = (new_events + self.events)[:20]

    def get_zones(self) -> list[Zone]:
        with self._lock:
            return translate_zones(self.current_frame, self.base_time)

    def get_zone_by_id(self, zone_id: str) -> Zone | None:
        with self._lock:
            for z in translate_zones(self.current_frame, self.base_time):
                if z.id == zone_id:
                    return z
            return None

    def get_dashboard_summary(self) -> DashboardSummary:
        with self._lock:
            return translate_dashboard_summary(self.current_frame, self.alerts, self.base_time)

    def get_forecast(self) -> list[RiskForecastPoint]:
        with self._lock:
            return translate_forecast(self.current_frame, self.base_time)

    def get_feature_contributions(self) -> list[FeatureContribution]:
        with self._lock:
            return translate_feature_contributions(self.current_frame)

    def get_flow_directions(self) -> list[FlowDirectionSlice]:
        with self._lock:
            return translate_flow_directions(self.current_frame)

    def get_zone_occupancy_distribution(self) -> list[ZoneOccupancySlice]:
        with self._lock:
            zones = translate_zones(self.current_frame, self.base_time)
            return [ZoneOccupancySlice(zoneName=z.name, occupancy=z.occupancy) for z in zones]

    def get_bottleneck_pressure(self) -> list[BottleneckPoint]:
        with self._lock:
            return translate_bottleneck_pressure(self.current_frame)

    def get_alerts(self) -> list[Alert]:
        with self._lock:
            return list(self.alerts)

    def get_events(self) -> list[OperationalEvent]:
        with self._lock:
            return list(self.events)

    def get_time_series(self, metric: str) -> list[TimeSeriesPoint]:
        with self._lock:
            buf_map = {
                "density": self.history_density,
                "risk": self.history_risk,
                "speed": self.history_speed,
                "variance": self.history_variance,
                "inflow": self.history_inflow,
                "outflow": self.history_outflow,
            }
            buf = buf_map.get(metric, self.history_density)
            return [TimeSeriesPoint(timestamp=ts, value=val) for ts, val in buf]

    def get_pipeline_status(self) -> PipelineStatusResponse:
        with self._lock:
            return PipelineStatusResponse(
                pipeline_status=self.current_frame.get("pipeline_status", "ok"),
                schema_version=self.current_frame.get("schema_version", "l1.0"),
                model_data_notice=self.current_frame.get("model_data_notice", ""),
                frame_id=int(self.current_frame.get("frame_id", 0)),
                timestamp_s=self.current_frame.get("timestamp_s"),
                warnings=list(self.current_frame.get("warnings") or []),
                incomplete_input_zones=list(self.current_frame.get("incomplete_input_zones") or []),
                capabilities=dict(self.current_frame.get("capabilities") or {}),
            )

    def get_diversion_recommendations(self) -> list[DiversionRecommendation]:
        with self._lock:
            recs = self.current_frame.get("diversion_recommendations") or []
            return [
                DiversionRecommendation(
                    source_zone=r.get("source_zone", "unknown"),
                    destination_zone=r.get("destination_zone"),
                    status=r.get("status", "NO_SAFE_DESTINATION"),
                    urgency=r.get("urgency", "LOW"),
                    reason=r.get("reason", ""),
                    confidence=r.get("confidence"),
                    source_inputs_complete=bool(r.get("source_inputs_complete", True)),
                )
                for r in recs
            ]


# Global singleton instance
state = BackendState()
