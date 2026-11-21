"""Pydantic schemas aligning with frontend domain types (src/types/domain.ts)
and exposing ML pipeline honesty metadata."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field

# Domain enums
RiskLevel = Literal["SAFE", "LOW", "MODERATE", "HIGH", "CRITICAL"]
AlertSeverity = Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
AlertStatus = Literal["ACTIVE", "ACKNOWLEDGED", "RESOLVED"]
ZoneStatus = Literal["NORMAL", "ELEVATED", "RESTRICTED", "OFFLINE"]
FlowDirection = Literal["NORTH", "SOUTH", "EAST", "WEST"]
TimeHorizon = Literal["CURRENT", "+5min", "+10min"]
TimeRange = Literal["1h", "6h", "24h", "custom"]


class Zone(BaseModel):
    id: str
    name: str
    density: float = Field(..., description="People per m²")
    occupancy: float = Field(..., description="Occupancy percentage (0-100)")
    capacity: int
    riskLevel: RiskLevel
    status: ZoneStatus
    updatedAt: str = Field(..., description="ISO 8601 timestamp")


class TimeSeriesPoint(BaseModel):
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    value: float


class RiskForecastPoint(BaseModel):
    horizon: TimeHorizon
    timestamp: str
    riskLevel: RiskLevel
    riskScore: float = Field(..., description="Score 0-100")
    confidence: float = Field(..., description="Confidence 0-100")


class FeatureContribution(BaseModel):
    feature: str
    contribution: float = Field(..., description="Signed contribution -100 to 100")
    description: str


class OperationalEvent(BaseModel):
    id: str
    timestamp: str
    zone: str
    type: str
    description: str
    severity: AlertSeverity


class Alert(BaseModel):
    id: str
    severity: AlertSeverity
    timestamp: str
    zone: str
    triggerReason: str
    status: AlertStatus


class DashboardSummary(BaseModel):
    crowdStatus: RiskLevel
    activePeopleCount: int
    currentRiskLevel: RiskLevel
    activeZoneCount: int
    totalZoneCount: int
    alertCount: int
    predictionConfidence: float = Field(..., description="Confidence 0-100")


class ZoneOccupancySlice(BaseModel):
    zoneName: str
    occupancy: float


class FlowDirectionSlice(BaseModel):
    direction: FlowDirection
    percentage: float


class BottleneckPoint(BaseModel):
    location: str
    pressure: float = Field(..., description="Pressure 0-100")


# ML Honesty and Pipeline Management Schemas
class PipelineStatusResponse(BaseModel):
    pipeline_status: str
    schema_version: str
    model_data_notice: str
    frame_id: int
    timestamp_s: float | None = None
    warnings: list[str] = Field(default_factory=list)
    incomplete_input_zones: list[str] = Field(default_factory=list)
    capabilities: dict[str, Any] = Field(default_factory=dict)


class DiversionRecommendation(BaseModel):
    source_zone: str
    destination_zone: str | None = None
    status: str
    urgency: str
    reason: str
    confidence: float | None = None
    source_inputs_complete: bool = True
