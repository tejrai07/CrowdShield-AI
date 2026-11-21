"""Dashboard API routes matching frontend dashboardApi.ts contract."""
from __future__ import annotations

from fastapi import APIRouter
from backend.schemas import (
    DashboardSummary,
    FlowDirectionSlice,
    OperationalEvent,
    TimeSeriesPoint,
    ZoneOccupancySlice,
)
from backend.state import state

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/summary", response_model=DashboardSummary)
def get_dashboard_summary() -> DashboardSummary:
    """Venue-wide summary metrics for crowd status, counts, and prediction confidence."""
    return state.get_dashboard_summary()


@router.get("/crowd-density-trend", response_model=list[TimeSeriesPoint])
def get_crowd_density_trend() -> list[TimeSeriesPoint]:
    """Historical crowd density rolling time-series points."""
    return state.get_time_series("density")


@router.get("/risk-level-trend", response_model=list[TimeSeriesPoint])
def get_risk_level_trend() -> list[TimeSeriesPoint]:
    """Historical risk level / score trend."""
    return state.get_time_series("risk")


@router.get("/zone-occupancy-distribution", response_model=list[ZoneOccupancySlice])
def get_zone_occupancy_distribution() -> list[ZoneOccupancySlice]:
    """Zone occupancy distribution breakdown."""
    return state.get_zone_occupancy_distribution()


@router.get("/flow-direction-distribution", response_model=list[FlowDirectionSlice])
def get_flow_direction_distribution() -> list[FlowDirectionSlice]:
    """Cardinal flow direction distribution across active zones."""
    return state.get_flow_directions()


@router.get("/recent-events", response_model=list[OperationalEvent])
def get_recent_events() -> list[OperationalEvent]:
    """Recent operational events and surge detections."""
    return state.get_events()
