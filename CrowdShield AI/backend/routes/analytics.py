"""Analytics API routes matching frontend analyticsApi.ts contract."""
from __future__ import annotations

from fastapi import APIRouter, Query
from backend.schemas import BottleneckPoint, TimeRange, TimeSeriesPoint
from backend.state import state

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/density-over-time", response_model=list[TimeSeriesPoint])
def get_density_over_time(range: TimeRange = Query("1h")) -> list[TimeSeriesPoint]:
    """Crowd density evolution over the specified time range."""
    return state.get_time_series("density")


@router.get("/movement-speed-distribution", response_model=list[TimeSeriesPoint])
def get_movement_speed_distribution(range: TimeRange = Query("1h")) -> list[TimeSeriesPoint]:
    """Estimated pedestrian movement velocity over time."""
    return state.get_time_series("speed")


@router.get("/direction-variance", response_model=list[TimeSeriesPoint])
def get_direction_variance(range: TimeRange = Query("1h")) -> list[TimeSeriesPoint]:
    """Direction variance (turbulent vs ordered flow) over time."""
    return state.get_time_series("variance")


@router.get("/inflow-rate", response_model=list[TimeSeriesPoint])
def get_inflow_rate(range: TimeRange = Query("1h")) -> list[TimeSeriesPoint]:
    """Rate of people entering monitored zones."""
    return state.get_time_series("inflow")


@router.get("/outflow-rate", response_model=list[TimeSeriesPoint])
def get_outflow_rate(range: TimeRange = Query("1h")) -> list[TimeSeriesPoint]:
    """Rate of people exiting monitored zones."""
    return state.get_time_series("outflow")


@router.get("/bottleneck-pressure", response_model=list[BottleneckPoint])
def get_bottleneck_pressure() -> list[BottleneckPoint]:
    """Current bottleneck pressure across configured gates and chokepoints."""
    return state.get_bottleneck_pressure()
