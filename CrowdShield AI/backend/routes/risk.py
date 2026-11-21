"""Risk API routes matching frontend riskApi.ts contract."""
from __future__ import annotations

from fastapi import APIRouter
from backend.schemas import FeatureContribution, RiskForecastPoint, TimeSeriesPoint
from backend.state import state

router = APIRouter(prefix="/risk", tags=["Risk"])


@router.get("/forecast", response_model=list[RiskForecastPoint])
def get_risk_forecast() -> list[RiskForecastPoint]:
    """Multi-horizon risk predictions: CURRENT, +5min, and +10min."""
    return state.get_forecast()


@router.get("/timeline", response_model=list[TimeSeriesPoint])
def get_risk_timeline() -> list[TimeSeriesPoint]:
    """Historical timeline of venue risk score evolution."""
    return state.get_time_series("risk")


@router.get("/feature-contributions", response_model=list[FeatureContribution])
def get_feature_contributions() -> list[FeatureContribution]:
    """Local occlusion/attribution feature importance for crowd risk predictions."""
    return state.get_feature_contributions()
