"""Alerts API routes matching frontend alertsApi.ts contract."""
from __future__ import annotations

from fastapi import APIRouter
from backend.schemas import Alert
from backend.state import state

router = APIRouter(prefix="/alerts", tags=["Alerts"])


@router.get("", response_model=list[Alert])
def get_alerts() -> list[Alert]:
    """Active operational alerts, surge warnings, and movement-incomplete notifications."""
    return state.get_alerts()
