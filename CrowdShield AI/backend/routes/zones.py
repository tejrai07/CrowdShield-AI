"""Zone API routes matching frontend zoneApi.ts contract."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from backend.schemas import Zone
from backend.state import state

router = APIRouter(prefix="/zones", tags=["Zones"])


@router.get("", response_model=list[Zone])
def get_zones() -> list[Zone]:
    """Retrieve all monitored venue zones with current occupancy and risk level."""
    return state.get_zones()


@router.get("/{zone_id}", response_model=Zone)
def get_zone_by_id(zone_id: str) -> Zone:
    """Retrieve a single zone by identifier."""
    zone = state.get_zone_by_id(zone_id)
    if not zone:
        raise HTTPException(status_code=404, detail=f"Zone '{zone_id}' not found")
    return zone
