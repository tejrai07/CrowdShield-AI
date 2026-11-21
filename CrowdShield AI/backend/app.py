"""CrowdShield AI FastAPI backend application entry point."""
from __future__ import annotations

import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure CrowdShield AI root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.routes.alerts import router as alerts_router
from backend.routes.analytics import router as analytics_router
from backend.routes.dashboard import router as dashboard_router
from backend.routes.pipeline import router as pipeline_router
from backend.routes.risk import router as risk_router
from backend.routes.zones import router as zones_router

app = FastAPI(
    title="CrowdShield AI — API Service",
    description=(
        "Backend bridging the ML video inference pipeline with the CrowdShield React dashboard. "
        "Exposes real-time crowd occupancy, multi-horizon risk predictions, model explainability, "
        "and pipeline transparency notices."
    ),
    version="1.0.0",
)

# Enable CORS for frontend Vite dev server and production origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all feature routers under /api
app.include_router(dashboard_router, prefix="/api")
app.include_router(zones_router, prefix="/api")
app.include_router(risk_router, prefix="/api")
app.include_router(alerts_router, prefix="/api")
app.include_router(analytics_router, prefix="/api")
app.include_router(pipeline_router, prefix="/api")


@app.get("/api/health", tags=["Health"])
def health_check() -> dict[str, str]:
    """Health check endpoint for service readiness."""
    return {"status": "ok", "service": "crowdshield-backend"}


@app.get("/", include_in_schema=False)
def root_redirect() -> dict[str, str]:
    return {
        "service": "CrowdShield AI Backend",
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
