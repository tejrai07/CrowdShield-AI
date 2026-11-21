# CrowdShield AI

> **AI-Powered Early-Warning & Crowd Disaster Management System**  
> Moving crowd management from reactive monitoring toward predictive public safety.

CrowdShield AI is an end-to-end intelligent platform designed to detect early warning signs of dangerous crowd congestion, flow collapse, and potential stampede conditions. It combines edge-capable computer vision, rolling temporal behavior analytics, multi-horizon machine learning risk forecasting, occlusion-based explainability, and capacity-aware crowd diversion decision support — visualized in a modern real-time operational dashboard.

---

## Architecture Overview

CrowdShield AI is organized as a decoupled, multi-tier system:

```
┌────────────────────────────────────────────────────────┐
│                      FRONTEND                          │
│     React 19 + TypeScript + Vite + TailwindCSS v4     │
│  src/services/api/*Api.ts  ──►  Vite Proxy /api        │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP / JSON
                            ▼
┌────────────────────────────────────────────────────────┐
│                   BACKEND GATEWAY                      │
│                 FastAPI + Uvicorn                      │
│                                                        │
│  ┌──────────────────────┐    ┌──────────────────────┐  │
│  │   REST Controllers   │    │  In-Memory State     │  │
│  │  /dashboard, /zones, │◄───┤  BackendState        │  │
│  │  /risk, /pipeline    │    │  (Ring Buffers: 120) │  │
│  └──────────────────────┘    └──────────▲───────────┘  │
│                                         │              │
│                              Translation Layer         │
│                              (translation.py)          │
│                                         ▲              │
└─────────────────────────────────────────┼──────────────┘
                                          │ Memory / State Update
┌─────────────────────────────────────────┴──────────────┐
│                  ASYNC ML WORKER                       │
│              (backend/ml_worker.py)                    │
│                                                        │
│   Video / Stream  ──►  InferencePipeline (L1)          │
│                        - YOLOv8n (Detection)           │
│                        - ByteTrack (Tracking)          │
│                        - MultiHorizonRiskModel (H1/I1) │
│                        - Occlusion Explainer (J1)      │
│                        - DiversionEngine (K1)          │
└────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
CrowdShield-AI/
├── frontend/                     # React 19 + Vite Dashboard
│   ├── src/
│   │   ├── app/                  # Router, Layouts, TopNav, PageTransitions
│   │   ├── components/           # UI primitives, tables, cards, forms
│   │   ├── features/             # Dashboard, Zones, Risk, Analytics, Alerts, Settings
│   │   ├── services/             # API client layer (dashboardApi, zoneApi, riskApi, etc.)
│   │   └── types/domain.ts       # Core UI domain schema contract
│   ├── package.json              # Frontend dependencies
│   └── vite.config.ts            # Vite config with backend API proxy
│
├── CrowdShield AI/               # ML Core & Backend Service
│   ├── backend/                  # FastAPI Gateway & State Aggregator (Phase 1)
│   │   ├── app.py                # FastAPI entry point & CORS configuration
│   │   ├── ml_worker.py          # Background worker (video pipeline & replay)
│   │   ├── state.py              # Thread-safe in-memory ring buffers & state
│   │   ├── translation.py        # ML L1 output -> Frontend schema translation
│   │   ├── schemas.py            # Pydantic response models
│   │   └── routes/               # Dashboard, zones, risk, alerts, analytics, pipeline
│   │
│   ├── vision/                   # Vision detection (YOLOv8n) & ByteTrack tracking
│   ├── features/                 # Zone polygon assignment, crowd metrics, temporal features
│   ├── models/                   # MultiHorizonRiskModel (HGB), occlusion explainability
│   ├── decision/                 # Crowd diversion & routing engine
│   ├── inference/                # L1 end-to-end inference pipeline
│   ├── config/                   # Typed Pydantic configuration & YAML venue definitions
│   ├── sample_data/              # Sample test video & synthetic calibration datasets
│   ├── scripts/                  # CLI benchmark, training, and demo runners
│   └── tests/                    # Complete pytest verification suite
│
├── IMPLEMENTATION_PLAN.md        # Comprehensive integration roadmap & risk matrix
└── README.md                     # Monorepo documentation
```

---

## Machine Learning & Vision Pipeline (Milestone 1)

The ML pipeline processes video footage through the following stages:

```
VIDEO -> PERSON DETECTION -> TRACKING -> ZONE-LEVEL METRICS
      -> TEMPORAL FEATURES -> MULTI-HORIZON RISK -> EXPLAINABILITY
      -> DIVERSION ENGINE -> STRUCTURED JSON OUTPUT
```

1. **Person Detection (Phase B1):** Ultralytics YOLOv8n CPU-optimized detector (`vision/detector.py`).
2. **Multi-Object Tracking (Phase C1):** ByteTrack motion-only IoU association (`vision/tracker.py`). Ephemeral session-scoped IDs with no facial recognition or biometric profiling.
3. **Zone Mapping & Metrics (Phase D1 & E1):** Ray-casting point-in-polygon assignment (`features/zones.py`) and instantaneous crowd metrics: density ($\text{people}/\text{m}^2$), velocity ($\text{px}/\text{s}$), circular flow direction and dispersion (`features/crowd_metrics.py`).
4. **Rolling Temporal Trends (Phase F1):** Bounded rolling windows ($5\text{s}, 15\text{s}, 30\text{s}, 60\text{s}$), linear density trend regression, surge flag triggers, and chokepoint bottleneck pressure (`features/temporal.py`).
5. **Multi-Horizon Risk Prediction (Phase H1 & I1):** HistGradientBoosting classifiers predicting current, $+5\text{min}$, and $+10\text{min}$ crowd risk classes and display scores ($0\text{--}100$) (`models/future_risk.py`).
6. **Local Explainability (Phase J1):** Lightweight occlusion-based feature attribution measuring individual metric contributions against training medians (`models/explainability.py`).
7. **Crowd Diversion Engine (Phase K1):** Graph-based adjacency and capacity-aware diversion decision engine identifying safe destination zones (`decision/diversion_engine.py`).
8. **End-to-End Orchestrator (Phase L1):** Fully assembled pipeline emitting validated, JSON-safe frames (`inference/pipeline.py`).

---

## ML Limitations & Honesty Disclaimers

CrowdShield AI is an engineering decision-support prototype and does **not** claim to predict stampedes with absolute certainty:
* **Synthetic Training Baseline:** Models are trained on synthetic temporal crowd scenarios (Phase G1) and have not been validated on real-world crowd stampede incidents.
* **Escalation Recall:** Future horizon models have low escalation recall ($\sim 12.5\%$ at $+5\text{m}$, $\sim 10.7\%$ at $+10\text{m}$). They are effective at recognizing persistence of established high risk, but conservative when anticipating fresh escalations.
* **Movement-Dependent Input Sensitivity:** In dense stationary crowds or tracking dropouts where track movement history is undefined (`NaN`), model outputs are flagged as diagnostic/provisional. The system explicitly issues `MEDIUM`/`HIGH` severity alerts rather than falsely reporting safety.

---

## Quickstart Guide

### 1. Prerequisites
* **Python 3.11** (recommended)
* **Node.js 20+** and **npm**

### 2. Backend & ML Setup
```bash
cd "CrowdShield AI"

# Activate your virtual environment (or create one)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r backend/requirements.txt

# Start the FastAPI backend service
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```
* **API Documentation (Swagger UI):** `http://localhost:8000/docs`
* **Health Check:** `http://localhost:8000/api/health`

### 3. Frontend Dashboard Setup
```bash
cd frontend

# Install frontend packages
npm install

# Start Vite development server
npm run dev
```
* **Dashboard URL:** `http://localhost:5173/`

---

## API Endpoints Summary

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/dashboard/summary` | Venue-wide aggregated metrics, crowd status & confidence |
| `GET` | `/api/dashboard/crowd-density-trend` | Historical crowd density rolling time-series |
| `GET` | `/api/dashboard/risk-level-trend` | Risk score evolution timeline |
| `GET` | `/api/dashboard/flow-direction-distribution` | Cardinal crowd flow distribution (N/S/E/W) |
| `GET` | `/api/zones` | Monitored venue zones with live occupancy and status |
| `GET` | `/api/risk/forecast` | Multi-horizon risk predictions (CURRENT, +5min, +10min) |
| `GET` | `/api/risk/feature-contributions` | Occlusion explainability feature attributions |
| `GET` | `/api/alerts` | Active crowd surges, threshold crossings & tracking warnings |
| `GET` | `/api/pipeline/status` | ML pipeline status, synthetic notice & capability flags |
| `POST` | `/api/pipeline/start` | Start background video inference worker |
| `POST` | `/api/pipeline/replay` | Start streaming pre-computed JSONL frame replay |
| `POST` | `/api/pipeline/stop` | Terminate active ML background worker |

---

## Testing

```bash
cd "CrowdShield AI"

# Run backend API and translation tests
pytest tests/test_backend_translation.py tests/test_backend_api.py -v

# Run complete ML pipeline test suite (300+ unit tests)
pytest tests/ -v -m "not integration"
```

---

## License

This project is developed for educational, research, and crowd safety decision-support prototyping.
