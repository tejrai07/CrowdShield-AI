<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11" />
  <img src="https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black" alt="React 19" />
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/YOLOv8n-Ultralytics-FF6F00" alt="YOLOv8n" />
  <img src="https://img.shields.io/badge/Tests-361%20passed-brightgreen" alt="361 tests passed" />
  <img src="https://img.shields.io/badge/License-Educational%20%2F%20Research-blue" alt="License" />
</p>

# CrowdShield AI

> **AI-Powered Early-Warning & Crowd Disaster Management System**

CrowdShield AI is an intelligent, end-to-end platform that watches live video feeds of crowded venues and answers one question: **"Is this crowd becoming dangerous, and what should we do about it?"**

It detects every person in the frame, tracks their movement across time, maps them into defined venue zones, measures crowd density and flow behavior, predicts whether risk levels will escalate in the next 5–10 minutes, explains *why* a prediction was made, and recommends where to divert people before a situation becomes critical — all visualized in a real-time operational dashboard built for venue operators and safety teams.

---

## What Problem Does This Solve?

Crowd disasters — crushes, stampedes, flow collapses — kill hundreds of people every year at religious gatherings, concerts, stadiums, and transit hubs. Most crowd management today is reactive: operators watch CCTV feeds and make gut decisions under pressure. By the time danger is visible to the human eye, it is often too late to intervene.

CrowdShield AI is an attempt to shift that paradigm from **reactive monitoring** to **predictive early-warning**. Instead of waiting for a stampede to begin, the system continuously analyzes crowd behavior and flags escalation risk *before* it becomes visually obvious — giving operators a window to act.

> **Important:** This is an engineering prototype and research project, not a production-certified safety system. See the [Honest Limitations](#honest-limitations--what-this-system-cannot-do) section below.

---

## What We Have Actually Built

### ML & Computer Vision Pipeline (Phases B1–L1) ✅

A complete, tested inference pipeline that processes video frame-by-frame:

| Stage | What It Does | Implementation |
|---|---|---|
| **Person Detection** | Finds every person in each video frame | YOLOv8n (CPU-optimized, ~8.95 FPS on single core) |
| **Multi-Object Tracking** | Follows individuals across frames without facial recognition | ByteTrack (IoU-based motion association) |
| **Zone Mapping** | Assigns each person to a defined venue zone | Ray-casting point-in-polygon (`features/zones.py`) |
| **Crowd Metrics** | Measures density (people/m²), velocity (px/s), flow direction, dispersion | Instantaneous per-zone computation (`features/crowd_metrics.py`) |
| **Temporal Analysis** | Detects trends over rolling windows (5s, 15s, 30s, 60s) | Linear regression on density trends, surge flags, bottleneck pressure (`features/temporal.py`) |
| **Risk Prediction** | Classifies current risk + forecasts at +5min and +10min horizons | HistGradientBoosting ensemble with calibrated display scores 0–100 (`models/future_risk.py`) |
| **Explainability** | Shows *which* metrics are driving a risk score up or down | Lightweight occlusion-based feature attribution (`models/explainability.py`) |
| **Diversion Recommendations** | Suggests where to redirect crowd flow when a zone is overloaded | Graph-based adjacency + capacity-aware routing (`decision/diversion_engine.py`) |
| **Pipeline Orchestrator** | Chains all stages and emits validated JSON per frame | End-to-end pipeline (`inference/pipeline.py`) |

**Test coverage:** 361 unit and integration tests passing across the full pipeline.

### FastAPI Backend Gateway ✅

A real-time backend that bridges the ML pipeline output to the frontend dashboard:

- **Thread-safe state engine** (`backend/state.py`) — `threading.RLock()` protecting shared state with rolling ring buffers (`collections.deque(maxlen=120)`) for time-series charts.
- **Singleton ML worker** (`backend/ml_worker.py`) — Models loaded once in memory, not per-request. Processes video in a background thread with explicit `try...finally` cleanup on OpenCV handles. Rejects concurrent sessions with HTTP 409.
- **Translation layer** (`backend/translation.py`) — Converts raw ML output into the exact JSON shapes the frontend expects: relative timestamps → ISO-8601, utilization decimals → 0–100 percentages, continuous angles → cardinal directions (with correct image-coordinate convention where y-axis points downward).
- **15 REST endpoints** covering dashboard summaries, zone status, risk forecasts, feature attributions, alerts, analytics time-series, pipeline control, and ML transparency metadata.

### React 19 Operational Dashboard ✅

A modern, responsive UI built for real-time crowd monitoring:

- **6 feature pages:** Dashboard overview, Zone monitoring, Risk analysis, Crowd analytics, Alert management, System settings.
- **Live data polling** via TanStack Query hooks with configurable refresh intervals (10–30s per feature).
- **Graceful degradation** — Every API call uses `fetchWithFallback()`: if the backend is running, the dashboard shows live ML data; if the backend is offline, it seamlessly falls back to client-side mock generators with no crashes or error screens.
- **ML honesty surfaced in the UI:**
  - Risk horizon cards display **"Provisional (~12% recall)"** badges on future predictions.
  - Settings page shows the backend's `model_data_notice` and lists any zones with incomplete movement inputs.
  - Model settings panel discloses training data source, evaluation metrics, and prediction limitations.

### Full-Stack Integration ✅

- Vite reverse proxy (`/api → http://127.0.0.1:8000`) for seamless development.
- Pydantic backend schemas match TypeScript `domain.ts` types field-for-field.
- Pipeline control from the browser: start video inference, replay pre-computed frames, or stop the worker — all via REST.

---

## Honest Limitations & What This System Cannot Do

We believe in transparency about what works and what doesn't. These are real, measured limitations — not theoretical risks:

### 🔶 Synthetic Training Data Only
All risk prediction models were trained on **synthetically generated crowd scenarios** (Phase G1). They have never seen real-world crowd incident footage. The models learn patterns from simulated density/velocity/flow distributions, which may not capture the full complexity of real crowd dynamics.

### 🔶 Low Future Escalation Recall
The future risk models are good at recognizing **persistence** of an already-high risk state, but poor at **anticipating fresh escalations** from a calm baseline:

| Horizon | Escalation Recall | What This Means |
|---|---|---|
| +5 min | ~12.5% | The model catches roughly 1 in 8 genuine escalations |
| +10 min | ~10.7% | Even worse at longer horizons |

This means: if the crowd is currently calm and a stampede begins developing, the model will likely **miss** the early warning window. It should not be trusted as a sole alerting mechanism for novel escalation events.

### 🔶 Movement-Dependent Input Sensitivity
When crowds are stationary or when the tracker loses people (occlusion, camera angle changes), velocity and flow direction inputs become `NaN` or unreliable. The system flags these zones as **"incomplete input"** and marks their predictions as provisional rather than silently producing false safety readings.

### 🔶 Single-Camera, Single-Session
The current architecture processes one video source at a time in a single background thread. It is not designed for multi-camera deployments, horizontal scaling, or high-availability production use.

### 🔶 No Real-World Validation
This system has not been deployed at any real venue, tested against real crowd incident data, or validated by crowd safety professionals. It is an engineering and research prototype.

---

## Architecture

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
                                          │
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
│   │   ├── app/                  # Router, layouts, navigation
│   │   ├── components/           # Reusable UI primitives
│   │   ├── features/             # Dashboard, Zones, Risk, Analytics, Alerts, Settings
│   │   ├── services/api/         # API client layer with backend/mock fallback
│   │   └── types/domain.ts       # Frontend domain type contract
│   ├── package.json
│   └── vite.config.ts            # Dev server with /api proxy to backend
│
├── CrowdShield AI/               # ML Core & Backend Service
│   ├── backend/                  # FastAPI gateway
│   │   ├── app.py                # Entry point & CORS
│   │   ├── ml_worker.py          # Singleton background inference worker
│   │   ├── state.py              # Thread-safe rolling state buffers
│   │   ├── translation.py        # ML output → frontend schema mapping
│   │   ├── schemas.py            # Pydantic response models
│   │   └── routes/               # Feature route modules
│   │
│   ├── vision/                   # YOLOv8n detection & ByteTrack tracking
│   ├── features/                 # Zone assignment, crowd metrics, temporal analysis
│   ├── models/                   # Risk prediction & explainability
│   ├── decision/                 # Diversion recommendation engine
│   ├── inference/                # L1 end-to-end pipeline orchestrator
│   ├── config/                   # Pydantic config & YAML venue definitions
│   ├── sample_data/              # Test video & synthetic datasets
│   ├── scripts/                  # CLI tools for benchmarking & demos
│   └── tests/                    # 361 unit & integration tests
│
├── IMPLEMENTATION_PLAN.md        # Architecture decisions & risk matrix
└── README.md
```

---

## Getting Started

### Prerequisites
- **Python 3.11** (3.12+ may work but is untested)
- **Node.js 20+** and **npm**
- ~1 GB RAM for ML models (YOLOv8n + risk ensemble)

### 1. Backend & ML Setup
```bash
cd "CrowdShield AI"

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install ML and backend dependencies
pip install -r requirements.txt
pip install -r backend/requirements.txt

# Start the FastAPI server
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

- **Swagger API docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health check:** [http://localhost:8000/api/health](http://localhost:8000/api/health)

### 2. Frontend Dashboard
```bash
cd frontend

npm install
npm run dev
```

- **Dashboard:** [http://localhost:5173](http://localhost:5173)

> The frontend works with or without the backend running. When the backend is offline, it displays mock data seamlessly.

### 3. Run the ML Pipeline
Once both servers are running, start video inference from the dashboard or via API:

```bash
# Start processing a video file
curl -X POST http://localhost:8000/api/pipeline/start \
  -H "Content-Type: application/json" \
  -d '{"video_path": "sample_data/test_video.mp4", "fps": 5.0}'

# Or replay pre-computed results
curl -X POST http://localhost:8000/api/pipeline/replay \
  -H "Content-Type: application/json" \
  -d '{"jsonl_path": "sample_data/output/demo_pipeline.jsonl"}'

# Stop the pipeline
curl -X POST http://localhost:8000/api/pipeline/stop
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/dashboard/summary` | Venue-wide crowd status, counts & prediction confidence |
| `GET` | `/api/dashboard/crowd-density-trend` | Historical crowd density time-series |
| `GET` | `/api/dashboard/risk-level-trend` | Risk score evolution over time |
| `GET` | `/api/dashboard/zone-occupancy-distribution` | Per-zone occupancy breakdown |
| `GET` | `/api/dashboard/flow-direction-distribution` | Cardinal flow direction distribution (N/S/E/W) |
| `GET` | `/api/dashboard/recent-events` | Recent surge detections & operational events |
| `GET` | `/api/zones` | All monitored zones with live occupancy and risk |
| `GET` | `/api/zones/{id}` | Single zone details |
| `GET` | `/api/risk/forecast` | Multi-horizon risk (CURRENT, +5min, +10min) |
| `GET` | `/api/risk/timeline` | Historical risk score timeline |
| `GET` | `/api/risk/feature-contributions` | Explainability: which metrics drive the risk score |
| `GET` | `/api/alerts` | Active alerts, surge warnings & tracking gap notices |
| `GET` | `/api/analytics/*` | Density, speed, variance, inflow/outflow, bottlenecks |
| `GET` | `/api/pipeline/status` | Pipeline health, model notices & capability flags |
| `POST` | `/api/pipeline/start` | Start background video inference |
| `POST` | `/api/pipeline/replay` | Replay pre-computed JSONL frames |
| `POST` | `/api/pipeline/stop` | Stop active ML worker |

---

## Testing

```bash
cd "CrowdShield AI"

# Full test suite (361 tests)
pytest tests/ -v

# Backend-only tests
pytest tests/test_backend_api.py tests/test_backend_translation.py -v

# Frontend checks
cd ../frontend
npm run build    # TypeScript compilation + Vite production bundle
npm run lint     # Oxlint static analysis
```

**Current test results:** 361 passed, 5 skipped (integration tests requiring GPU/camera), 0 failures.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Detection** | YOLOv8n (Ultralytics) |
| **Tracking** | ByteTrack (IoU-based, no biometrics) |
| **Risk Models** | HistGradientBoosting + RandomForest ensemble (scikit-learn) |
| **Explainability** | Occlusion-based feature attribution |
| **Backend** | FastAPI + Uvicorn + Pydantic |
| **Frontend** | React 19 + TypeScript + Vite + TailwindCSS v4 |
| **State Management** | TanStack Query (server), Zustand (client) |
| **Charts** | Recharts |
| **Testing** | pytest (backend), oxlint (frontend) |

---

## What's Next

These are directions for future work — not promises, and not currently implemented:

- [ ] **Real-world data validation** — Test against actual crowd footage and incident recordings.
- [ ] **Model retraining** — Improve future escalation recall with real or higher-fidelity synthetic data.
- [ ] **Multi-camera support** — Process multiple video streams simultaneously.
- [ ] **WebSocket streaming** — Replace polling with push-based real-time updates.
- [ ] **Persistent storage** — Add database backing for historical analysis and audit trails.
- [ ] **Authentication & RBAC** — Role-based access for operators vs. administrators.
- [ ] **Containerized deployment** — Docker Compose for one-command setup.

---

## License

This project is developed for educational, research, and crowd safety decision-support prototyping. It is not certified or validated for production safety-critical use.
