# CrowdShield AI — End-to-End Integration & Implementation Plan

**Role:** Senior Software Architect  
**Project:** CrowdShield AI (ML Core & Dashboard Frontend Integration)  
**Date:** November 21, 2026  
**Status:** Approved Architecture Plan  

---

## Executive Summary & Strategic Approach

### Should we implement Phase 1 first or tackle the operational/technical risks?
**Answer: We implement Phase 1 directly, with all operational and technical risk mitigations engineered into the code from Day 1.**

In production systems architecture, operational and technical risks (such as process lifetime leaks, thread contention, and frame queue lag) are not tackled in a vacuum. Instead, they serve as non-negotiable **architectural constraints** for each implementation phase:
1. **Singleton Worker Lifecycle:** Tackled in Phase 1 by instantiating `InferencePipeline` once in `backend/ml_worker.py` rather than on demand per HTTP request.
2. **CPU Pacing & Queue Starvation:** Tackled in Phase 1 by bounding `FrameSource` sampling rate (`target_fps: 5.0`) with sleep pacing, preventing memory and CPU exhaustion on single-core setups.
3. **Concurrency Locks:** Tackled in Phase 1 by protecting `BackendState` with `threading.RLock()` and rejecting concurrent video sessions with HTTP 409 Conflict.
4. **Resource Teardown:** Tackled in Phase 1 with mandatory `try...finally` block cleanup on OpenCV video handles.
5. **ML Honesty & Limitations:** Tackled in Phase 1 (translation mapping `incomplete_input_zones` to alerts) and Phase 4 (displaying notices in the UI).

---

## 1. System Architecture Overview

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

## 2. Phased Execution Plan

### Phase 1: Backend Setup, State Store & Translation Engine
* **Objective:** Establish the FastAPI application, in-memory rolling state buffer, and the ML-to-UI data translation layer inside `CrowdShield AI/backend/`.
* **Exact Files to Create / Modify:**
  * `CrowdShield AI/backend/app.py`: FastAPI application entry point, CORS middleware, mounting feature routers under `/api`.
  * `CrowdShield AI/backend/schemas.py`: Pydantic models aligning exactly with frontend domain contracts in `src/types/domain.ts`.
  * `CrowdShield AI/backend/state.py`: Thread-safe `BackendState` with `threading.RLock()` and ring buffers (`collections.deque(maxlen=120)`) for time-series charts.
  * `CrowdShield AI/backend/translation.py`: Mapping layer from ML L1 output to frontend schemas:
    * Relative seconds (`timestamp_s`) $\to$ ISO-8601 UTC strings.
    * ML utilization decimal $\to$ 0–100 percentage occupancy.
    * Continuous image-space angles $\to$ 4 cardinal slices (`NORTH`, `SOUTH`, `EAST`, `WEST`) with image-space $y$-down correction ($45^\circ$–$135^\circ$ is `SOUTH`, $225^\circ$–$315^\circ$ is `NORTH`).
    * Derivation of `Alert` and `OperationalEvent` lists from critical risks and warnings.
  * `CrowdShield AI/backend/ml_worker.py`: Background worker managing video pipeline execution and pre-computed JSONL replay.
  * `CrowdShield AI/backend/routes/`:
    * `dashboard.py`: `/api/dashboard/*`
    * `zones.py`: `/api/zones/*`
    * `risk.py`: `/api/risk/*`
    * `alerts.py`: `/api/alerts`
    * `analytics.py`: `/api/analytics/*`
    * `pipeline.py`: `/api/pipeline/status`, `/start`, `/replay`, `/stop`
  * `CrowdShield AI/backend/requirements.txt`: FastAPI, Uvicorn, Pydantic, Python-multipart.
* **Testing & Verification:**
  * Execute standalone unit tests verifying translation null-safety, coordinate transformation, and schema compliance.
  * Smoke test FastAPI service startup (`python -m uvicorn backend.app:app --port 8000`).

---

### Phase 2: Frontend Repository Integration into Workspace
* **Objective:** Place the frontend repository into the workspace root (`frontend/`) and ensure builds/dependencies are clean.
* **Exact Steps:**
  * Clone or copy `https://github.com/tejrai07/CrowdSheild-AI` into `frontend/`.
  * Verify `package.json` scripts:
    * `npm run build` (TypeScript check + Vite production bundle).
    * `npm run lint` (Oxlint check).

---

### Phase 3: Connect Frontend API Layer to Backend Gateway
* **Objective:** Switch the frontend from simulated client-side mocks (`simulateLatency`) to real REST calls against the backend, while keeping fallbacks for offline development.
* **Exact Files to Modify:**
  * `frontend/vite.config.ts`: Configure reverse proxy routing `/api` to `http://127.0.0.1:8000`.
  * `frontend/src/services/api/dashboardApi.ts`: Implement `fetch('/api/dashboard/...')` calls with graceful fallback.
  * `frontend/src/services/api/zoneApi.ts`: Implement `fetch('/api/zones/...')`.
  * `frontend/src/services/api/riskApi.ts`: Implement `fetch('/api/risk/...')`.
  * `frontend/src/services/api/analyticsApi.ts`: Implement `fetch('/api/analytics/...')`.
  * `frontend/src/services/api/alertsApi.ts`: Implement `fetch('/api/alerts')`.
* **Testing & Verification:**
  * Run frontend dev server (`npm run dev`).
  * Verify browser network tab shows successful `200 OK` calls to `/api/dashboard/summary`, `/api/zones`, etc.

---

### Phase 4: Surfacing ML Honesty & Limitations in Dashboard UI
* **Objective:** Transparently communicate ML limitations in the UI per project honesty rules without redesigning existing layouts.
* **Exact Files to Modify:**
  * `frontend/src/features/settings/components/SystemInformationPanel.tsx`:
    * Fetch and display `model_data_notice` and pipeline health from `/api/pipeline/status`.
    * Display low future escalation recall notice (~12.5% at +5m, ~10.7% at +10m).
  * `frontend/src/features/risk/components/RiskHorizonCards.tsx`:
    * Render a badge indicating when predictions are "Provisional" due to `incomplete_input_zones`.
  * `CrowdShield AI/backend/translation.py`:
    * Automatically emit a `MEDIUM` severity alert whenever an incomplete track history or stationary dropout occurs.

---

### Phase 5: End-to-End Pipeline Verification & Live Stream Replay
* **Objective:** Verify end-to-end data flow: video file $\to$ inference $\to$ backend state buffer $\to$ frontend dashboard.
* **Exact Steps:**
  1. Trigger video processing via `POST /api/pipeline/start` using `sample_data/test_video.mp4`.
  2. Confirm ML worker processes detections and tracks via ByteTrack.
  3. Verify zone occupancy updates in real time on the Dashboard (`http://localhost:5173/`).
  4. Verify risk score timeline and future horizons reflect pipeline calculations.
  5. Test `POST /api/pipeline/stop` and confirm clean thread termination and resource cleanup.

---

## 3. Risk Matrix & Architectural Mitigations

| # | Risk Description | Architectural Impact | Engineered Mitigation Strategy | Implementation Phase |
|---|---|---|---|---|
| **1** | **Memory Leaks from Model Loading** | PyTorch and YOLOv8n allocate ~900 MB RAM. Reloading models per HTTP request causes rapid memory exhaustion and process crash. | The `InferencePipeline` and YOLO weights are instantiated **exactly once** in a singleton worker instance (`MLWorker`). Request handlers never instantiate models. | Phase 1 |
| **2** | **Inference Latency & CPU Starvation** | On a 1-core CPU, YOLOv8n runs at ~8.95 FPS (~111 ms/frame). Ingesting 30 FPS video causes runaway frame queues and latency spikes. | `FrameSource` clamps processing to `effective_target_fps = min(target_fps, source_fps)`. The `MLWorker` thread paces execution at 5.0 FPS with explicit sleep delays. | Phase 1 |
| **3** | **Concurrent Session Overwrites** | Multiple browser clients polling or multiple video uploads simultaneously corrupt the shared `BackendState`. | `BackendState` uses `threading.RLock()`. `MLWorker` enforces a single active pipeline session mutex, rejecting concurrent runs with HTTP 409 Conflict. | Phase 1 |
| **4** | **Unclosed Video File Handles** | Prematurely stopping video processing or unhandled worker exceptions leave OpenCV `VideoCapture` handles open. | All frame loops in `MLWorker._run_pipeline()` are wrapped in strict `try ... finally` blocks that guarantee `FrameSource.close()` is called. | Phase 1 |
| **5** | **Low Future Escalation Recall (~11-12%)** | Operators may mistake low future risk for guaranteed safety when the model merely failed to forecast escalation. | The backend preserves and passes `model_data_notice` in every status call; the frontend displays this warning on the Settings and Risk pages. | Phase 4 |
| **6** | **Out-of-Distribution Movement Inputs** | Stationary crowds or tracking dropouts cause `NaN` speed and variance, producing unreliable/provisional risk scores. | The translation layer generates explicit `MEDIUM` severity alerts for any zone listed in `incomplete_input_zones`, alerting operators to tracking gaps. | Phase 1 & 4 |
| **7** | **Coordinate System Angle Inversion** | ML image coordinates treat $+y$ as downward ($90^\circ$ = South). Traditional Cartesian math ($90^\circ$ = North) would invert flow direction charts. | Translation logic explicitly implements the image coordinate convention: $[45^\circ, 135^\circ) \to \text{SOUTH}$, $[225^\circ, 315^\circ) \to \text{NORTH}$. | Phase 1 |

---

## 4. Definition of Done (DoD)

1. [ ] Backend starts without warnings on `http://127.0.0.1:8000` exposing OpenAPI docs at `/docs`.
2. [ ] All API endpoints return data matching the schemas defined in `src/types/domain.ts`.
3. [ ] Frontend builds cleanly with zero TypeScript errors (`npm run build`).
4. [ ] Frontend dashboard displays live crowd metrics when the background worker processes `sample_data/test_video.mp4`.
5. [ ] Explainability and multi-horizon risk predictions populate without crashes or `NaN` values.
6. [ ] Limitations regarding synthetic data and low escalation recall are prominently surfaced.
7. [ ] Stopping the pipeline cleans up memory and background threads without lingering processes.
