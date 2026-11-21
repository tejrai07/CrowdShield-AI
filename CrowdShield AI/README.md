# CrowdShield AI

AI-powered early-warning system for preventing dangerous crowd congestion and
potential stampede situations. Moves crowd management from reactive
monitoring toward predictive public safety.

Pipeline (target, Milestone 1):

```
VIDEO -> PERSON DETECTION -> TRACKING -> ZONE-LEVEL CROWD METRICS
      -> TEMPORAL FEATURES -> RISK MODEL -> CURRENT/5-MIN/10-MIN RISK
```

CrowdShield does not claim to predict stampedes with certainty. It estimates
elevated probability of crowd-risk escalation from crowd-level behavioral
signals (density, speed, flow). It never performs facial recognition or
identity tracking — tracking IDs are ephemeral, session-only technical
identifiers used solely for trajectory/speed calculation.

## Current Milestone

**Milestone 1 — AI/ML Core & API Gateway**, Phase **Phase 1 (Backend Gateway & State Integration)** — COMPLETE.

## Implementation Status

| Phase | Description | Status |
|---|---|---|
| A1 | Repo skeleton, config, logging, reproducibility | ✅ Complete |
| B1 | YOLO person detection + benchmarking | ✅ Complete |
| C1 | Multi-object tracking (ByteTrack) | ✅ Complete |
| D1 | Venue zones + point-in-polygon assignment | ✅ Complete |
| E1 | Per-zone crowd metrics | ✅ Complete |
| F1 | Rolling temporal features + trend detection | ✅ Complete |
| G1 | Synthetic crowd-risk scenario generator | ✅ Complete |
| H1 | Baseline risk model (Logistic Regression / HistGradientBoosting) | ✅ Complete |
| I1 | 5-min / 10-min future-risk prediction | ✅ Complete |
| J1 | Explainability (occlusion-based local explanations) | ✅ Complete |
| K1 | Crowd diversion / venue decision engine | ✅ Complete |
| L1 | End-to-end inference pipeline (video -> JSON) | ✅ Complete |
| L1-fix | Post-L1 ML reliability fix (retrained H1/I1, undefined-input handling, K1 reliability policy) | ✅ Complete |
| **Phase 1** | **FastAPI backend gateway, in-memory state buffer, translation engine & ML worker** | ✅ **Complete** |
| Phase 2 | Frontend workspace integration & build verification | ⏳ In Progress |
| Phase 3 | Frontend API integration with live backend endpoints | ⏳ Pending |
| Phase 4 | Surfacing ML honesty & limitations in dashboard UI | ⏳ Pending |
| Phase 5 | End-to-end verification & live video stream validation | ⏳ Pending |

## Architecture

```
crowdshield/
├── backend/          # FastAPI REST API & real-time state gateway (Phase 1)
│   ├── app.py          # FastAPI application entry point, CORS & route mounting
│   ├── ml_worker.py    # Background worker executing InferencePipeline or replay
│   ├── state.py        # In-memory thread-safe state store & rolling metric histories
│   ├── translation.py  # L1 frame -> frontend domain models (src/types/domain.ts)
│   ├── schemas.py      # Pydantic response models for frontend contracts
│   └── routes/         # Feature routes: dashboard, zones, risk, alerts, analytics, pipeline
├── config/          # typed config schema (pydantic) + YAML loader
│   ├── schema.py
│   ├── loader.py
│   ├── default.yaml
│   └── venues/
├── vision/          # detection + tracking + benchmarking (Phase B1/C1)
│   ├── detector.py    # PersonDetector — the only module that imports ultralytics detection
│   ├── tracker.py     # PersonTracker — the only module that imports ByteTrack
│   └── benchmark.py   # FrameSource (sampling/streaming) + benchmark helpers
├── scripts/         # thin CLI runners, not imported by the pipeline
│   ├── run_detection_demo.py   # produces an annotated detection-only demo video
│   ├── run_tracking_demo.py    # produces an annotated detection+tracking demo video
│   ├── run_zone_demo.py        # produces an annotated detection+tracking+zones demo video
│   ├── run_metrics_demo.py     # prints per-zone E1 metric snapshots to console
│   ├── run_temporal_demo.py    # prints per-zone F1 rolling-window features to console
│   ├── run_benchmark.py        # compares YOLO model variants
│   ├── generate_synthetic_data.py  # G1 dataset generation CLI (--duration/--realizations-* overrides added in I1)
│   ├── analyze_synthetic_data.py   # G1 class-distribution/correlation sanity + plots
│   ├── train_risk_model.py         # H1: trains, compares, selects, evaluates, saves the current-risk model
│   ├── train_future_risk.py        # I1: trains/evaluates/saves the current+5m+10m model bundle
│   ├── run_explainability_demo.py  # J1: loads packaged models, prints representative explanations
│   ├── run_diversion_demo.py       # K1: console demo of the diversion decision engine
│   ├── run_pipeline.py             # L1: CLI - video -> one JSON object per processed frame (JSONL)
│   ├── analyze_training_coverage.py # reliability fix: audits which operating states a dataset contains, per split
│   └── evaluate_reliability.py     # reliability fix: old-vs-new evaluation -> models/artifacts/reliability_report.json
├── features/        # zone assignment (D1) + crowd metrics (E1) + temporal features (F1)
│   ├── zones.py          # point_in_polygon, ZoneManager, ZoneTransitionTracker
│   ├── crowd_metrics.py  # ZoneMetrics, CrowdMetricsCalculator, speed/direction/circular-stats helpers
│   └── temporal.py       # WindowFeatures, TemporalFeatures, ZoneTemporalTracker, TemporalFeatureBuilder
├── data/            # synthetic dataset generation (Phase G1)
│   ├── schemas.py             # ScenarioConfig, TransitionConfig, GeneratorConfig (pydantic)
│   ├── loader.py               # YAML loader for scenarios.yaml, mirrors config/loader.py
│   ├── scenarios.yaml          # the 6 scenario + 6 transition definitions (engineering assumptions; unchanged, legacy datasets reproduce from it)
│   ├── scenarios_v2.yaml       # reliability fix: 19 scenarios + 11 transitions incl. empty/stationary/dropout states (provisional labels)
│   └── synthetic_generator.py  # generate_sequence/generate_dataset/split_dataset/save_dataset/load_dataset
├── models/          # H1 current-risk model + I1 future-risk models + J1 explainability
│   ├── features.py         # feature selection — single source of truth for leakage prevention
│   ├── risk_model.py       # RiskModel: fit/predict/predict_proba/predict_risk_score/evaluate/save/load
│   ├── future_risk.py      # I1: prepare_future_features, MultiHorizonRiskModel
│   ├── explainability.py   # J1: ModelExplainer, MultiHorizonExplainer, RiskExplanation, format_explanation
│   ├── training_utils.py   # shared H1/I1 training helpers (leakage audit, HGB pipeline, selection, latency)
│   └── artifacts/           # risk_model.pkl (H1), future_risk_{current,5m,10m}.pkl (I1) - all retrained on v2; reports;
│                            #   reliability_report.json; archive_pre_reliability/ = the previous artifacts (kept for the old-vs-new comparison)
├── decision/         # K1: crowd-diversion decision engine (decision-support layer)
│   ├── schemas.py          # ZoneState, DiversionRecommendation, ScoreBreakdown, RejectedCandidate
│   ├── zone_graph.py       # explicit configured zone adjacency + routing cost
│   ├── diversion_engine.py # hard rejection, weighted scoring, multi-source greedy allocation
│   └── state_builder.py    # E1/F1/H1/I1 outputs -> ZoneState bridge
├── inference/        # L1: end-to-end pipeline
│   ├── pipeline.py     # InferencePipeline: frame -> detect -> track -> zones/metrics -> temporal -> H1/I1 -> J1 -> K1 -> JSON
│   └── schema.py       # output schema constants, JSON-safety helper, validate_result()
├── evaluation/       # H1/I1 risk-specific metrics
│   ├── metrics.py       # dangerous_recall/precision, false_safe_rate, ordinal_mae, escalation_recall
│   └── probes.py        # controlled-state probe suite (empty zone, standing crowd, ...) through the real inference path
├── tests/
├── sample_data/
│   ├── test_video.mp4        # 20s pedestrian clip (see "Test Video" below)
│   ├── output/                 # generated: annotated videos, benchmark JSONs
│   ├── synthetic/               # generated: G1's 120s dataset (H1 training), metadata.json, plots/
│   ├── synthetic_long/          # generated: I1's legacy 900s dataset (has non-null future labels) + coverage_report.json
│   └── synthetic_v2/            # reliability fix: scenarios_v2 dataset (csv.gz), metadata.json, coverage_report.json
├── logging_setup.py
├── seed.py
└── main.py
```

Design principle: `vision/`, `features/`, `models/`, and `inference/` are
isolated behind `inference/pipeline.py` (added in Phase K). The backend team
will only ever call `process_frame()` / `process_video()` and consume
structured JSON — never import YOLO or tracking internals directly.

## Performance Architecture (permanent policy)

CrowdShield is a real-time prototype and is optimized for the lightest
architecture that still gives credible crowd-risk intelligence — not for
model accuracy alone.

```
Video -> Detection/Tracking (10-30 FPS) -> short rolling state buffer
      -> analytics ~1-2 Hz -> risk inference ~1 Hz -> async output
```

Rules enforced throughout the project:
- No expensive computation runs on every frame; only detection/tracking run
  at frame rate — analytics and risk inference run at 1-2 Hz.
- SHAP and any future LLM calls never run inside the hot video loop; only on
  state-change, operator request, or periodic snapshot.
- Pandas is never used in the real-time inference path — only for offline
  dataset generation, training, and evaluation. Live state uses `deque`,
  NumPy arrays, dicts, and typed lightweight objects.
- Trajectory/feature history is bounded (see `analytics.trajectory_history_seconds`,
  capped at 60s by schema validation) — no unbounded memory growth.
- YOLO model size is chosen by benchmarking (FPS, latency, detection
  quality, CPU/GPU/RAM), not assumed — smallest viable model wins.
- No continuous per-frame database writes; only meaningful events/snapshots
  are persisted, and that happens later via the backend.

## Setup

Requires **Python 3.11** (recommended; developed against 3.12 in the
current sandbox with no observed issues, but 3.11 has the most mature
Ultralytics/tracking library support for later phases).

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Dependency strategy

`requirements.txt` only lists dependencies needed by **completed** stages.
Each future phase adds its own dependencies at the time that phase is
implemented — see the comment header in `requirements.txt` for the planned
per-phase additions (ultralytics/opencv in Phase B, numpy/pandas in
Phase E-G, scikit-learn/xgboost in Phase H, shap in Phase J). This keeps
early-stage installs fast and avoids forcing large CV/ML packages onto
contributors who are just working on config or tests.

For development (adds pytest, pytest-cov, ruff):

```bash
pip install -r requirements-dev.txt
```

## How to Run

### 1. Backend API Service (Phase 1)

The FastAPI backend bridges the ML pipeline to the React frontend dashboard:

```bash
# Start backend server with auto-reload (port 8000)
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

* API Documentation (Swagger UI): `http://localhost:8000/docs`
* OpenAPI Schema: `http://localhost:8000/openapi.json`
* Health Check: `http://localhost:8000/api/health`

### 2. Standalone ML Pipeline CLI (Phase L1)

```bash
python main.py                                              # default config
python main.py --config path/to/config.yaml                 # explicit config
python main.py --venue config/venues/example_venue.yaml     # default + venue override
python main.py --config path/to/config.yaml --venue path/to/venue.yaml
```

Config is loaded exactly once per run (no duplicate reads), following:

| Arguments | Result |
|---|---|
| none | default config |
| `--config X` | config X |
| `--venue Y` | default config + venue override Y |
| `--config X --venue Y` | config X + venue override Y |

## Configuration Notes

- **Three related FPS values — do not conflate them:**
  - **`video.target_fps`** (configured) — the pipeline's requested
    *processing* rate, set in `config/default.yaml` (currently `12`, to
    match the bundled test video). This is independent of the source
    video's native frame rate — a 30fps source could still be requested
    at `target_fps: 15` to save compute via frame sampling.
  - **source FPS** (`FrameSource.source_fps`) — the *actual* native frame
    rate reported by the video file itself. This is fixed by the file, not
    configurable.
  - **`effective_target_fps`** (`FrameSource.effective_target_fps`) — the
    rate actually used for sampling: `min(target_fps, source_fps)`. The
    system cannot manufacture additional unique frames per second from a
    lower-FPS source, so `effective_target_fps` is only ever the
    configured `target_fps` clamped down to what the source can provide —
    never up. If `target_fps` exceeds `source_fps`, `FrameSource` logs a
    `WARNING` and processes at `source_fps` instead. See
    `config/schema.py::VideoConfig` and `vision/benchmark.py::FrameSource`.
- **`AppConfig` is frozen** (`ConfigDict(frozen=True)`) — immutable once
  loaded, so the hot inference loop can't accidentally mutate shared config
  mid-run.
- **`analytics.trajectory_history_seconds`** is schema-validated to never
  exceed 60s, enforcing the bounded-memory rule at load time rather than
  relying on runtime discipline.
- **`temporal.windows_seconds`** is schema-validated to be exactly
  `[5, 15, 30, 60]` — Milestone 1 explicitly does not support arbitrary
  window combinations. `temporal.density_growth_threshold` drives F1's
  `surge_flag` and is documented as an engineering trigger, not a
  validated safety threshold. `AppConfig` also cross-validates that
  `analytics.feature_history_seconds` (the bounded history buffer F1
  reuses) is at least as large as the biggest configured window.

## Tests

```bash
python -m pytest tests/test_smoke.py -v
```

Smoke tests are deliberately import-light (no ultralytics/torch/cv2/numpy/
pandas) so they run in well under a second and can gate every future
commit cheaply. This is enforced by `test_config_module_has_no_heavy_imports`,
which checks — in an isolated subprocess, so it isn't affected by whatever
else the test suite loads — that importing `config.loader`/`config.schema`
alone never pulls in a heavy dependency.

## Performance Notes (Phase A1)

- Startup time (`python main.py`, cold): ~0.22s
- Full smoke-test suite: 7 tests in ~0.34s
- No heavy imports (torch/ultralytics/cv2/numpy/pandas) triggered by config
  loading — verified by isolated-subprocess test

## Known Limitations

- No actual video/detection/tracking/ML yet — Phase A only validates
  environment, configuration, and logging.
- Zone polygon coordinates and `area_m2` are placeholder values in
  `config/venues/example_venue.yaml`; real venue calibration happens in
  Phase D.
- Sandbox currently runs Python 3.12 rather than the recommended 3.11;
  no incompatibilities observed for this phase's dependencies, but this
  should be revisited before Phase B (ultralytics/torch compatibility).

## Phase B1 — Person Detection + Benchmarking

### Detector architecture

`vision/detector.py` is the **only** module in the project that imports
`ultralytics`/`torch`. Everything downstream receives lightweight
`Detection` objects (`x1, y1, x2, y2, confidence, class_id`) — a frozen
dataclass, not a framework-specific object.

- **Bounding box convention:** `(x1, y1, x2, y2)` = top-left / bottom-right
  corners, in **pixel coordinates** of the source frame (not normalized),
  origin `(0, 0)` at the top-left (standard OpenCV/image-array convention).
  This convention is fixed now so tracking (Phase C) and zone mapping
  (Phase D) build on a stable contract.
- **Person-only filtering:** the model is called with `classes=[0]` (COCO
  `person`) so non-person classes are never computed downstream in
  postprocessing, and `detect()` defensively filters again in case a
  future model version behaves differently.
- **Model loaded once:** `PersonDetector.__init__` constructs the YOLO
  model exactly once; `detect()` only calls `.predict()` — verified by
  `test_model_constructed_exactly_once`.
- **Device handling:** `device="auto"` resolves to `cuda` only if
  `torch.cuda.is_available()`, else `cpu` — resolved device is logged
  explicitly (`Resolved device: CPU`), never silently assumed.

### Frame sampling (`vision/benchmark.py::FrameSource`)

Streams frames from a video file and samples down to `video.target_fps`
using **frame-index/timestamp comparison**, not `sleep()`. This makes
sampling deterministic and independent of how fast the machine actually
processes frames. Example: a 30 FPS source sampled at a 15 FPS target
yields approximately every second frame. `FrameSource.__iter__` is a
generator — frames are never accumulated into a list, so memory stays
bounded regardless of video length.

### How to run

```bash
pip install -r requirements.txt   # now includes ultralytics, opencv-python-headless

# Annotated demo video (bounding boxes + live count/FPS overlay)
python scripts/run_detection_demo.py \
  --video sample_data/test_video.mp4 \
  --output sample_data/output/annotated.mp4

# Model benchmark (compares yolov8n vs yolov8s)
python scripts/run_benchmark.py \
  --video sample_data/test_video.mp4 \
  --models yolov8n.pt yolov8s.pt --device cpu
```

### Test video

**File:** `sample_data/test_video.mp4` (2.1 MB, 20s, 12 FPS, 768×432)

**Source:** trimmed from `people-detection.mp4` in the public
[`intel-iot-devkit/sample-videos`](https://github.com/intel-iot-devkit/sample-videos)
repository (indoor pedestrian demo footage, commonly used for CV samples).
Trimmed to 20s with `ffmpeg -t 20 -c copy` for a manageable artifact size.
This is real (not synthetic) footage, included directly in the ZIP since
it's small — no separate download step needed.

**Honest caveat:** this clip shows at most 1–4 people at once in an indoor
room, not a dense crowd. It's sufficient for validating detection
correctness end-to-end (boxes, filtering, confidence, FPS) but **not**
representative of real crowd-density conditions. Sourcing or filming
genuine dense-crowd footage is flagged as a limitation below and should be
revisited by Phase D/E when zone-level density needs real validation data.

### Model benchmark results

Measured on this sandbox's CPU (**1 vCPU**, no GPU — see Performance
observations below), `image_size=640`, `confidence=0.4`, 100 processed
frames after a 5-frame warm-up (cold start excluded from steady-state
numbers):

| Model | Input | Device | Effective FPS | Avg Inference Latency | Avg People Detected | Weight size |
|---|---|---|---:|---:|---:|---:|
| yolov8n.pt | 640 | CPU | **8.95** | **111.7 ms** | 0.57 | 6.2 MB |
| yolov8s.pt | 640 | CPU | 3.70 | 270.3 ms | 0.57 | 21.5 MB |

Raw results: `sample_data/output/benchmark_results.json`.

Peak RSS memory (separately measured via `/usr/bin/time -v`, 60-frame runs):
yolov8n ≈ 895 MB, yolov8s ≈ 941 MB (dominated by PyTorch's own footprint,
not the weight file size).

### Selected model: **yolov8n.pt**

Both models detected the same average number of people on this sparse test
clip (0.57), so detection-quality tradeoff was not distinguishable on this
footage. On effective FPS, nano is **~2.4x faster** than small under
identical conditions, with a 3.5x smaller weight file. Per the "smallest
model that provides sufficiently credible person detection" rule, nano is
the clear choice until denser-crowd footage shows a small-person /
occlusion case where nano's recall degrades — at which point we'd
re-benchmark with real evidence rather than assume small is needed.

### Qualitative detection quality

Manual inspection of annotated frames (`sample_data/output/annotated.mp4`)
confirms: correct single-person bounding boxes with sensible confidence
(~0.87–0.9) on clearly visible subjects, empty frames correctly show 0
detections, no obvious false positives observed. Small/far and heavily
occluded person detection, and dense-crowd degradation, could **not** be
evaluated on this footage — see limitations.

### Model weights

YOLO weights (`yolov8n.pt`, `yolov8s.pt`) are **not** committed to the
repository or the ZIP. Ultralytics downloads them automatically on first
use from `https://github.com/ultralytics/assets/releases/...` and caches
them locally. First run will show a short download; subsequent runs reuse
the cached file.

### Tests

```bash
python -m pytest tests/ -v
```

- `tests/test_detector.py` — fast, mock-based unit tests (no real model
  download): person-class filtering, bounding box representation,
  configuration pass-through, device resolution, single-construction
  guarantee.
- `tests/test_frame_source.py` — fast, uses a tiny synthetically-generated
  video: sampling ratio correctness, no-sleep guarantee, generator
  laziness, missing-file handling.
- `tests/test_detector_integration.py` — **opt-in only**, uses the real
  model + real sample video. Skipped by default; enable with
  `CROWDSHIELD_RUN_INTEGRATION=1 python -m pytest tests/test_detector_integration.py`.
  Kept separate so the standard suite stays fast and doesn't require
  network access.

Full suite (excluding integration): **24 passed, 1 skipped in ~3.1s**.

### Performance observations

- This sandbox has **1 CPU core** and ~3.9 GB RAM — well below a realistic
  hackathon laptop or edge deployment target. The ≥10 FPS CPU engineering
  target from the spec was **not quite met** (8.95 FPS effective with
  nano) under this constraint; on a typical multi-core laptop CPU this
  would be expected to clear 10 FPS comfortably, since YOLO inference
  parallelizes across cores. This is documented rather than glossed over.
- Model inference latency ≈ end-to-end latency in this benchmark because
  drawing/writing happens in the demo script, not inside the benchmarked
  detect() path — the benchmark isolates model cost specifically.
- Cold start (model construction + first load) ranged ~0.3–6.5s across
  runs depending on whether Ultralytics' internal caches were warm; this
  is excluded from steady-state latency numbers per the spec's warm-up
  requirement.

### Known limitations

- Test video is a sparse indoor pedestrian clip, not a dense crowd —
  dense-crowd detection quality (occlusion, small/far people) is
  **unverified** and should be re-benchmarked once suitable footage is
  available (flagged for Phase D/E).
- Benchmark ran on a 1-core CPU sandbox; FPS numbers should be re-measured
  on target deployment hardware before treating them as representative.
- No formal precision/recall reported — no ground-truth annotations exist
  for this footage, consistent with the instruction not to fabricate
  quality numbers. Only qualitative/manual sanity checks were performed.
- GPU path (`device="cuda"`) is implemented and logged correctly but
  **untested** in this environment (no GPU available in the sandbox).

## Phase C1 — Multi-Object Tracking (ByteTrack)

### Tracker architecture

`vision/tracker.py` is the **only** module that imports ByteTrack
internals (`ultralytics.trackers.byte_tracker.BYTETracker`). Downstream
code receives lightweight `Track` objects
(`track_id, x1, y1, x2, y2, confidence, age_frames, frames_seen, last_seen_s`)
— a frozen dataclass, matching `Detection`'s bounding box convention
(pixel coordinates, top-left origin).

**Tracker choice: ByteTrack, not BoT-SORT/DeepSORT.** We only need
short-term trajectory continuity for speed/direction/flow calculations —
not identity persistence across long occlusions or camera cuts — so
motion-only (IoU) association is sufficient. No re-identification, face,
or appearance-embedding models are used.

**API:** `tracker.update(detections, timestamp_s) -> list[Track]`. A
`_DetectionArray` adapter (private to this module) converts our
`Detection` objects into the minimal array-like interface ByteTrack
expects (`xywh`, `conf`, `cls`), so ultralytics' tracking internals never
leak past this file.

### Privacy behavior

Track IDs are purely technical, session-scoped integers:
- Assigned by ByteTrack's internal counter, which is **reset to 0 every
  time a new `PersonTracker` is constructed** (verified by
  `test_multiple_tracker_instances_reset_id_counter`) — IDs never persist
  or carry meaning across sessions.
- Derived **only** from bounding-box motion/IoU — no face detection, no
  biometric features, no appearance re-identification.
- Never persisted to disk as identity information by this module.
- Carry no meaning beyond "this box was probably the same physical person
  as that box a moment ago" for the duration of one processing run.

### Trajectory representation

Each track's history is a bounded `deque[(timestamp_s, center_x, center_y)]`,
kept in **pixel-space** coordinates — not meters. Real-world speed
conversion is explicitly deferred to venue/camera calibration in a later
phase (do not assume pixels = meters). The buffer is pruned on every
`update()` call, dropping points older than `analytics.trajectory_history_seconds`
(config-driven, currently 15s) — this bounds memory regardless of session
length or how many distinct people have been seen, and does not depend on
ByteTrack ever explicitly signaling a track's removal.

### New-track confirmation delay (real ByteTrack behavior, not a bug)

A newly detected person does **not** appear in `update()`'s return value
until matched again in a **subsequent** frame — except for detections
present in the very first frame the tracker ever processes, which activate
immediately. This avoids spurious tracks from one-off false-positive
detections. `tests/test_tracker.py::test_new_person_gets_a_new_track_id`
encodes and verifies this behavior explicitly.

### Expected tracking quality (not promised)

- Same person across consecutive frames → same ID (verified).
- New person enters → new ID (verified, subject to the confirmation delay above).
- Person leaves → track expires from active output once it stops matching
  within `tracking.max_age` frames (verified).
- Short occlusion (a few frames) → ByteTrack's lost-track buffer may
  bridge the gap and preserve the ID — not guaranteed.
- **Heavy or prolonged occlusion, dense-crowd path crossing, or a person
  leaving and re-entering the frame will likely produce a new track_id.**
  This is an accepted limitation of motion-only tracking.

### FPS semantics fix (regression from B1)

B1's default config (`target_fps: 15`) exceeded the bundled test video's
native rate (`source_fps: 12`), which could make a benchmark report appear
to process more unique frames/sec than the source actually contains.
Fixed in `FrameSource`: the configured `target_fps` is preserved as-is,
but sampling internally uses `effective_target_fps = min(target_fps, source_fps)`,
with a `WARNING`-level log when clamping occurs. Regression tests:
`test_frame_source_clamps_target_fps_to_source_fps`,
`test_frame_source_no_clamp_when_target_below_source`.

**Follow-up fix:** the clamp above handles the *runtime* case for any
video, but the *shipped default* was still inconsistent — `config/default.yaml`
requested `target_fps: 15` against the bundled clip's 12 FPS, so a default
run would silently clamp every time. `config/default.yaml` now sets
`target_fps: 12` to match `sample_data/test_video.mp4`'s native rate, so
the out-of-the-box default no longer triggers the clamp warning. The
runtime clamp itself is unchanged and still applies to any other video
with a lower source FPS than configured.

### How to run

```bash
# Annotated tracking demo (bounding boxes + track ID + count/FPS overlay)
python scripts/run_tracking_demo.py \
  --video sample_data/test_video.mp4 \
  --output sample_data/output/tracked.mp4
```

### Tracking benchmark: detector-only vs. detector+tracker

Measured on this sandbox's CPU (1 vCPU), `yolov8n.pt`, 640px, 100 processed
frames (first 100 frames of the clip — see caveat below):

| Pipeline | Measured FPS | Avg End-to-End Latency | Tracking Overhead |
|---|---:|---:|---:|
| Detector only | 11.26 | 88.78 ms | — |
| Detector + ByteTrack | 10.86 | 92.08 ms | ~3 ms |

Raw results: `sample_data/output/tracking_benchmark.json`.

**Honest note on overhead:** across repeated runs, measured overhead
ranged from about **-7 ms to +3 ms** — i.e. within this sandbox's
measurement noise floor, not a real, consistently reproducible cost.
ByteTrack (IoU-based, no neural network) is expected to be cheap relative
to YOLO's ~90 ms/frame, and these numbers are consistent with that: the
tracker's true cost is too small to resolve reliably against single-core
CPU timing jitter. On this run: max 1 simultaneous track, 1 unique track
ID (this 100-frame benchmark window happened not to include the clip's
two-person segment — see full-clip demo numbers below for that).

**Full-clip demo run** (`scripts/run_tracking_demo.py`, all 240 frames):
max 2 simultaneous tracks, 3 total unique track IDs over the full 20s clip.

### Tracking validation

- **ID persistence:** verified via `test_same_person_across_consecutive_frames_keeps_same_id`.
- **New tracks:** verified via `test_new_person_gets_a_new_track_id`, correctly modeling the confirmation delay.
- **Expiration:** verified via `test_track_expires_after_max_age_frames`.
- **Multiple people:** verified via `test_multiple_simultaneous_people_get_distinct_ids`.
- **Trajectory buffer:** verified via `test_trajectory_buffer_stores_timestamps_and_centers`, `test_trajectory_buffer_respects_maximum_duration`, `test_trajectory_entry_removed_after_track_goes_stale`.
- **Bounding box conversion:** verified via `test_detection_to_tracker_bbox_conversion_is_correct`.
- **Qualitative visual check:** manual inspection of `sample_data/output/tracked.mp4` frames confirms correct `ID:N` overlay tracking the same visible person, and correct 0-count/empty output on frames with nobody present.

**Dense-crowd tracking remains unvalidated** and will be revisited before
final demo evaluation — the bundled clip shows at most 2 people
simultaneously. No source of genuinely dense pedestrian footage was found
within this project's approved network domains at this stage (checked
several other clips in the same `intel-iot-devkit/sample-videos` repo —
all are single/few-person demo clips by design, not crowd footage).

### Tests

```bash
python -m pytest tests/ -v
```

- `tests/test_tracker.py` — 11 fast unit tests using the **real** ByteTrack
  implementation (pure Python/NumPy, no video/YOLO/network needed):
  ID persistence, new-track confirmation, multi-person distinctness,
  expiration, trajectory bounding, bbox conversion, privacy/ID-reset.
- `tests/test_tracker_integration.py` — **opt-in only**
  (`CROWDSHIELD_RUN_INTEGRATION=1`), real detector + real tracker on the
  real sample video.
- `tests/test_frame_source.py` — extended with 2 regression tests for the
  FPS clamping fix.

Full suite (excluding integration): **37 passed, 2 skipped in ~2.5s.**

### Known limitations

- Dense-crowd tracking is unvalidated (see above) — flagged for revisit
  before final demo evaluation.
- Tracking benchmark overhead measurement is noise-dominated on this
  1-core sandbox; a cleaner measurement would need a less contended CPU
  or many more repeated runs averaged together.
- ByteTrack's new-track confirmation delay means a person entering the
  frame is invisible to `update()`'s output for exactly one frame after
  their first detection — acceptable for our use case, but worth knowing
  if a consumer expects zero-latency track creation.
- Track IDs can and will change across occlusion/crossing events in a
  real crowd — any Phase E crowd-flow calculation must be robust to
  track fragmentation, not assume one ID = one person for a whole session.

## Phase D1 — Zone Configuration + Point-in-Polygon Assignment

### Zone configuration

Zones are configured through the existing venue configuration system
(`config/venues/*.yaml`, loaded via `config/loader.py`) — no second
configuration mechanism was introduced. `ZoneConfig` (in
`config/schema.py`) now includes:

```yaml
zones:
  - id: entrance          # unique within the venue (enforced by validation)
    name: Entrance
    polygon: [[0,0], [220,0], [220,432], [0,432]]
    area_m2: 30.0          # stored for Phase E's density calc; unused in D1
    enabled: true           # disabled zones are skipped during assignment
```

### Coordinate system

Image **pixel** coordinates, origin `(0, 0)` at the frame's top-left, x
increases rightward, y increases downward — the same convention as
`Detection`/`Track` bounding boxes (see `vision/detector.py`,
`vision/tracker.py`). **Not** normalized (0-1) coordinates. Perspective
transformation / real-world calibration is explicitly deferred to a later
phase.

### Point representation

A track's spatial point is its bounding-box **center**
(`(x1+x2)/2, (y1+y2)/2`), already exposed as `Track.center` from Phase C1
— no new helper was needed. The full bounding box, face position, or pose
is deliberately not used for zone assignment.

### Point-in-polygon algorithm

`features/zones.py::point_in_polygon()` is a plain-Python ray-casting
(even-odd rule) implementation — no Shapely/GeoPandas or other heavy
dependency. Handles arbitrary simple polygons, convex or concave.

### Boundary behavior

**A point exactly on a zone's polygon boundary (edge or corner) counts as
inside that zone.** This is checked explicitly via `_point_on_segment()`
before the ray-casting pass runs, so the rule is deterministic and doesn't
depend on floating-point luck in the ray test itself. Verified by
`test_point_on_edge_counts_as_inside` and `test_point_on_corner_counts_as_inside`.

### Overlapping / adjacent zones

**Zones are evaluated in configuration order; the first enabled zone
whose polygon contains the point wins.** This also deterministically
resolves the common case of two zones sharing a boundary edge (e.g. two
zones drawn side-by-side) — the earlier zone in the YAML wins that shared
edge. Verified by `test_overlapping_zones_first_configured_wins` and
`test_adjacent_zones_shared_edge_first_configured_wins`.

### Zone assignment API

`features/zones.py::ZoneManager` owns exactly one responsibility — point/track
→ zone — and nothing else (no density, speed, flow, risk, or
visualization logic):

```python
zone_manager = ZoneManager(cfg.venue.zones)
zone_id = zone_manager.assign_point(x, y)      # str | None
zone_id = zone_manager.assign_track(track)     # uses track.center
```

Polygons are preprocessed once at construction; the hot-path methods
(`assign_point`/`assign_track`) do no per-call parsing or heavy
allocation — safe to call once per active track per processed frame.

### Zone transitions (API only, no metrics yet)

`features/zones.py::ZoneTransitionTracker` is a small, bounded,
session-scoped helper that remembers each track's most recently assigned
zone and reports `(previous_zone, current_zone)` on each update. **D1 does
not compute inflow/outflow** — this only exposes the raw transition pair
for Phase E to build on. State is capped at `max_tracked_ids` (default
500) with oldest-updated entries evicted first, so a long session can't
accumulate one entry per track_id ever seen.

### Config validation

Extends the existing pydantic schema (no new validation framework):
- Polygon must have ≥3 points (existing `min_length=3` constraint).
- Polygon coordinates must be numeric (enforced automatically by the
  `tuple[float, float]` type — non-numeric values raise a validation
  error, no extra code needed).
- `area_m2` must be positive (existing `gt=0` constraint).
- `enabled` must be boolean (standard pydantic bool field).
- **Zone IDs must be unique within a venue** — new
  `VenueConfig._unique_zone_ids` validator, raises on any duplicate.

Geometric validation intentionally stops there — D1 does not attempt to
prove polygons are simple/non-self-intersecting, per the "don't
over-engineer geometric validation" guidance for this stage.

### Example venue

`config/venues/example_venue.yaml` now defines three **demo-only** zones
(`entrance`, `main_area`, `exit`) as non-overlapping vertical bands across
the bundled test video's 768×432 frame. The file is explicitly commented
as illustrative — **not a real venue calibration**.

### How to run

```bash
python scripts/run_zone_demo.py \
  --video sample_data/test_video.mp4 \
  --output sample_data/output/zones.mp4 \
  --venue config/venues/example_venue.yaml
```

### Performance

Measured on this sandbox's CPU, 100 processed frames, comparing
detector+tracker vs. detector+tracker+zone-assignment:

| Pipeline | Avg End-to-End Latency | Zone Assignment Overhead |
|---|---:|---:|
| Detector + Tracker | 72.68 ms | — |
| Detector + Tracker + Zones | 73.75 ms | **~1.1 ms** |

Raw results: `sample_data/output/zone_benchmark.json`. As expected, zone
assignment overhead is negligible — plain-Python point-in-polygon on at
most 1-2 active tracks per frame is far cheaper than YOLO inference
(~70-90ms/frame), which remains the dominant cost by a wide margin.

### Visual validation

`scripts/run_zone_demo.py` produces `sample_data/output/zones.mp4`,
overlaying zone polygon outlines + names, track bounding boxes, track IDs,
and each track's assigned zone (`ID:N zone:zone_id`). Manual inspection
confirms: zone boundaries render at the correct pixel positions, a
visible person's box center correctly determines which zone label is
shown, and the label updates correctly as the person crosses from one
zone into another over the clip.

### Tests

```bash
python -m pytest tests/ -v
```

- `tests/test_zones.py` — 22 fast unit tests: point-in-polygon on simple
  and concave polygons, boundary behavior (edges/corners), multiple
  zones, overlapping/adjacent zones, disabled zones, track→zone mapping,
  config validation (invalid polygon, non-numeric coordinates, duplicate
  IDs, negative area), and `ZoneTransitionTracker` (previous/current
  reporting, per-track-id isolation, bounded memory).
- `tests/test_zones_integration.py` — **opt-in only**
  (`CROWDSHIELD_RUN_INTEGRATION=1`), real detector + real tracker + real
  `ZoneManager` on the real sample video against the example venue.

Full suite (excluding integration): **59 passed, 3 skipped in ~2.6s.**
One pre-existing smoke test (`test_venue_override_layers_correctly`) was
updated to match the new 3-zone example venue content (previously
asserted the old single placeholder `zone_a`).

### Known limitations

- Example venue coordinates are illustrative only, not a real venue
  calibration — no perspective transformation or real-world coordinate
  mapping exists yet.
- No density/flow/risk logic — that's Phase E/F/H. `area_m2` is stored
  but unused in D1.
- Geometric validation does not verify polygons are simple
  (non-self-intersecting); a malformed self-intersecting polygon would
  not be rejected at config-load time.
- `ZoneTransitionTracker`'s bounded eviction is a simple insertion-order
  cap, not a true LRU — acceptable for D1's purpose (exposing the raw
  transition pair) but worth revisiting if Phase E needs stricter
  recency guarantees.

## Phase E1 — Per-Zone Crowd Metrics

### Objective and scope

E1 converts C1 trajectories + D1 zone assignments into a **stateless**
per-zone snapshot: `tracks + zone assignments + transition events -> ZoneMetrics`.
No rolling windows, no synthetic data, no risk model, no SHAP, no
dashboard — those are Phase F/G/H/J/K's job. `features/crowd_metrics.py`
knows nothing about YOLO, Ultralytics, ByteTrack internals, ML models,
risk thresholds, or dashboards.

### Pixel space, not real-world speed

C1 trajectories are pixel coordinates and **no camera calibration
exists**. Every speed value is named `avg_speed_px_s` (pixels/second) —
never `m/s` or similar. `density` is the only metric with a real-world
unit at this stage, since `area_m2` is an explicitly configured physical
value (D1), unlike pixel trajectories. A dedicated test
(`test_speed_never_mislabeled_as_mps`) asserts the field name itself never
drifts toward implying calibrated units.

### Metric definitions

| Metric | Formula | Unit | Empty/insufficient-data convention |
|---|---|---|---|
| `people_count` | count of active tracks assigned to this zone | count | `0` |
| `density` | `people_count / area_m2` | people/m² | `0.0` |
| `avg_speed_px_s` | mean of per-track instantaneous pixel speed | px/s | `None` if 0 valid observations |
| `speed_variance` | population variance of per-track speeds | (px/s)² | `None` if 0 observations; `0.0` if exactly 1 (documented convention: a single observation is trivially "no spread") |
| `flow_direction_deg` | circular mean of per-track movement angles | degrees, [0, 360) | `None` if 0 valid observations |
| `direction_variance` | circular variance = `1 - R` (R = resultant vector length) | [0, 1]; 0 = fully aligned, 1 = fully dispersed | `None` if 0 valid observations; `0.0` if exactly 1 |
| `inflow_count` | tracks that transitioned INTO this zone during this one snapshot call | count (per-call, not accumulated) | `0` |
| `outflow_count` | tracks that transitioned OUT OF this zone during this one snapshot call | count (per-call, not accumulated) | `0` |

**Per-track instantaneous speed** uses the last two trajectory points and
*actual timestamps* — never an assumed `1/FPS`:
`distance((x1,y1),(x2,y2)) / (t2 - t1)`. A track needs ≥2 trajectory
points; fewer produces no speed for that track (excluded from the zone
average, not fabricated as `0`).

### Direction angle convention

Image coordinates: `x` increases rightward, `y` increases **downward**.
`angle = atan2(dy, dx) mod 360`:

```
0°   = moving right   (+x)
90°  = moving down    (+y)
180° = moving left    (-x)
270° = moving up      (-y)
```

Because `y` increases downward (not standard math/Cartesian y-up),
increasing angle from 0°→90°→180°→270° is a **clockwise** sweep as viewed
on screen — deliberately documented so no caller assumes Cartesian
semantics. A perfectly stationary track (zero displacement) has an
**undefined** direction and is excluded from zone aggregation, not
reported as an arbitrary `0°`.

### Circular statistics (flow direction + dispersion)

Arithmetic averaging of angles is wrong at the wrap-around boundary
(naively averaging 359° and 1° gives 180°, the opposite of correct).
`circular_mean_and_variance()` uses standard circular statistics instead:

```
C = mean(cos(θᵢ)),  S = mean(sin(θᵢ))
R = sqrt(C² + S²)                    # resultant length, [0, 1]
mean_direction = atan2(S, C) mod 360  # degrees
circular_variance = 1 - R             # [0, 1]
```

This correctly resolves wrap-around (359° + 1° → ~0°, verified by
`test_circular_wraparound_near_zero_not_180`) and correctly signals when
directions are too dispersed for the mean to be meaningful — two exactly
opposing tracks (0° and 180°) give `R ≈ 0`, so `direction_variance ≈ 1.0`,
which is the signal to distrust `flow_direction_deg` in that situation
rather than trusting a numerically-defined but physically meaningless
mean angle.

*(A subtle floating-point edge case was caught and fixed during
development: the wrap-around case could round to exactly `360.0` instead
of `0.0` at the `[0, 360)` boundary due to float precision in `atan2`. Both
`track_direction_deg()` and `circular_mean_and_variance()` now explicitly
clamp `360.0 → 0.0`.)*

### Inflow / outflow

Reuses D1's `ZoneTransitionTracker` rather than rebuilding transition
tracking. On every `compute_snapshot()` call, each active track's
`(previous_zone, current_zone)` is compared — **only an actual change
counts**:

- `None -> A`: `inflow_count(A) += 1`
- `A -> None`: `outflow_count(A) += 1`
- `A -> B`: `outflow_count(A) += 1` and `inflow_count(B) += 1`
- `A -> A` (no change): no event — a person staying in a zone for 20
  frames generates exactly 1 inflow event total, not 20
  (`test_no_duplicate_transitions_while_remaining_in_zone`).

`inflow_count`/`outflow_count` on a `ZoneMetrics` are **single-snapshot
counts**, not an accumulated rate — temporal aggregation into a rolling
rate/bottleneck-pressure signal is explicitly Phase F1's job, keeping E1
stateless (aside from the already-bounded `ZoneTransitionTracker`) and F1
temporal.

### Snapshot architecture

`CrowdMetricsCalculator(zone_manager, transition_tracker=None)` exposes
one method:

```python
snapshot = calculator.compute_snapshot(tracks, trajectories, timestamp_s)
# -> dict[zone_id, ZoneMetrics], one entry per ENABLED zone
```

`trajectories` is a plain `dict[track_id, list[(timestamp, x, y)]]` —
callers build it via `{t.track_id: tracker.get_trajectory(t.track_id) for t in tracks}`.
Metrics are passed data, not a `PersonTracker` reference, keeping this
layer decoupled and trivially unit-testable with synthetic data. Disabled
zones produce no entry in the returned dict at all.

### How to run

```bash
python scripts/run_metrics_demo.py \
  --video sample_data/test_video.mp4 \
  --venue config/venues/example_venue.yaml \
  --print-every 24
```

Console-only by design — E1 metrics are numeric snapshots, not a new
visual overlay; `scripts/run_zone_demo.py` (D1) remains the visual
zone-boundary tool.

### Performance

Measured on this sandbox's CPU, 100 processed frames, comparing C1
(detector+tracker) vs. C1+E1 (detector+tracker+zones+metrics):

| Pipeline | Avg End-to-End Latency | Additional Overhead |
|---|---:|---:|
| C1 (detector + tracker) | 68.50 ms | — |
| C1 + E1 (+ zone assignment + metrics) | 64.39 ms | **~-4 ms (noise-level)** |

Raw results: `sample_data/output/metrics_benchmark.json`. Consistent with
D1's zone-assignment overhead finding: plain-Python arithmetic on at most
1-2 active tracks per frame is far cheaper than YOLO's ~65-70ms/frame,
which remains the dominant cost by a wide margin — the measured
"overhead" here is within this single-core sandbox's timing noise, not a
real, reproducible cost.

### Visual/console validation

`scripts/run_metrics_demo.py` output against the bundled clip (excerpt):

```
--- t=16.00s (frame 192) ---
main_area    people=2   density=0.033 speed=46.7px/s   dir=254°   dir_var=0.19  in=0 out=0

--- t=18.00s (frame 216) ---
entrance     people=1   density=0.033 speed=58.6px/s   dir=194°   dir_var=0.00  in=0 out=0
main_area    people=1   density=0.017 speed=26.4px/s   dir=210°   dir_var=0.00  in=0 out=0
```

Manual inspection confirms: density scales correctly with `people_count`
and `area_m2`; speed/direction populate once trajectory history
accumulates and stay blank (`—`) otherwise; the 2-person `main_area`
snapshot shows a non-zero `dir_var=0.19` (moderate — two people not
moving in exactly the same direction), a real demonstration of the
circular dispersion metric on actual (if sparse) footage.

### Tests

```bash
python -m pytest tests/ -v
```

- `tests/test_crowd_metrics.py` — 30 fast unit tests: people count,
  density formula, speed (known displacement, insufficient history,
  stationary, zone average, variance with known values, single-observation
  convention, zero-observation `None`, unit-name regression guard),
  direction (all 4 cardinal angles under the documented convention,
  stationary/insufficient-history `None`), circular statistics
  (wrap-around, aligned/opposing/fully-dispersed dispersion, single-angle,
  empty), inflow/outflow (entry, exit, zone-to-zone, no-duplicate-while-
  staying), and empty/disabled zone defaults.
- `tests/test_crowd_metrics_integration.py` — **opt-in only**
  (`CROWDSHIELD_RUN_INTEGRATION=1`), real detector + tracker + zones +
  metrics on the real sample video; asserts well-formed output ranges,
  not real-world accuracy.

Full suite (excluding integration): **89 passed, 4 skipped in ~2.5s.**

### Known limitations

- Speed is pixel-space only (`avg_speed_px_s`) — no camera calibration
  exists yet, so it cannot be converted to a real-world speed.
- Inflow/outflow are single-snapshot counts; no rolling rate or
  accumulated bottleneck pressure yet (Phase F1).
- The bundled test clip is sparse (max ~2 people) — E1's numeric output
  is real (not fabricated) but should not be read as validation of
  crowd-risk-relevant behavior at real density.
- `direction_variance` near 1.0 signals `flow_direction_deg` is not
  meaningful, but the field still returns a numeric value in that case
  rather than `None` — callers must check variance before trusting
  direction.

## Phase F1 — Rolling Temporal Features + Growth/Trend Detection

### Objective and scope

F1 converts E1's stateless per-snapshot `ZoneMetrics` into short rolling
temporal windows and trend features: `ZoneMetrics(t) -> TemporalFeatureBuilder -> TemporalFeatures`.
No ML training, no synthetic data, no SHAP, no future-risk prediction, no
dashboard — those are Phase G/H/J/K's job. `features/temporal.py` does not
re-run YOLO, ByteTrack, zone assignment, or geometry; it consumes only
E1's public `ZoneMetrics`.

### Temporal windows

Exactly **5s / 15s / 30s / 60s** — schema-enforced
(`TemporalConfig.windows_seconds` must equal `[5, 15, 30, 60]`, config
loading raises otherwise). No arbitrary combinations at this stage.

### Boundary rule

An observation exactly `window_seconds` old is **included** in that
window (`elapsed <= window_seconds`, not `<`) — consistent with D1's
"boundary counts as inside" convention. Verified by
`test_boundary_observation_exactly_at_window_edge_is_included` and
`test_observation_just_past_window_edge_is_excluded`.

### Bounded memory

Each zone keeps exactly one `deque` of timestamped observations
(`_SnapshotRecord`, only the fields F1 needs — not a full `ZoneMetrics`
copy), capped at the largest configured window (60s by default, tied to
`analytics.feature_history_seconds` via a config cross-validator). Old
observations are evicted every time a new one arrives —
`test_memory_bounded_regardless_of_duration` feeds 1000 seconds of data
and confirms the retained history never exceeds ~61 samples.

### Rolling statistics (per window)

| Statistic | Definition | Empty-window convention |
|---|---|---|
| `density_mean` / `max` / `min` | over samples in the window | `None` if 0 samples |
| `speed_mean` / `min` | over samples with a valid `avg_speed_px_s` | `None` if 0 valid samples |
| `direction_variance_mean` / `max` | over samples with a valid `direction_variance` | `None` if 0 valid samples |

### Trend / slope features

Rather than a naive `last - first`, every slope uses a **lightweight
least-squares regression** over actual timestamps (`_linear_regression_slope`,
pure Python, no numpy/scipy — window sizes are small enough that this is
simple and fast):

```
slope = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²
```

| Feature | Unit | Meaning |
|---|---|---|
| `density_slope` | people/m²/s | positive = density increasing |
| `speed_slope` | pixels/s² | negative = slowing, positive = accelerating |
| `direction_variance_slope` | 1/s | positive = movement becoming more dispersed/conflicting |

Returns `None` (never a fabricated `0`) when fewer than 2 valid
observations exist, or when all observations share an identical
timestamp (slope mathematically undefined). Verified against known
synthetic sequences: increasing/decreasing/stable density
(`test_density_slope_*`), a slowing speed sequence, and an increasing
direction-variance sequence — plus a directly-checked known-line case
(`y = 2x + 1 -> slope == 2.0`).

### Inflow / outflow rate

Converts E1's per-snapshot transition counts into a rate:

```
rate = (Σ inflow_count in window) / (elapsed time spanned by the window's observations)
```

**Units: people/second.** The denominator is the *actual* time span
covered by retained observations — **not** the nominal `window_seconds`
— so a rate computed during warm-up (e.g. only 3s of data available in a
60s window) isn't artificially diluted as if it had been observed for the
full 60s. Returns `None` with fewer than 2 samples or zero elapsed span.
`net_flow_rate = inflow_rate - outflow_rate` (people/second), `None` if
either input is `None`.

### Bottleneck-pressure proxy

```
bottleneck_pressure = max(0, cumulative_inflow - cumulative_outflow) within the window
```

Explicitly a **crowd-flow proxy**, not a physically measured pressure —
documented as such wherever it appears. Unlike the rate fields, this only
needs ≥1 sample (it's a bounded cumulative sum, not a rate over time), so
it becomes available slightly earlier during warm-up. Floored at 0 per
the original architecture's definition (`max(0, inflow - outflow)`).

### Surge flag

```
surge_flag = (5s window's density_slope is not None) AND (density_slope > temporal.density_growth_threshold)
```

Uses the fastest-reacting (5s) window deliberately — a longer window
would dilute a sudden local spike. Defaults to `False` (never fabricated
`True`) when the 5s slope isn't available yet. **This is an engineering
trigger for the prototype**, not a medically or physically validated
crowd-safety threshold — interpretation/tuning is explicitly deferred to
a later phase.

### Warm-up / missing-data handling

Every `WindowFeatures` exposes `sample_count` and `coverage_s` (actual
span of retained data, capped at `window_seconds`) plus a derived
`is_mature` flag (`coverage_s >= window_seconds`, within a small
epsilon). A zone that has only been observed for 8 seconds will show its
5s window as mature but its 15s/30s/60s windows as `is_mature=False` with
`coverage_s=8.0` — never silently padded with fabricated data. Confirmed
on the real (short, ~20s) test clip: the 5s window matures early, the
15s window matures by the end of the clip, and the 30s/60s windows
**correctly remain immature for the entire clip** (`test_real_pipeline_produces_temporal_features_end_to_end`
explicitly asserts the 60s window never matures on this clip length).

### Irregular timestamps

All rates and slopes use actual timestamps, never an assumed fixed
FPS/interval. `test_irregular_timestamps_produce_correct_rates_and_slopes`
feeds an irregular sequence (`t=0, 0.7, 1.9, 3.2`) and confirms both the
slope and the inflow rate come out correct against the true elapsed time.

### Config

```yaml
temporal:
  windows_seconds: [5, 15, 30, 60]   # schema-enforced, exactly these 4
  density_growth_threshold: 0.05      # people/m²/s — engineering trigger for surge_flag
```

`AppConfig` cross-validates `analytics.feature_history_seconds >= max(temporal.windows_seconds)`
at load time — the same bounded buffer concept, not two independent
settings that could silently drift out of sync.

### How to run

```bash
python scripts/run_temporal_demo.py \
  --video sample_data/test_video.mp4 \
  --venue config/venues/example_venue.yaml \
  --print-every 24
```

### Performance

Measured on this sandbox's CPU in a single consolidated process run (200
processed frames, avoids cross-process timing noise seen in earlier
phases' benchmarks):

| Pipeline | Latency | Incremental overhead |
|---|---:|---:|
| C1 (detector + tracker) | 75.34 ms | — |
| C1 + D1 + E1 (+ zone assignment + metrics) | 75.38 ms | ~0.04 ms |
| C1 + D1 + E1 + F1 (+ temporal features) | 75.98 ms | ~0.60 ms |

Raw results: `sample_data/output/temporal_benchmark.json`. F1's own
compute time (isolated from YOLO/tracker) averaged ~0.6ms/frame across
200 frames — well under 1% of the ~75ms YOLO-dominated pipeline latency,
consistent with the "tiny compared with YOLO inference" target. As with
D1/E1, subtracting independently-run full-pipeline benchmarks proved too
noisy on this single-core sandbox to be meaningful on its own — the
consolidated single-process measurement above is the more trustworthy
number.

### Visual/console validation

`scripts/run_temporal_demo.py` output against the bundled clip (excerpt,
`main_area` zone at t=16.00s):

```
main_area (surge_flag=False):
   5s [mature        ] n=61  density_slope=+0.006  speed_slope=-109.79 pressure=2.0
  15s [mature        ] n=181 density_slope=-0.000  speed_slope=-1.65   pressure=2.0
  30s [warmup(16.0s) ] n=193 density_slope=+0.000  speed_slope=-1.65   pressure=2.0
  60s [warmup(16.0s) ] n=193 density_slope=+0.000  speed_slope=-1.65   pressure=2.0
```

Manual inspection confirms: `sample_count` grows correctly and matches
the expected ~12 samples/sec × window_seconds once mature (61 ≈ 12×5);
the 30s/60s windows correctly stay marked `warmup` throughout this ~20s
clip; `speed_slope` shows real (if noisy, given sparse tracking)
deceleration values; `surge_flag` correctly stays `False` throughout,
consistent with this clip's gentle, non-surging density pattern.

### Tests

```bash
python -m pytest tests/ -v
```

- `tests/test_temporal.py` — 31 fast unit tests: window boundary
  inclusion/exclusion, correct rolling behavior across all 4 windows,
  eviction, bounded memory, density/speed/direction-variance slopes
  (increasing/decreasing/stable/known-line), the regression helper
  directly, inflow/outflow/net-flow rates, bottleneck pressure
  (including the single-sample and floor-at-zero cases), surge flag
  (true/false/warm-up-default), warm-up maturity marking, irregular
  timestamps, empty history, and `TemporalFeatureBuilder` multi-zone
  routing.
- `tests/test_temporal_integration.py` — **opt-in only**
  (`CROWDSHIELD_RUN_INTEGRATION=1`), full real A1-F1 pipeline on the real
  sample video; asserts history grows, short windows mature before long
  ones (and confirms the 60s window correctly never matures on this short
  clip), and internal history size stays bounded.

Full suite (excluding integration): **120 passed, 5 skipped in ~2.7s.**

### Known limitations

- `surge_flag` and `density_growth_threshold` are prototype engineering
  triggers, not clinically/physically validated crowd-safety thresholds.
- Speed-derived features remain pixel-space (`speed_slope` in
  pixels/s²) — no camera calibration exists yet (inherited from E1).
- The bundled ~20s test clip is too short to ever mature the 30s/60s
  windows — confirmed, not worked around; longer real footage will be
  needed to validate those windows' behavior.
- F1 assumes non-decreasing timestamps from the caller (consistent with a
  real-time processing loop) — out-of-order timestamps are not defended
  against.
- No synthetic risk-model training data or ML model exists yet (Phase
  G/H).

## Phase G1 — Synthetic Crowd-Risk Scenario Generator

### ⚠️ Synthetic data disclaimer

> **Synthetic data is an engineering aid for developing and testing the
> risk model. It is NOT real-world crowd-safety ground truth and does
> not establish clinically or operationally validated thresholds.**

Every generated row carries a literal `data_source = "synthetic"` column
so this can never be silently forgotten downstream. Real public crowd
datasets are useful for detection/counting/trajectory work but don't
provide enough labeled examples of dangerous crowd *escalation* — G1
exists purely to give the future H1 risk model plausible, controlled,
reproducible training examples of that escalation dynamic.

### What G1 does NOT do

No fake video, no YOLO, no ByteTrack, no bounding boxes. G1 generates
**feature-level** time series directly at the E1/F1 output shape —
`ScenarioConfig -> synthetic ZoneMetrics per timestamp -> real F1 TemporalFeatureBuilder -> flat feature row`.

### Key design decision: reusing real F1 code

Rather than reimplementing slope/rate/window math for synthetic data, the
generator constructs a real `features.crowd_metrics.ZoneMetrics` object
at each synthetic timestamp and feeds it through a real
`features.temporal.TemporalFeatureBuilder` — the **exact same code path**
the live A1-F1 pipeline uses. This guarantees the synthetic dataset's
`w5_density_slope`, `w15_bottleneck_pressure`, etc. columns are formula-
and schema-identical to what the real pipeline would produce, with zero
duplicated logic to keep in sync.

### The six scenarios

| Scenario | Label | Characteristics |
|---|---|---|
| `NORMAL_STABLE` | SAFE | low/moderate density, stable, balanced in/outflow |
| `GRADUAL_BUILDUP` | LOW | density gradually rising, speed gradually falling, inflow > outflow |
| `RAPID_SURGE` | MODERATE | density rising fast, strong inflow imbalance, surge_flag typically active |
| `BOTTLENECK` | HIGH | sustained inflow > outflow, rising density, rising bottleneck pressure |
| `CONFLICTING_FLOW` | HIGH | elevated/rising direction variance, density need NOT be extreme |
| `CRITICAL_ESCALATION` | CRITICAL | already-elevated density, rapid growth, strong imbalance, high conflict |

Full numeric parameters (all explicit engineering assumptions) live in
`data/scenarios.yaml`. `CONFLICTING_FLOW` and `BOTTLENECK` are both
labeled HIGH but reach it through different mechanisms — this is
deliberate (spec: "danger should not depend solely on density") and is
visible in the generated plots (see below).

### Generated features

Per timestamp: the full E1 `ZoneMetrics` shape (`people_count`,
`density`, `avg_speed_px_s`, `speed_variance`, `flow_direction_deg`,
`direction_variance`, `inflow_count`, `outflow_count`) plus the full F1
`TemporalFeatures` shape flattened per window (`w5_`/`w15_`/`w30_`/`w60_`
prefixed: sample_count, coverage_s, is_mature, density/speed/direction-
variance rolling stats, slopes, flow rates, bottleneck_pressure) plus
`surge_flag`, `label`, `sequence_id`, `scenario`, `zone_id`,
`timestamp_s`, and `data_source`.

### Physical/engineering consistency

Values aren't independent random columns — `density_trend`,
`speed_trend_px_s2`, `inflow_rate`, `outflow_rate` are coupled per
scenario so that (e.g.) `GRADUAL_BUILDUP` genuinely produces density↑ +
speed↓ + inflow>outflow together, not as separate unrelated signals. See
`data/scenarios.yaml` for exact values.

### Noise

Controlled Gaussian noise (density, speed, direction, direction_variance)
and Poisson-sampled integer inflow/outflow counts, scaled by each
scenario's `noise_level`. All values are clipped to physically sensible
bounds afterward: `density >= 0`, `speed >= 0`, `direction ∈ [0, 360)`,
`direction_variance ∈ [0, 1]`, counts `>= 0` — verified by
`test_bounds_hold_across_full_dataset`.

### Reproducibility

Every RNG is `numpy.random.default_rng(seed)`, and every sequence's seed
is deterministically derived from `base_seed` plus a fixed counter —
never from wall-clock time. Same `scenarios.yaml` + same `base_seed` →
byte-identical dataset (`test_same_seed_produces_identical_dataset`); a
different seed produces a different realization
(`test_different_seed_produces_different_realization`).

### Controlled overlap between classes (avoiding trivial labels)

Two mechanisms, deliberately, instead of an artificial "make the noise
big enough" hack:
1. **Natural per-timestep noise** (above) means no scenario is a perfectly
   flat line — a `NORMAL_STABLE` sequence can transiently look a little
   like `LOW`, and vice versa.
2. **Scripted transitions** (below): the label switches at
   `transition_start_s`, but the underlying feature *values* continue
   blending from scenario A's regime into scenario B's over
   `transition_duration_s` — so for a short window after the label
   changes, the features still partially resemble the previous class.
   This is the primary, intentional source of near-boundary class
   overlap, verified by `test_transition_features_blend_smoothly_not_jump`.

The generator was **not** tuned to maximize feature/label separability —
see "Correlation sanity check" below and the `Do NOT optimize for model
score yet` principle carried over from the spec.

### Scenario transitions

Six scripted transition pairs (`data/scenarios.yaml`), e.g.
`NORMAL_STABLE -> GRADUAL_BUILDUP`, `RAPID_SURGE -> CRITICAL_ESCALATION`.
For each transition realization, `transition_start_s` is drawn from a
dedicated seeded RNG (uniform in `[0.25×duration, 0.6×duration]`) so
different realizations of the same transition pair happen at different
times — not identical copies. **Labeling rule:** the label switches to
scenario B's label immediately at `transition_start_s`; the feature
values keep blending smoothly for `transition_duration_s` afterward
(scenario A's own trend line keeps running on its own clock up to that
point; scenario B's trend line starts fresh, i.e. "as if B began now").
**No leakage:** the blend weight and both scenarios' local clocks depend
only on the current timestamp and the (known-in-advance) transition
start — never on any future value.

### Dataset format

CSV, one row per (timestamp × sequence). `data/scenarios.yaml`'s default
settings (8 pure realizations/scenario × 6 scenarios + 6 transitions × 6
realizations = 84 sequences × 120 rows) produce **10,080 rows across 84
sequences**, a 12MB CSV — well under the "tens of thousands of rows, not
millions" / "not hundreds of MB" targets. A companion `metadata.json`
records `generator_version`, `base_seed`, `duration_s`,
`sample_interval_s`, `windows_seconds`, scenario/transition lists, label
counts, split counts, and the disclaimer text above.

### Train / validation / test split

`split_dataset()` assigns a `split` column **by whole `sequence_id`**,
never by individual row — splitting rows from the same sequence across
train/val/test would leak temporally adjacent information. Verified by
`test_split_no_sequence_appears_in_multiple_splits`. Default split on the
84-sequence dataset: **59 train / 13 val / 12 test** sequences (7,080 /
1,560 / 1,440 rows) — deterministic for a fixed split seed
(`test_split_is_deterministic_for_same_seed`).

### Class balance

| Label | Rows | % |
|---|---:|---:|
| SAFE | 1,601 | 15.9% |
| LOW | 2,017 | 20.0% |
| MODERATE | 1,672 | 16.6% |
| HIGH | 3,005 | 29.8% |
| CRITICAL | 1,785 | 17.7% |

No class is dominant (spec's "SAFE=70%, CRITICAL=2%" failure mode
explicitly avoided) — HIGH is largest only because two pure scenarios
(`BOTTLENECK`, `CONFLICTING_FLOW`) and two transition targets map to it.
`test_class_counts_reasonably_balanced` enforces every label stays
between 5% and 50% of rows.

### Future labels (5-min / 10-min)

`future_5min_label`/`future_10min_label` are computed by shifting each
sequence's own `label` column by 300s/600s (`pandas` `.shift()`, grouped
per sequence — reads only a *later* row's already-known label into a
*new* column at the earlier row, never touches existing feature columns).
With the default 120s sequence duration, both are `NaN` for essentially
every row (120s « 300s) — this is **correct, not a bug**: a 120s sequence
genuinely doesn't contain information about what happens 5 minutes later.
Verified directly with a 700s test sequence
(`test_future_labels_correctly_shifted`), confirming the shift is exactly
right including across a scripted transition (i.e. `future_10min_label`
correctly shows the *later*, different-regime label). H1/I1 can generate
longer sequences via `duration_s` in `scenarios.yaml` if non-null future
labels are needed for forecasting model training.

### Correlation sanity check (not optimized for)

Per-label feature means, generated (not tuned) — from
`scripts/analyze_synthetic_data.py`:

| label | density | speed (px/s) | direction_variance | w5 net_flow | w5 bottleneck |
|---|---:|---:|---:|---:|---:|
| SAFE | 0.50 | 60.0 | 0.050 | 0.014 | 0.76 |
| LOW | 1.24 | 48.3 | 0.124 | 0.187 | 1.38 |
| MODERATE | 4.76 | 31.1 | 0.195 | 0.976 | 4.78 |
| HIGH | 2.06 | 34.9 | 0.389 | 0.462 | 2.63 |
| CRITICAL | 4.98 | 13.7 | 0.497 | 1.393 | 6.90 |

Density and speed are **not** perfectly monotonic across SAFE→CRITICAL
(HIGH's density is lower than MODERATE's) — this is the intended,
deliberate consequence of `CONFLICTING_FLOW` reaching HIGH via
`direction_variance` rather than density. `direction_variance` itself
**is** monotonically increasing, exactly reflecting that design.

### Data leakage checks

Explicitly tested: future labels never enter current feature columns
(`test_no_leakage_current_features_unaffected_by_future_label_columns`);
sequence-based split integrity (above); deterministic seeding (above);
generated labels are a genuine target column, never duplicated as an
input feature.

### How to run

```bash
python scripts/generate_synthetic_data.py \
  --scenarios data/scenarios.yaml \
  --output-dir sample_data/synthetic \
  --seed 42

python scripts/analyze_synthetic_data.py \
  --dataset sample_data/synthetic/synthetic_dataset.csv \
  --plots-dir sample_data/synthetic/plots
```

`analyze_synthetic_data.py`'s plotting is optional (wrapped in a
try/except `ImportError` for `matplotlib`, a dev-only dependency) — the
text-based class distribution and correlation sanity report always runs.

### Performance

Full default dataset (84 sequences, 10,080 rows) generates in **~2.2s**,
CPU-only, no torch/YOLO/OpenCV/GPU involved — `data/synthetic_generator.py`
imports only `numpy`, `pandas`, and the existing `features.crowd_metrics`
/ `features.temporal` modules. ~10MB in memory, 12MB on disk as CSV.

### Tests

```bash
python -m pytest tests/ -v
```

`tests/test_synthetic_generator.py` — 24 fast unit tests: schema
validation (unknown scenario name, missing scenario, bad split
fractions), determinism, seed variation, per-scenario bounds and
semantics (`GRADUAL_BUILDUP` density trend, `BOTTLENECK` net flow,
`CONFLICTING_FLOW` direction variance, `RAPID_SURGE` triggers
`surge_flag`), label coverage and balance, unique sequence IDs,
transition label-switch timing and smooth feature blending, split
integrity/determinism/rejection of bad fractions, future-label shift
correctness (including across a transition) and near-end `NaN`
behavior, and the no-leakage regression guard.

Full suite (excluding integration): **144 passed, 5 skipped in ~9s**
(two G1 tests generate the full default dataset, ~2s each — the rest are
sub-100ms with a tiny config fixture).

### Known limitations

- Synthetic data is engineering data, not validated crowd-safety ground
  truth (repeated deliberately — see disclaimer above).
- Scenario numeric parameters (`data/scenarios.yaml`) are explicit
  engineering assumptions, not calibrated against real incident data.
- With the recommended 120s sequence duration, future 5-min/10-min labels
  are `NaN` for virtually every row — correct given the duration, but
  means this dataset as-shipped isn't yet useful for forecasting-model
  training without regenerating with a longer `duration_s`.
- Each sequence models exactly one synthetic zone
  (`zone_id = "synthetic_zone"` by default) — no multi-zone interaction
  is simulated within a single sequence.
- Direction (`flow_direction_deg`) is linearly blended during a
  transition, which is a simplification (real circular quantities should
  arguably blend via circular interpolation) — acceptable since
  `direction_variance`, not the absolute direction value, is the
  feature that actually matters for risk.
- No risk model exists yet to consume this dataset (Phase H1).

## Phase H1 — Baseline Risk Model Training

### ⚠️ Critical limitation — read this first

> **H1 demonstrates that the engineered feature space can support
> risk-state classification on the synthetic scenarios produced by G1.
> It does NOT demonstrate real-world crowd-safety accuracy.**

The model is trained **only** on G1's synthetic data. A 95%+ test
accuracy here means the model learned the synthetic generator's encoded
relationships — it says nothing about real-world crowd-safety accuracy,
and this is **not** a deployment-ready model. Real-video validation has
not been performed and is out of scope for Milestone 1.

### Dataset used

`sample_data/synthetic/synthetic_dataset.csv` (G1's shipped dataset,
unmodified): 10,080 rows, 84 sequences. The existing `split` column
(sequence-based, from G1) is used exactly as-is — H1 does **not**
reshuffle rows or create a new split.

| Split | Rows | Sequences |
|---|---:|---:|
| train | 7,080 | 59 |
| val | 1,560 | 13 |
| test | 1,440 | 12 |

### Feature selection

76 features from the F1/E1 schema: the 8 direct `ZoneMetrics` fields
(`people_count`, `density`, `avg_speed_px_s`, `speed_variance`,
`direction_variance`, `inflow_count`, `outflow_count`, `surge_flag`) plus
17 fields × 4 windows (`w5_`/`w15_`/`w30_`/`w60_`-prefixed rolling stats,
slopes, flow rates, bottleneck pressure). Explicitly excluded from `X`
(see `models/features.py::EXCLUDED_COLUMNS`, the single source of truth
every leakage check reads from): `sequence_id`, `scenario`, `zone_id`,
`timestamp_s`, `data_source`, `split`, `label`, `future_5min_label`,
`future_10min_label`.

**`flow_direction_deg` is deliberately excluded** (spec's "Option B"):
in this single-synthetic-zone setup the absolute direction has no
inherent risk meaning (an arbitrary axis, not oriented to a real venue),
and `direction_variance` already captures the risk-relevant "conflicting
movement" signal on its own — simpler and equally defensible as a sin/cos
encoding, without needing to justify a circular-encoding choice for a
feature we weren't going to rely on anyway.

### Models evaluated

1. **Logistic Regression** — `SimpleImputer(median) -> StandardScaler -> LogisticRegression`
   in a single sklearn `Pipeline`, fit only on train data (no val/test
   leakage into the scaler/imputer). Minimal `C` sweep: `{0.1, 1.0, 10.0}`.
2. **HistGradientBoostingClassifier** — chosen over RandomForest
   specifically because it **handles F1's warm-up NaN values natively**,
   with no imputation step required — a genuine, documented reason for
   the choice, not an arbitrary pick. Minimal `max_leaf_nodes` sweep:
   `{15, 31}`. XGBoost was not introduced — sklearn's built-in HGB was
   sufficient and keeps the dependency footprint unchanged.

### Multi-class target & ordinal mapping

```
SAFE=0, LOW=1, MODERATE=2, HIGH=3, CRITICAL=4
```

Standard multiclass classification (not ordinal-specific modeling) for
H1, per the spec. This fixed order (`models/features.py::LABELS_ORDERED`)
is used everywhere — critically, **not** the same as sklearn's internal
alphabetical class ordering (`CRITICAL, HIGH, LOW, MODERATE, SAFE`).
`RiskModel.predict_proba()` always reindexes sklearn's output into
`LABELS_ORDERED` columns before returning, so every downstream
consumer (risk score, log loss, tests) works with one consistent order.

### 🐛 Bug caught and fixed during development

`sklearn.metrics.log_loss` **silently mis-scores probabilities** when
given a non-lexicographic `labels=` order — it emits a `UserWarning` but
does **not** actually reorder the probability columns to match. Verified
with a hand-built example (expected log loss 0.29, sklearn returned 2.30
when naively passed our ordinal label order). **Fix:** `evaluation/metrics.py::compute_all_metrics`
now explicitly reorders probability columns into lexicographic order
before calling `log_loss`, with a regression test
(`test_log_loss_correct_with_ordinal_label_order`) locking in the fix
against the hand-computed expected value. (`precision_score`/
`recall_score`/`f1_score` were independently verified to correctly
respect a custom `labels=` order — only `log_loss` had this issue.)

### Risk score (0-100 display value)

```
risk_score = 100 * Σ( P(class_i) * ordinal_i / 4 )   for i in [SAFE..CRITICAL]
```

**This is a decision-support display number, NOT a probability of
injury, NOT a probability of a stampede, and NOT a physically validated
danger percentage.** 0 → fully SAFE expectation, 100 → fully CRITICAL
expectation.

### Required metrics

Accuracy, macro F1, weighted F1, balanced accuracy, per-class
precision/recall, confusion matrix — all via `evaluation/metrics.py`,
which wraps sklearn's implementations plus three CrowdShield-specific
safety metrics that don't exist in sklearn:

| Metric | Definition | Why it matters |
|---|---|---|
| `dangerous_recall` | Of all true HIGH/CRITICAL, how many were caught? | Missing a dangerous state is unacceptable regardless of accuracy |
| `dangerous_precision` | Of all predicted HIGH/CRITICAL, how many were real? | Excess false alarms erode operator trust |
| `false_safe_rate` | Of all true HIGH/CRITICAL, how many were predicted SAFE/LOW? | **The most safety-critical failure mode** |
| `ordinal_mae` | Mean absolute error on SAFE=0..CRITICAL=4 | Distinguishes a HIGH→MODERATE miss from the far worse HIGH→SAFE miss |

`dangerous = {HIGH, CRITICAL}`, defined once in `models/features.py::DANGEROUS_LABELS`.

### Selection criteria (safety-first, in priority order)

```
1. dangerous_recall (higher better)
2. macro_f1 (higher better)
3. false_safe_rate (lower better)
4. inference latency (lower better)
5. model size (lower better)
6. accuracy (higher better)
```

Implemented as an explicit lexicographic sort
(`scripts/train_risk_model.py::select_best`) — never picks on accuracy
alone.

### Validation comparison (all 5 candidates)

| Model | Accuracy | Macro F1 | Dangerous Recall | False-Safe Rate | Latency | Size |
|---|---:|---:|---:|---:|---:|---:|
| LogisticRegression (C=0.1) | 0.9417 | 0.9367 | 0.9776 | 0.0138 | 3.00 ms | 8.6 KB |
| LogisticRegression (C=1.0) | 0.9513 | 0.9431 | 0.9828 | 0.0103 | 3.00 ms | 8.6 KB |
| LogisticRegression (C=10.0) | 0.9468 | 0.9439 | 0.9828 | 0.0086 | 3.07 ms | 8.6 KB |
| HistGradientBoosting (15 leaves) | 0.9891 | 0.9831 | 0.9845 | 0.0052 | 7.87 ms | 935.1 KB |
| **HistGradientBoosting (31 leaves)** | **0.9878** | **0.9824** | **0.9862** | **0.0052** | 7.45 ms | 1427.4 KB |

### Selected model: HistGradientBoosting (31 leaves)

Highest validation `dangerous_recall` (0.9862) among all 5 candidates,
with the lowest `false_safe_rate` (0.0052) tied with the 15-leaf variant
— selected over the 15-leaf variant on the next tiebreaker
(`dangerous_recall`). Both HGB variants substantially beat every Logistic
Regression configuration on every safety metric; the ~1.4MB model size
and ~7.5ms single-sample latency were judged acceptable trade-offs for
the safety improvement, consistent with the priority order above (safety
metrics outrank latency/size).

### Test-set evaluation (evaluated exactly once)

| Metric | Value |
|---|---:|
| Accuracy | 0.9521 |
| Macro F1 | **0.7711** ⚠️ see below |
| Weighted F1 | 0.9573 |
| Balanced Accuracy | 0.9642 |
| Dangerous Recall | 0.9296 |
| Dangerous Precision | 1.0000 |
| False-Safe Rate | 0.0505 |
| Ordinal MAE | 0.0764 |
| Log Loss | 0.4611 |

### ⚠️ Investigated finding: test macro F1 is deflated by zero MODERATE test coverage

The test split (12 sequences, fixed by G1, used as-is per the spec) **has
zero MODERATE-labeled rows by chance** — none of `RAPID_SURGE`'s pure
sequences or the transitions passing through MODERATE happened to land in
the 12-sequence test split. `MODERATE` precision/recall are therefore
both `0.0` (no support at all, not a false prediction), which single-
handedly drags `macro_f1` down from 0.98 (validation) to 0.77 (test) —
this is **not** a modeling failure. Evidence this is a coverage artifact,
not a generalization problem:
- `weighted_f1` (which down-weights classes with no support) stays high
  and consistent with validation: **0.9573**.
- `dangerous_recall`/`dangerous_precision`/`false_safe_rate` — none of
  which depend on MODERATE — all remain strong and consistent with
  validation.
- The confusion matrix's MODERATE row and column are entirely zero (no
  MODERATE ground truth to evaluate against), confirming this is a
  **coverage gap**, not a misclassification pattern.

Per the spec ("use the splits already created by G1... do not create a
new random split"), this was **not** worked around by re-splitting or
regenerating G1's dataset — the shipped `split_seed=7` dataset is used
exactly as delivered. **Recommendation for a future revision:** increase
`realizations_per_scenario`/`realizations_per_transition` in
`data/scenarios.yaml` (more test sequences reduces the chance of a
zero-coverage class) — flagged here, not implemented, since G1 is locked
for this phase.

### Feature importance (top 10, selected model)

Computed via `sklearn.inspection.permutation_importance` (macro-F1 drop,
5 repeats, evaluated on val) — `HistGradientBoostingClassifier` has no
native `.feature_importances_`, unlike `RandomForestClassifier`, so
permutation importance is the correct choice here, not an arbitrary one:

| Rank | Feature | Importance (macro-F1 drop) |
|---:|---|---:|
| 1 | `direction_variance` | 0.10338 |
| 2 | `w60_direction_variance_mean` | 0.01042 |
| 3 | `w30_density_min` | 0.00589 |
| 4 | `w30_direction_variance_slope` | 0.00390 |
| 5 | `speed_variance` | 0.00362 |
| 6 | `w60_inflow_rate` | 0.00299 |
| 7 | `w60_density_min` | 0.00294 |
| 8 | `w5_density_min` | 0.00252 |
| 9 | `w60_speed_min` | 0.00239 |
| 10 | `w15_density_slope` | 0.00184 |

`direction_variance` dominates by an order of magnitude over every other
feature — consistent with `CONFLICTING_FLOW`'s design (elevated
direction variance at only moderate density) being a strong, learnable
signal, and with G1's design goal that "danger should not depend solely
on density." This was **not** forced or expected in advance — it's an
observed result, reported as found.

### Inference performance

| | Single-sample latency | Batch (test set, 1,440 rows) |
|---|---:|---:|
| Selected model (HGB, 31 leaves) | 7.45 ms | ~0.03 ms/row |

Model artifact: **1,427.4 KB** (`models/artifacts/risk_model.pkl`).
Single-sample latency is higher than the sub-millisecond target for
Logistic Regression, but still comfortably fast enough for CrowdShield's
~1Hz risk-inference rate (Phase K's real-time budget) — the safety-metric
improvement was judged to outweigh the latency cost per the selection
priority order.

### Serialization & portability fix

`RiskModel.save()` writes **two** sibling files: a `joblib`-serialized
fitted sklearn `Pipeline` (`risk_model.pkl`) and a plain-JSON metadata
sidecar (`risk_model.pkl.meta.json`) holding feature names, model
version, sklearn's class order, and — critically — the exact
**scikit-learn version used to train the model**.

**Real bug found during cross-machine validation:** the original
single-file `pickle`-based design, combined with `requirements.txt`'s
loose `scikit-learn>=1.4` pin, let a fresh install on another machine
silently resolve a *different* sklearn version than the one used to
train the shipped artifact. `HistGradientBoostingClassifier` internally
holds a reference to a private loss-function object
(`sklearn._loss.loss.HalfMultinomialLoss`) whose module path is not
stable across sklearn versions — unpickling under a different version
failed with a cryptic `ModuleNotFoundError: No module named '_loss'`.

**Fix (two parts):**
1. `requirements.txt` now pins scikit-learn to an **exact version**
   (`scikit-learn==1.8.0`), not a lower bound — a deliberate, documented
   exception to the project's general "loose lower-bound" dependency
   policy, since sklearn does not guarantee pickle/joblib compatibility
   across versions. Any future intentional upgrade of this pin must also
   regenerate the model artifact.
2. `RiskModel.load()` reads the JSON metadata sidecar **first** (always
   succeeds — it's plain JSON, no pickle involved regardless of sklearn
   version) and compares the recorded training-time version against the
   installed one. A mismatch emits an immediate `RuntimeWarning`; if the
   subsequent pipeline load then fails, the error is re-raised as a
   `RuntimeError` naming both versions and the exact fix
   (`pip install scikit-learn==<trained_version>`) instead of a bare
   traceback.

**Trust note:** `joblib.load` (like `pickle.load`) can execute arbitrary
code for an untrusted file — this is safe here because `risk_model.pkl`
is this project's own generated artifact
(`scripts/train_risk_model.py`), not sourced externally. Never load a
model artifact from an untrusted origin.

**Honest verification scope:** this sandbox has exactly one Python/
scikit-learn installation, so a genuine cross-version load failure could
not be literally reproduced here (that requires two different sklearn
installations, e.g. an actual second machine). What *was* verified: the
version-mismatch **detection** mechanism fires correctly (tampered
metadata → `RuntimeWarning` naming the mismatch,
`test_load_warns_on_sklearn_version_mismatch`), and a genuine load
failure is wrapped into a clear, actionable `RuntimeError` rather than a
raw traceback (`test_load_wraps_actual_load_failure_with_clear_message`,
simulated by corrupting the pipeline file's bytes). The **primary** fix
— pinning the exact sklearn version — eliminates the root cause for any
future install that follows `requirements.txt`, which is the durable
guarantee; the version-check/clear-error mechanism is the defense-in-depth
layer for if that pin is ever bypassed or intentionally changed without
regenerating the artifact.

### How to run

```bash
python scripts/train_risk_model.py \
  --dataset sample_data/synthetic/synthetic_dataset.csv \
  --output-dir models/artifacts
```

Prints the full leakage audit, validation comparison table, test metrics,
confusion matrix, and top-10 feature importances; saves
`models/artifacts/risk_model.pkl` and `models/artifacts/model_report.json`.

### Tests

```bash
python -m pytest tests/ -v
```

- `tests/test_metrics.py` — 13 tests: dangerous recall/precision,
  false-safe rate, ordinal MAE (including the "HIGH→SAFE worse than
  HIGH→MODERATE" distinction), `compute_all_metrics` structure, and the
  `log_loss` bug regression guard.
- `tests/test_risk_model.py` — 19 tests: feature exclusion/leakage,
  boolean casting, ordinal mapping, training, prediction validity,
  `predict_proba` column order and sum-to-1, risk-score bounds and
  formula correctness, missing-feature error handling, save/load
  round-trip (identical predictions), determinism, and 4 portability-fix
  regression tests (metadata sidecar written correctly, missing-sidecar
  error, version-mismatch warning, wrapped load-failure error message).
- `tests/test_split_integrity.py` — 6 tests against the **actual packaged
  G1 dataset**: sequence-ID disjointness across splits, every row
  assigned, no excluded/future-label columns in the feature set.
- `tests/test_risk_model_scenarios.py` — 7 tests: manual scenario sanity
  checks (spec section 26) — trains a small model and asserts *relative*
  risk-score ordering (`CRITICAL_ESCALATION` highest, `NORMAL_STABLE`
  lowest, `CONFLICTING_FLOW` elevated despite moderate density) — no
  hardcoded exact scores.
- `tests/test_packaged_model.py` — 5 tests against the actual shipped
  `models/artifacts/risk_model.pkl`, including a check that its metadata
  sidecar's recorded sklearn version matches the currently installed one.

Full suite (excluding integration): **194 passed, 5 skipped in ~14s.**

### Known limitations

- **Trained only on synthetic data** — does not demonstrate real-world
  crowd-safety accuracy; real-video validation is out of scope for
  Milestone 1.
- Test-set macro F1 is deflated by a zero-MODERATE-coverage artifact of
  the small (12-sequence) test split — investigated and documented above,
  not a generalization failure, but a legitimate reason to increase G1's
  realization counts in a future revision.
- Scenario parameters (G1) are engineering assumptions, not calibrated
  against real incident data — the model can only be as good as the
  synthetic relationships it was trained on.
- HistGradientBoosting's ~7.5ms single-sample latency is higher than
  Logistic Regression's ~3ms — acceptable for this application's ~1Hz
  inference rate, but worth knowing if latency requirements tighten later.
- No calibration curve was added — `log_loss` was inspected as the
  probability-quality signal, per the spec's "don't add calibration
  unless it materially improves the model" guidance.
- No ordinal-specific classifier was used (e.g. ordinal logistic
  regression) — standard multiclass classification, per spec, with
  `ordinal_mae` as the ordinal-awareness metric instead.
- `requirements.txt` now pins scikit-learn to an exact version
  (`==1.8.0`) rather than a lower bound — intentional (see "Serialization
  & portability fix" above), but means upgrading scikit-learn requires
  also regenerating `models/artifacts/risk_model.pkl`, not just editing
  the version string.
- Cross-sklearn-version load-failure recovery was verified via the
  detection/error-wrapping mechanism, not via an actual second
  scikit-learn installation (this sandbox only has one) — see
  "Serialization & portability fix" for the exact, honest verification
  scope.

## Phase I1 — 5-Minute / 10-Minute Future-Risk Prediction

### ⚠️ Critical limitation — read this first

> **I1 is a MACHINE-LEARNING FORECASTING FEATURE trained on G1's
> synthetic temporal scenarios. It is NOT a physical or clinical
> prediction, and it does NOT demonstrate real-world crowd-safety
> forecasting accuracy.**

Same caveat as H1, extended to the future horizon: the model has only
ever seen synthetic scenario trajectories, never real crowd footage.

### Approach: direct future-label classification (Approach A)

Three independently trained `HistGradientBoostingClassifier` models,
sharing the exact same feature representation as H1 — **not** an
LSTM/GRU/Transformer sequence model, per the spec's explicit lightweight
requirement:

| Model | Target | Trained on rows where |
|---|---|---|
| `current` | `label` (state at t) | always defined |
| `future_5m` | `future_5min_label` (state at t+300s) | t+300s exists within the same sequence |
| `future_10m` | `future_10min_label` (state at t+600s) | t+600s exists within the same sequence |

### Key finding: G1 already implements leakage-safe future labels

Before writing anything new, I1 inspected `data/synthetic_generator.py`
and found `future_5min_label`/`future_10min_label` **already generated
correctly** — each sequence's `DataFrame` is built and `.shift()`-ed
independently before concatenation (see G1's `generate_sequence()`), so
a future label can never be pulled from a different sequence, and rows
too close to a sequence's end already get `NaN` (never a fabricated
label). **I1 added nothing to this mechanism** — no new leakage-exclusion
logic, no new label-generation code. `models/future_risk.py` only:
1. filters out rows where the target horizon's label is `NaN` before
   training/evaluating (`prepare_future_features()`), and
2. reuses `models.features.get_feature_columns()`/`prepare_features()`
   completely unchanged — these already exclude both future-label columns
   from `X`, so there was no new exclusion logic to write either.

### Why a new dataset was needed (not a new generator)

G1's shipped 120s dataset has **zero** non-null future labels (120s <
300s horizon) — documented in G1's own README, not a bug. I1 regenerates
a **longer-duration** dataset using the *exact same* `data/scenarios.yaml`
and generator, via new (additive, backward-compatible) CLI flags on
`scripts/generate_synthetic_data.py`:

```bash
python scripts/generate_synthetic_data.py \
  --output-dir sample_data/synthetic_long \
  --duration 900 --realizations-per-scenario 4 --realizations-per-transition 3 \
  --seed 42 --split-seed 7
```

Result: **37,800 rows, 42 sequences**, 900s each — `future_5min_label`
coverage **66.7%**, `future_10min_label` coverage **33.3%** (both,
correctly, less than 100%: the last 300s/600s of every sequence has no
valid target, by construction — verified directly by
`test_last_300s_has_no_valid_5m_target` /
`test_last_600s_has_no_valid_10m_target`).

### Temporal leakage — explicitly audited, not assumed

Beyond reusing G1's already-safe mechanism, I1 adds its own independent
verification, since this is explicitly the most important part of the
spec:
- `scripts/train_future_risk.py` runs a **future-label provenance audit**
  before training: for a sample of rows, it looks up the actual row at
  `t+300s` in the *same* `sequence_id` and asserts its `label` matches
  the row's `future_5min_label` exactly.
- `test_future_label_never_crosses_sequence_boundary` — for every
  sequence, rows within 300s of that sequence's own end must have a
  `NaN` `future_5min_label` (if leakage occurred, a label could be
  incorrectly pulled from a different, longer-running sequence).
- `test_prepare_future_features_X_excludes_targets` — `X` never contains
  `label`, `future_5min_label`, or `future_10min_label`, for any horizon.
- The existing sequence-based train/val/test split (from G1, reused
  as-is — I1 never reshuffles rows or creates a new split) is
  re-verified via `leakage_audit()` (same function H1 uses, now shared
  via `models/training_utils.py`).

### Future-risk prediction API

Extends `RiskModel` rather than inventing a parallel system —
`models/future_risk.py::MultiHorizonRiskModel` bundles three `RiskModel`
instances and loads them **once**, reused for every prediction (never
reloaded per-sample):

```python
bundle = MultiHorizonRiskModel.load("models/artifacts")
result = bundle.predict(X)
# {
#   "current":    {"risk_class": [...], "risk_score": [...], "class_probabilities": [...], "dangerous_probability": [...], "horizon_seconds": 0.0, ...},
#   "future_5m":  {..., "horizon_seconds": 300.0, ...},
#   "future_10m": {..., "horizon_seconds": 600.0, ...},
# }
```

`dangerous_probability` = `P(HIGH) + P(CRITICAL)` from each horizon's own
`predict_proba()`. The risk score reuses **the exact same H1 formula**
(`risk_score_from_proba()`, factored out of `RiskModel` so both H1 and I1
call the identical implementation — no second scoring system was
invented): `100 * Σ(P(class_i) * ordinal_i / 4)`. Still `[0, 100]`, still
a **display/decision-support value**, never a probability of injury,
stampede, or physical danger.

### 🐛 Real inefficiency caught during I1 benchmarking

The first working version of `MultiHorizonRiskModel.predict()` called
`model.predict()`, `model.predict_proba()`, **and**
`model.predict_risk_score()` separately per model — but
`predict_risk_score()` itself calls `predict_proba()` internally, so this
computed probabilities **twice** (plus a separate `predict()` call) for
every one of the three models. Measured combined-inference latency:
**69.68ms**. Fixed: `predict_proba()` is now called **exactly once** per
model, with `risk_class` derived via `proba.idxmax(axis=1)` and
`risk_score` via the shared `risk_score_from_proba()` helper — both
proven **exactly equivalent** to the original separate calls
(`test_multi_horizon_idxmax_matches_pipeline_predict`: 0 mismatches over
500 real predictions). Fixed latency: **28.84ms** — a genuine ~2.4x
improvement, not a rounding artifact.

### Models & artifacts (H1's model is never touched)

| Artifact | Model | Trained on |
|---|---|---|
| `models/artifacts/risk_model.pkl` | H1's canonical current-risk model | G1's 120s dataset (unchanged from H1) |
| `models/artifacts/future_risk_current.pkl` | I1's own current-risk retrain | I1's 900s dataset (for apples-to-apples comparison with the 5m/10m models below) |
| `models/artifacts/future_risk_5m.pkl` | +5 minute forecast | I1's 900s dataset |
| `models/artifacts/future_risk_10m.pkl` | +10 minute forecast | I1's 900s dataset |

`test_current_model_saved_separately_from_h1_model` explicitly guards
against I1 ever overwriting `risk_model.pkl`. Every I1 artifact's
`.meta.json` sidecar records everything H1's did (`sklearn_version`,
`feature_names`, `model_version`) **plus** `horizon`, `horizon_seconds`,
`target_column`, `training_dataset`, `training_seed`, and
`hyperparameters` — via `RiskModel.save()`'s new optional
`extra_metadata` parameter (additive, backward-compatible — H1's own
`model.save(path)` calls are unchanged).

### Model selection

Per the spec, only `HistGradientBoostingClassifier` was evaluated for I1
(not Logistic Regression — the spec explicitly says "start with the H1
model family"), sweeping `max_leaf_nodes ∈ {15, 31, 63}` per horizon,
selected via the same safety-first criteria as H1 (reused from
`models/training_utils.py::select_best`, not duplicated).
`max_leaf_nodes=31` was selected for all three horizons.

### Results (test set) — honest, not hidden

| Horizon | Rows (train/val/test) | Accuracy | Macro F1 | Weighted F1 | Dangerous Recall | False-Safe Rate | Ordinal MAE | Log Loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| current | 26,100 / 5,400 / 6,300 | 0.9822 | 0.7867 | 0.9889 | 0.9711 | 0.0050 | 0.0338 | 0.1444 |
| +5 min | 17,400 / 3,600 / 4,200 | 0.7521 | 0.5815 | 0.7586 | 0.7992 | 0.1832 | 0.4402 | 2.2286 |
| +10 min | 8,700 / 1,800 / 2,100 | 0.5643 | 0.4545 | 0.5913 | 0.6427 | 0.2873 | 0.7100 | 3.2179 |

**Exactly the expected degradation pattern** (spec section 13): current
is easiest, +5m degrades noticeably, +10m degrades further. This is
reported as-is — no result was hidden or softened.

### ⚠️ The most important finding: escalation recall is much lower than dangerous recall

```
                    +5 min    +10 min
dangerous_recall:    0.7992    0.6427
escalation_recall:   0.1250    0.1067
```

`dangerous_recall` counts **every** truly-dangerous future sample,
including ones that were *already* HIGH/CRITICAL now and simply stay
that way — an easy case. `escalation_recall` (new I1 metric,
`evaluation/metrics.py::escalation_recall`) isolates the **hard, actually
useful** case: of samples that were SAFE/LOW/MODERATE *now* but
genuinely became HIGH/CRITICAL at the horizon, how often did the model
catch it in advance? Only **12.5%** at +5m and **10.7%** at +10m.

**Honest interpretation:** this baseline is good at recognizing that an
*already-dangerous* state will likely persist, but poor at anticipating a
*new* escalation before it starts. This is a plausible, defensible
limitation of a snapshot-feature classifier trained on scenarios where
the transition start time is randomized per realization — predicting
*when* an escalation will begin from pre-transition steady-state features
alone is a genuinely hard forecasting problem, not obviously solvable by
a lightweight tabular baseline. This is reported prominently, not buried,
per the spec's explicit "do not hide weak results" instruction.

### Inference performance

| | Latency |
|---|---:|
| current-only | 7.78 ms/sample |
| current + 5m + 10m (bundled) | 28.84 ms/sample |

Higher than the spec's aspirational "a few milliseconds added" — three
full `HistGradientBoostingClassifier` models each cost roughly what H1's
single model costs (~7-8ms), and three of them run sequentially. This is
reported honestly rather than claiming an artificially small number;
28.84ms/sample remains comfortably fast enough for CrowdShield's ~1Hz
per-zone inference rate (Phase K).

### How to run

```bash
python scripts/train_future_risk.py \
  --dataset sample_data/synthetic_long/synthetic_dataset.csv \
  --output-dir models/artifacts
```

### Tests

```bash
python -m pytest tests/ -v
```

- `tests/test_future_risk.py` — 18 tests: future-label shift correctness,
  sequence-boundary leakage, horizon truncation (both directions),
  feature-leakage exclusion, feature-ordering compatibility, save/load
  round-trip with extended metadata, `MultiHorizonRiskModel` schema and
  the `idxmax`-equivalence regression test, dangerous-probability bounds,
  risk-score bounds, probability sum-to-1, determinism.
- `tests/test_future_risk_integration.py` — 9 tests against the actual
  packaged I1 artifacts: all three models load, identical feature
  ordering across horizons, correct `horizon_seconds`/`sklearn_version`
  metadata, real predictions on the real test split, no metadata
  leakage, `escalation_recall` computes without crashing on real data,
  the report/disclaimer exists, and I1 never overwrites H1's model file.

Full suite (excluding integration): **221 passed, 5 skipped in ~27s.**

### Known limitations

- **Trained only on synthetic data** — same caveat as H1, extended to
  the future horizon.
- **Escalation recall is low (~11-13%)** — the model rarely anticipates a
  genuinely new escalation before it starts; it's better at recognizing
  persistence of an already-dangerous state. This is the single most
  important limitation to understand before treating I1's future
  predictions as an early-warning signal.
- Both +5m and +10m accuracy/macro-F1 are meaningfully lower than
  current-state accuracy — expected, documented, not hidden.
- Combined 3-model inference latency (~29ms) exceeds the spec's "a few
  milliseconds" aspiration — still fast enough for the ~1Hz target rate,
  but not negligible.
- I1's dataset (900s sequences, fewer realizations for size reasons) is
  a **different** synthetic dataset than H1's shipped 120s dataset — the
  `current` horizon's I1-trained metrics are therefore not directly
  comparable to H1's own reported numbers; they're evaluated separately
  and specifically for apples-to-apples comparison with I1's own 5m/10m
  models.
- Small val/test split sizes (6/7 sequences) again produced some
  per-class coverage gaps, consistent with the pattern H1 already
  documented — not re-litigated here in depth, same root cause (small
  sequence counts at this split fraction).
- No SHAP/explainability integration yet (explicitly deferred to a later
  phase per spec).

## Phase J1 — Explainability

### ⚠️ Carrying forward I1's limitation — read this first

> **I1's future-risk escalation recall remains low (~12.5% at +5m and
> ~10.7% at +10m). Future-risk explanations describe the model's
> reasoning — which features drove a prediction — but this does NOT
> imply reliable early-warning performance.** A clear, well-grounded
> explanation of a forecast does not make that forecast more accurate.

### Goal

Answer "why did CrowdShield classify this zone as SAFE/LOW/MODERATE/
HIGH/CRITICAL?" and, for I1's future models, "why does the model believe
this zone will become more/less dangerous?" — grounded in the actual
features the actual trained model used, never a fabricated narrative.

### Method investigated first: SHAP

`shap.TreeExplainer` was evaluated against the real shipped, exactly-
pinned `HistGradientBoostingClassifier` (sklearn 1.8.0) before deciding
anything:
- **Worked correctly**: additivity verified (`expected_value + sum(shap_values)`
  reproduces the actual predicted class via argmax on real test rows),
  determinism verified (identical `shap_values()` output across repeated
  calls), multiclass-native, ~2.7ms/row.
- **Rejected anyway**: `pip install shap` pulled in `numba` + `llvmlite`
  — **206MB** of an LLVM JIT compiler toolchain — as transitive
  dependencies, solely to explain a small tabular gradient-boosting
  model. This directly conflicts with the project's lightweight/
  CPU-friendly principle, carried through every phase (B1's model-size
  discipline, F1's "don't add numpy for simple arithmetic," I1's "no
  heavyweight forecasting models"). Rejected on genuine, measured
  dependency-weight grounds — not because it didn't work, and not
  reflexively "because SHAP is heavy" without checking.

### Method chosen: occlusion (ablation) explanation

For each feature, temporarily replace it with a reference "typical"
value (the training set's per-feature median — the same convention H1's
median-imputation already used) and re-run `predict_proba()`. The drop
in confidence for the predicted class when a feature reverts to typical
is that feature's contribution:

```
contribution(feature) = confidence(actual row) - confidence(row with feature -> reference)
```

- **Grounded directly in the actual model's actual behavior** — an
  empirical measurement via real `predict_proba()` calls, not an
  approximation of internal tree structure that could drift across
  sklearn versions (the exact category of risk that caused H1's
  portability bug).
- **Fully deterministic** — no sampling, no randomness.
- **Zero new dependencies** — uses only `RiskModel.predict_proba()`,
  already present.
- **Coarser than SHAP** — ignores feature interactions beyond the single
  perturbed feature. Stated plainly, not oversold.

### 🐛 Real inefficiency caught during J1 benchmarking

The first working version built the perturbation matrix with a Python
loop of 76 individual `pandas.DataFrame.loc[i, feature] = ...`
assignments, and separately called `predict()`, `predict_proba()`, and
`predict_risk_score()` (which itself calls `predict_proba()` again
internally — the exact inefficiency I1 already fixed once for
`MultiHorizonRiskModel`). Measured: **62.7ms** per explanation. **Fixed**
by building the perturbation matrix with `numpy.tile` + `numpy.fill_diagonal`
(vectorized, no Python loop) and calling `predict_proba()` exactly once,
deriving both `predicted_class` (via `proba.idxmax(axis=1)`, verified
exactly equivalent to `pipeline.predict()`) and `risk_score` (via the
shared `risk_score_from_proba()` helper) from that single result. Fixed
latency: **26ms** — a genuine ~2.4x improvement.

### 🐛 Real robustness gap found and fixed in `RiskModel`

Building J1's tests with a small dataset fixture surfaced a pre-existing
bug: if a model's training data happened to be missing one of the 5
classes entirely (a real, already-documented small-dataset risk from
H1/I1's coverage-gap findings), `pipeline.classes_` only has 4 entries,
and `RiskModel.predict_proba()`'s original `raw_df[LABELS_ORDERED]`
direct indexing raised a `KeyError`. **Fixed** via
`.reindex(columns=LABELS_ORDERED, fill_value=0.0)` — the model now
correctly reports `0.0` probability for a class it never saw during
training, rather than crashing. This is a `RiskModel`-level robustness
fix (not explainability-specific) with its own dedicated regression test.

### Explainability API

```python
from models.explainability import ModelExplainer, format_explanation

explainer = ModelExplainer.from_training_data(risk_model, reference_df=X_train)
explanation = explainer.explain(X_row)   # exactly 1 row

explanation.predicted_class          # "HIGH"
explanation.predicted_probability    # 0.91
explanation.risk_score               # 76.4
explanation.top_positive_features    # up to top_k FeatureContribution, sorted desc
explanation.top_negative_features    # up to top_k FeatureContribution, sorted asc
explanation.feature_contributions    # dict: every feature, not just top-k
explanation.feature_values           # dict: actual observed value per feature
```

`MultiHorizonExplainer.from_bundle(bundle, reference_df)` bundles three
`ModelExplainer`s (current/+5m/+10m), built **once** from one shared
reference dataset — mirrors `MultiHorizonRiskModel`'s "load once, reuse"
design; `explain_all(X_row)` returns all three horizons' explanations in
one call.

### Human-readable formatting

`human_readable_feature_name()` is a deterministic **parser**, not a
hand-maintained 76-entry lookup table — it splits F1's systematic
`w{N}_{suffix}` naming pattern (e.g. `w30_direction_variance_slope` →
"30s direction-variance trend") and falls back to a small base-metric
dictionary for the 8 direct E1 fields, or the raw feature name for
anything unmapped. This avoids typos/omissions and stays correct
automatically if F1 ever adds a new statistic.

**Real edge case found and fixed:** a feature can be `NaN` in the actual
row (F1's warm-up convention — not enough rolling-window history yet)
while still having a measurable contribution (HGB routes NaN through a
learned split direction, so reverting it to a reference value can
genuinely shift the prediction). The original formatter printed literal
`"nan (below typical ...)"`, which is nonsensical. **Fixed**: NaN values
are now described as `"not available yet (insufficient rolling-window
history)"` — honest, consistent with F1's own "missing data, not
fabricated data" philosophy.

**Language discipline** (spec requirement, tested):
`test_format_explanation_avoids_fabricated_psychological_language`
asserts the formatter's output never contains "panic," "scared,"
"afraid," "terrified," or "chaos" — only precise, measured language like
"movement direction variance ... above typical."

### Example output (real, from the shipped model)

```
Current risk: CRITICAL
Confidence: 100%
Risk score: 100.0/100

Primary contributors (supporting this prediction):
  1. movement direction variance (conflicting movement): 0.286 (above typical 0.149)
  2. 60s minimum density: 2.389 (above typical 0.826)
  3. 30s minimum density: 2.389 (above typical 0.990)
  4. 5s average direction variance: 0.286 (above typical 0.143)
  5. 60s average direction variance: 0.286 (above typical 0.122)

Features working against this prediction:
  1. 15s sample count: 1.000 (below typical 16.000)
  2. 5s density trend: not available yet (insufficient rolling-window history)
  3. 5s speed trend: not available yet (insufficient rolling-window history)
  4. 60s speed trend: not available yet (insufficient rolling-window history)
  5. 60s density trend: not available yet (insufficient rolling-window history)
```

### How to run

```bash
python scripts/run_explainability_demo.py \
  --model-dir models/artifacts \
  --dataset sample_data/synthetic_long/synthetic_dataset.csv \
  --n-examples 3
```

Prints current/+5m/+10m explanations for a few representative test rows,
plus the I1 escalation-recall disclaimer above. No large generated
artifact — this is a console demo, consistent with E1/F1's demo scripts.

### Performance

| | Latency |
|---|---:|
| Prediction only | ~8ms |
| Explanation only | ~26ms |
| Combined (predict + explain) | ~34ms |

Per the spec's guidance, explanation is **not** designed to run every
frame — it's for when risk changes, crosses a threshold, or an operator
explicitly requests detail (Phase K's real-time loop should keep calling
`predict_risk_score()` at its ~1Hz rate and only call `explain()` on
demand).

### Tests

```bash
python -m pytest tests/ -v
```

- `tests/test_explainability.py` — 24 tests: explanation object creation,
  input validation, feature-name/leakage checks, top-k dimensions and
  sort order, determinism, predicted-class/probability/risk-score
  agreement with `RiskModel`, NaN handling (both in `explain()` and in
  `format_explanation()`), human-readable name parsing (window-prefixed,
  base-metric, unknown fallback), forbidden-language check,
  `idxmax`-vs-`predict()` equivalence, `MultiHorizonExplainer` across all
  three horizons with independent predictions, and a save/load
  round-trip compatibility check.
- `tests/test_risk_model.py` — regression test for the missing-class
  `predict_proba()` robustness fix, using a minimal synthetic 4-class
  training set.

Full suite (excluding integration): **246 passed, 5 skipped.**

### Known limitations

- Occlusion-based contributions ignore feature interactions beyond the
  single perturbed feature — a coarser approximation than SHAP's exact
  attribution, by design (see method tradeoff above).
- The reference ("typical") value is a single global median per feature,
  not conditioned on zone or scenario context — an unusually-low value in
  a naturally-low-density zone could be reported as "below typical" even
  though it's normal for that specific zone.
- Explanation latency (~26ms) is higher than prediction latency (~8ms) —
  fine for on-demand use, not designed for a per-frame hot loop.
- **I1's escalation recall remains low (~12.5% at +5m, ~10.7% at +10m)** —
  a well-explained future prediction is still a prediction with that
  limitation; J1 does not change or improve I1's underlying accuracy.
- No SHAP or other formal attribution method is available as an
  alternative/fallback if the occlusion method's simplifying assumption
  (single-feature perturbation) proves inadequate later — would need to
  be revisited if this becomes a problem.

## Phase K1 — Crowd Diversion / Venue Decision Engine

The pipeline now moves from "detect → analyze → predict" to
**"detect → analyze → predict → DECIDE → recommend action."**

K1 answers: *if zone A is becoming unsafe, which connected zone should
CrowdShield recommend diverting people toward?*

> **K1 recommends diversion destinations. It does not physically control
> crowds, open gates, or dispatch anyone.** It is a decision-support layer,
> not a certified emergency evacuation system. CrowdShield does **not**
> know real-world evacuation routes, signage, staffing, or human behavior
> unless they are explicitly encoded in the venue configuration. A
> "recommended diversion destination" is not a "guaranteed optimal
> evacuation route." No new ML model is trained in K1 — it reuses H1/I1
> risk outputs plus D1/E1/F1 signals.

### Zone graph (explicit adjacency)

`config/schema.py::ZoneConfig` gained `capacity`,
`preferred_utilization_threshold`, `neighbors`, and `neighbor_costs`
(same Pydantic/YAML system — no second config mechanism). Connectivity is
**declared, never inferred from polygon geometry**: `neighbors` is a
per-zone directed list, so a one-way relationship is expressed by
omitting the reverse direction. `VenueConfig` rejects unknown neighbors,
self-neighbors, and `neighbor_costs` for non-neighbors.
`decision/zone_graph.py::ZoneGraph` exposes neighbors in declared order
(deterministic) and `routing_cost` (default 1.0; `inf` for non-neighbors).
Scope is **direct neighbors only** — no multi-hop pathfinding.
`config/venues/k1_demo_venue.yaml` is a purpose-built 6-zone demo graph
(engineering assumptions, not a real venue).

### Diversion trigger (which zones need diversion)

`is_diversion_source()` fires on ANY of: current risk ≥ `source_current_risk_threshold`
(default HIGH); +5m or +10m risk ≥ `source_future_risk_threshold`;
`surge_flag` (if `source_surge_triggers`); 60s bottleneck pressure ≥
`source_bottleneck_pressure_threshold` (if set). **High density alone
never triggers diversion**, and disabled zones are never sources.

### Destination selection: two separate phases

1. **Hard rejection** (a rejected candidate is never scored or selected):
   disabled zone; capacity not configured; at/over capacity; utilization
   above `preferred_utilization_threshold`; current risk above
   `destination_max_current_risk` (default MODERATE); +5m/+10m risk above
   `destination_max_future_risk` (default MODERATE); no state available.
2. **Weighted scoring** among survivors (lower is better):
   `score = w_risk·risk + w_capacity·utilization + w_congestion·density + w_flow·flow + w_routing·cost`
   where `risk` is the ordinal of the worst known (current/+5m/+10m) label,
   `flow = max(0, net_flow_rate_60s) + 0.5·bottleneck_pressure_60s (+5.0 if surge)`,
   and `cost` is the configured edge cost. Defaults
   (`DiversionConfig`): risk 10, capacity 3, congestion 2, flow 1.5,
   routing 1 — an explicit **engineering priority ordering** (risk ≫
   capacity > congestion > flow ≥ routing), **not empirically tuned or
   validated**; all are configurable. NaN/missing flow values degrade to
   0.0 rather than propagating.

Each `DiversionRecommendation` carries `destination_zone`, `urgency`,
`confidence`, `reason` (generated only from actual values),
`rejected_candidates` (with reasons), `score_breakdown`,
`timestamp_s`, and `model_horizon_used`. `confidence` is the winner's
relative margin over the runner-up — **not a statistical probability**.

### Safety behavior

- **NO_SAFE_DESTINATION**: if every neighbor is rejected, `destination_zone`
  is `None` with the rejection reasons — K1 never forces a destination.
- **Emergency mode** (worst known source risk is CRITICAL): `urgency=EMERGENCY`
  and a different weight profile (more risk-averse, less routing-sensitive).
  It **never relaxes the hard rejection filters** — verified by test.
- **Deterministic tie-breaking**: lowest score, then alphabetical `zone_id`.

### Multi-source allocation

`recommend_all()` evaluates every source zone, processing the most severe
first (then alphabetical). Each accepted diversion **reserves** the source's
`people_count` at the destination, so later sources see reduced headroom
and cannot collectively overload it (greedy, not an optimizer — no
optimization library). Example from the tests: `entrance` and
`exit_north` both can only reach `main_hall`; once `entrance` reserves
space, `exit_north` correctly receives NO_SAFE_DESTINATION.

### Pipeline wiring

`decision/state_builder.py::build_zone_state()` maps real E1
`ZoneMetrics`, F1 `TemporalFeatures`, and H1/I1 risk labels into a
`ZoneState` — pure field mapping, no duplicated math; F1 may be `None`
during warm-up.

### How to run

```bash
python scripts/run_diversion_demo.py --venue config/venues/k1_demo_venue.yaml
```

The demo constructs illustrative `ZoneState` inputs (not a video) and
prints genuinely computed candidates, scores, rejections, and
recommendations through the real engine.

### Tests

`tests/test_decision_engine.py` — **31 tests**: config/graph validation
(unknown/self neighbors, bad costs), disconnected-zone unreachability,
basic diversion, risk-over-distance, capacity and future-risk rejection,
disabled zones, NO_SAFE_DESTINATION, deterministic repeated calls, score
breakdown vs. configured weights, multi-source overload prevention, NaN
and missing-state handling, every source-trigger rule, and emergency mode
never relaxing safety. Full suite: **277 passed, 5 skipped**.

### Known limitations

- Direct-neighbor diversion only; no multi-hop routing or path planning.
- Capacity, adjacency, and routing costs exist only as configured —
  CrowdShield has no knowledge of real egress routes, signage, or staffing.
- Reservation uses the source's full `people_count` as the diverted
  amount (conservative simplification); no partial/split diversion.
- Scoring weights, thresholds, and the flow/surge penalties are
  engineering policy choices, not validated safety standards.
- **I1's escalation recall remains low (~12.5% at +5m, ~10.7% at +10m).**
  K1 consumes I1's future-risk labels as inputs, so its future-risk-based
  rejections and triggers inherit that limitation; K1 does not improve I1.
- H1/I1 were trained only on synthetic data; K1 has not been validated
  against real venue incidents. The demo uses hand-built states, and no
  opt-in integration test against the live video pipeline was added.

## Phase L1 — End-to-End Inference Pipeline

L1 connects every earlier phase into one executable flow. It is **integration only**: no new model,
no change to any model artifact, threshold, or K1 scoring policy.

```
video / frames ──► person detection (B1) ──► tracking (C1) ──► zone assignment + per-zone metrics (D1/E1)
   ──► rolling temporal features (F1) ──► H1/I1 risk: current, +5m, +10m ──► J1 explanations
   ──► K1 diversion recommendations ──► one JSON object per processed frame
```

Reused as-is: `FrameSource` (FPS sampling), `PersonDetector`, `PersonTracker`, `CrowdMetricsCalculator`
(which already performs zone assignment and inflow/outflow transitions), `TemporalFeatureBuilder`,
the packaged H1/I1 models, `ModelExplainer`, `build_zone_state`, `DiversionEngine`. The only new
computation is `build_feature_row()`, which flattens E1/F1 objects into the model's feature row
(no existing helper did this); a test checks it against the real model's `feature_names`.
Models, detector, tracker and explainers are loaded **once**; results are yielded, never accumulated.

### Run it

```bash
python scripts/run_pipeline.py --video sample_data/test_video.mp4 \
    --venue config/venues/example_venue.yaml --max-frames 24 \
    --output sample_data/output/pipeline_output.jsonl --validate
python scripts/run_pipeline.py --help
python -m pytest tests/test_pipeline.py -v      # L1 tests (no YOLO weights / network needed)
python -m pytest tests/ -q                      # full suite
```

Output is JSON Lines (one object per processed frame; `--output -` for stdout, logs then go to stderr).
Exit codes: `0` ok · `1` pipeline failure or schema violation · `2` bad arguments, paths or config.
Ultralytics downloads `yolov8n.pt` on first real run (weights are not shipped). Programmatic use:
`InferencePipeline.from_config(cfg).run(FrameSource(path, fps), max_frames=N)`. Settings live in the existing
config system (`config/schema.py::InferenceConfig`: `model_dir`, `reference_dataset`, `explain_mode`
`off|alerts|all`, `explain_top_k`, `max_consecutive_failures`).

### Output schema (`schema_version: "l1.0"`)

| Field | Meaning |
|---|---|
| `timestamp_s`, `frame_id` | source timestamp (seconds) and source frame index (`timestamp_s` is `null` only for an invalid timestamp) |
| `pipeline_status` | `ok` · `warming_up` (a zone's 60 s window is not yet full) · `degraded` (component failure, skipped frame, or any unavailable prediction) |
| `model_data_notice` | always present: synthetic-only training, low I1 escalation recall, decision-support only |
| `capabilities` | which of `future_5m`, `future_10m`, `explanations` are loaded/enabled |
| `zones[]` | per configured zone: `status` (`ok`/`warmup`/`unavailable`/`disabled`/`no_data`), `people_count`, `capacity`, `utilization`, `density`, `metrics`, `temporal` (surge, bottleneck, flow, per-window maturity), `risk.{current,future_5m,future_10m}` (`status`, `risk_class`, `risk_score`, `dangerous_probability`, `class_probabilities`, `reason` when unavailable), `input_quality` |
| `explanations{}` | per explained zone and horizon: predicted class and top supporting/opposing features (`feature`, `description`, `value`, `reference_value`, `contribution`, `direction`). Which zones are explained depends on `explain_mode` |
| `diversion_recommendations[]` | K1 output: `source_zone`, `destination_zone`, `status` (`RECOMMENDED`/`NO_SAFE_DESTINATION`), `urgency`, `confidence`, `reason`, `rejected_candidates[]`, `score_breakdown`, `model_horizon_used`, `during_warmup`, `source_inputs_complete` |
| `warnings[]` | missing inputs, warm-up/disabled zones, unavailable predictions, skipped frames, degraded components |
| `incomplete_input_zones[]` | zones whose model inputs are partly unavailable (see below) |

Rules: values are strictly JSON-safe (NumPy converted; NaN/Infinity never emitted); **`null` always means
unavailable, never zero or safe** - a genuine zero (e.g. `people_count: 0`) stays `0`. A zone whose current
prediction is unavailable gets `risk_class: null` and is **excluded from K1 rather than treated as safe**.
`validate_result()` (`inference/schema.py`) checks a result against this schema; `--validate` applies it to every frame.

### Example (real output: `sample_data/test_video.mp4`, real YOLOv8n, packaged models, frame 23, trimmed)

```json
{
  "schema_version": "l1.0",
  "timestamp_s": 1.9166666666666667,
  "frame_id": 23,
  "pipeline_status": "warming_up",
  "model_data_notice": "Risk models were trained ONLY on synthetic data (Phase G1) and have not been validated on real crowd incidents…",
  "capabilities": {
    "future_5m": true,
    "future_10m": true,
    "explanations": true
  },
  "zones": [
    {
      "zone_id": "main_area",
      "status": "warmup",
      "people_count": 1,
      "capacity": 120.0,
      "utilization": 0.0083,
      "metrics": {
        "avg_speed_px_s": 145.175,
        "speed_variance": 0.0,
        "flow_direction_deg": 310.848,
        "direction_variance": 0.0,
        "inflow_count": 0,
        "outflow_count": 0
      },
      "temporal": {
        "surge_flag": false,
        "window_maturity": {
          "60s": {
            "sample_count": 24,
            "coverage_s": 1.9166666666666667,
            "is_mature": false
          }
        }
      },
      "risk": {
        "current": {
          "status": "ok",
          "risk_class": "SAFE",
          "risk_score": 0.0,
          "dangerous_probability": 0.0
        },
        "future_5m": {
          "status": "ok",
          "risk_class": "SAFE",
          "risk_score": 0.002,
          "dangerous_probability": 0.0
        },
        "future_10m": {
          "status": "ok",
          "risk_class": "SAFE",
          "risk_score": 0.629,
          "dangerous_probability": 0.0
        }
      },
      "input_quality": {
        "complete": true,
        "missing_feature_count": 0,
        "model_feature_count": 76,
        "unavailable_metrics": []
      }
    },
    {
      "zone_id": "entrance",
      "status": "warmup",
      "people_count": 0,
      "metrics": {
        "avg_speed_px_s": null,
        "direction_variance": null
      },
      "risk": {
        "current": {
          "status": "ok",
          "risk_class": "SAFE",
          "risk_score": 30.057
        },
        "future_5m": {
          "status": "ok",
          "risk_class": "LOW",
          "risk_score": 20.87
        },
        "future_10m": {
          "status": "ok",
          "risk_class": "CRITICAL",
          "risk_score": 66.578
        }
      },
      "input_quality": {
        "complete": false,
        "missing_feature_count": 27,
        "model_feature_count": 76,
        "unavailable_metrics": [
          "avg_speed_px_s",
          "speed_variance",
          "flow_direction_deg",
          "direction_variance"
        ]
      }
    },
    "... (remaining zones omitted)"
  ],
  "explanations": {
    "entrance": {
      "future_10m": {
        "status": "ok",
        "predicted_class": "CRITICAL",
        "top_positive_features": [
          {
            "feature": "w15_density_mean",
            "description": "15s average density",
            "value": 0.0,
            "reference_value": 5.697,
            "contribution": 0.443,
            "direction": "supports_prediction"
          }
        ]
      }
    }
  },
  "diversion_recommendations": [
    {
      "source_zone": "entrance",
      "destination_zone": "main_area",
      "status": "RECOMMENDED",
      "urgency": "EMERGENCY",
      "confidence": 1.0,
      "reason": "entrance has current risk SAFE, +5m risk LOW, +10m risk CRITICAL - EMERGENCY mode active. main_area selected: risk SAFE, 1% utilization, density 0.02 …",
      "rejected_candidates": [],
      "model_horizon_used": "current+5m+10m",
      "during_warmup": true,
      "source_inputs_complete": false
    }
  ],
  "warnings": [
    "model inputs unavailable for zone(s) entrance, exit (e.g. no tracked movement, so speed/direction metrics are undefined)…"
  ],
  "incomplete_input_zones": [
    "entrance",
    "exit"
  ]
}
```

### ⚠️ Finding: predictions on zones with unavailable inputs are out-of-distribution

> **Update - resolved by the post-L1 reliability fix:** see [Post-L1 ML Reliability Fix](#post-l1-ml-reliability-fix). The analysis below is kept as the original finding.

When nobody is moving in a zone (empty, or a *stationary* crowd), E1 reports speed/direction metrics as
undefined (`null`). Those 27 of the model's 76 inputs then become `NaN`. The synthetic training data **never**
contains this after warm-up (0 of 35,280 rows with t ≥ 60 s had NaN speed/direction inputs; minimum density
0.32, so empty zones never occurred). Diagnosis: an empty zone with fully mature windows is predicted
`SAFE / LOW / CRITICAL` (current / +5m / +10m); replacing only those NaNs with SAFE-class medians (diagnostic
only - the pipeline never imputes) turns +10m into `SAFE`. The `CRITICAL` is therefore an artifact of
out-of-distribution input, not a risk signal. The same effect appears on the real sample video: the empty
`entrance` and `exit` zones get +10m `CRITICAL`, which K1 turns into `EMERGENCY` recommendations, while
`main_area` (one moving person, complete inputs) is `SAFE` at every horizon. J1 makes it visible: the top
"supporting" feature for the empty zone's CRITICAL is a density of 0.0, far *below* typical.

What L1 does about it: **reports, does not hide or impute**. Each zone carries `input_quality`, the frame lists
`incomplete_input_zones`, every recommendation carries `source_inputs_complete`, a warning is emitted, and
the CLI logs how many recommendations came from such zones. L1 deliberately does **not** drop these zones from
K1: a dense stationary crowd (a genuinely dangerous case) also has undefined direction metrics, so dropping
them would silence a real alarm. It also does not modify models or K1 policy. **Unresolved - needs a decision
outside L1:** retrain I1 with empty/stationary-zone scenarios, and/or add a K1 rule such as requiring
`people_count > 0` for a diversion source. Until then, treat recommendations with
`source_inputs_complete: false` as low-reliability.

### What is and is not integrated

- **Integrated and executed:** video -> YOLO -> ByteTrack -> zones/metrics -> temporal -> H1/I1 -> J1 -> K1 -> JSON,
  verified on the bundled 20 s sample video (24 frames, 3 zones, schema-validated) and, without YOLO, with a fake
  detector driving the real tracker, metrics, temporal, models, explainer and K1 in tests.
- **Not tested:** live camera/RTSP streams, GPU, Windows, videos other than the bundled sample, long-duration
  real-video runs, and throughput/latency benchmarking of the whole pipeline.
- **Not part of L1:** dashboard, database, API server, deployment, multi-hop routing, any crowd control.

### Tests

`tests/test_pipeline.py` (45 tests): mocked full path and data flow into models and K1; recommendations
computed by K1 (inputs change -> outputs change; spy + independent recomputation); strict-JSON safety
(NumPy, NaN, Infinity, real zero vs null); empty frames, no people, disabled zones; warm-up then mature;
unavailable future models; invalid/raising model output (never treated as safe, never passed to K1);
detector failure then recovery, repeated-failure abort, invalid frame and timestamp handling; input-quality
flags; explanation modes and isolation of explainer failures; source cleanup on completion, early stop and
error (including real `cv2.VideoCapture` release counting); bounded state over 700 frames; feature-row parity
with the real model schema; model/detector loading failures; config validation; `lap` declared in
requirements; schema-validator rejection cases; CLI `--help`, bad paths/config, JSONL output, stdout mode;
and the packaged H1/I1 models end to end with a fake detector.

### Known limitations

- Models are trained **only on synthetic data (G1)**; nothing here is validated on real incidents. Synthetic
  speeds are in arbitrary pixel units, so real-video feature scales may not match.
- **I1 escalation recall is low (~12.5% at +5m, ~10.7% at +10m)**; explanations describe model reasoning and do
  not make a future prediction more reliable.
- Zones with unavailable inputs produce out-of-distribution predictions (finding above) - unresolved.
- K1 uses **direct neighbours only**, configured capacity/adjacency only, and reserves a source's full
  `people_count` (no partial crowd splits). CrowdShield recommends destinations; it does not control crowds
  and is **not a certified evacuation system**.
- Zone assignment is by pixel polygon; people outside every zone are not counted. During the first 60 s
  every zone is in warm-up.
- `explain_mode: alerts` keeps cost low; `all` runs up to 3 explanations per zone per frame (about 26 ms each).
- `lap` is now declared in `requirements.txt` (previously relied on Ultralytics' lazy auto-install, which failed
  in a clean environment and broke the tracker).

## Post-L1 ML Reliability Fix

**Status:** complete and verified (see CHANGELOG `[L1-fix]`). Scope: ML reliability only - training data, H1/I1 retraining, undefined-input handling, K1/L1
safety policy, tests, docs. No UI/backend/deployment work.

> **Everything below is measured on synthetic data.** Near-perfect synthetic accuracy (~0.997 for the current-risk models) shows how cleanly the generator
> separates its classes, **not** real-world skill. The models are not validated on real crowds; I1's future-risk escalation recall is still low; K1 is
> decision support, not a certified evacuation system.

### 1. Root cause - confirmed from code and data (not assumed)

L1 reported that an empty zone was predicted CRITICAL at +10 min, driving EMERGENCY recommendations. Investigation found this was one symptom of **three**
distinct training/inference mismatches (`scripts/analyze_training_coverage.py`, every number computed from the CSVs):

1. **Undefined states never occur in training.** Legacy I1 data: 35,280 post-warm-up rows (the reported figure, verified), **0** with undefined speed/direction,
   **0** empty zones, minimum people_count 13 (H1 data) / 16 (I1 data). `data/synthetic_generator.py` always emitted a numeric speed and a direction variance in [0, 1].
2. **Low speed was tied to danger.** In the legacy 900 s data every non-SAFE scenario's speed decays to ~0 (`speed_trend_px_s2` < 0) while SAFE stays at 60 px/s:
   P(stationary | SAFE) = 0.00; in H1's data every stationary row is CRITICAL. A *standing but safe* zone had never been seen.
3. **Unbounded extrapolation.** Linear trends ran for the whole 900 s: up to 5,144 people in a 50 m² zone (median 1,088 for CRITICAL) while the label stayed constant.

Measured consequences on controlled states (`evaluation/probes.py`, real inference path) - both H1 **and** I1 were affected, in opposite directions:
H1 called one stationary person or ten standing people HIGH (false alarm); I1 +10m called an empty zone CRITICAL (false alarm); and **I1-current called a dense stationary
crowd (1.6 and 2.4 people/m²) LOW - missed danger** - because the undefined direction metric hid it (the same crowd with direction defined was correctly HIGH).

### 2. Which features become undefined, and why

For an **empty zone with fully mature windows, exactly 27 of the 76 model inputs are undefined** (verified from the feature row; `flow_direction_deg` is not a model input).
E1 (`features/crowd_metrics.py`) defines speed from tracks with ≥ 2 trajectory points and direction from tracks with non-zero displacement.

| Situation | Undefined model inputs | Why | Valid zeros (NOT undefined) |
|---|---|---|---|
| **Empty zone** (no tracks) | 27: `avg_speed_px_s`, `speed_variance`, `direction_variance` + for each of the 4 windows `speed_mean`, `speed_min`, `speed_slope`, `direction_variance_mean`, `direction_variance_max`, `direction_variance_slope` | no tracks → no speed or direction exists (mathematically undefined, not zero) | 37 features incl. `people_count`, `density`, inflow/outflow counts, `surge_flag`, density window means/min/max, slopes of density |
| **Standing people** (zero displacement) | 13: `direction_variance` + each window's `direction_variance_{mean,max,slope}` | speed **is** defined (a real 0.0), but direction of zero displacement is undefined | `avg_speed_px_s`, `speed_variance`, all speed window stats = 0.0 |
| **Dropout / new tracks** (people present, < 2 trajectory points) | up to the empty-zone set | no usable motion history yet - distinct from "not moving" | `people_count`, `density` > 0 |
| **Warm-up** (first sample) | slope and rate features of every window (e.g. 33 undefined for standing people at t = 0) | slopes need ≥ 2 samples (F1) | counts and levels |
| **Moving crowd, mature** | 0 | everything defined | - |

Zero and unknown are kept distinct end to end: `NaN`/`None`/`±Infinity` all become "undefined" in `build_feature_row` (infinity is not a measurement), a real `0.0` is never touched, and the
HistGradientBoosting models treat NaN natively (they route it down branches *learned from training rows with NaN* - which is exactly what was missing). Nothing is imputed anywhere.
J1 uses the same `predict_proba`, so explanations are consistent with the model and report undefined inputs as `null` (tested). K1 previously consumed future-risk labels with no notion of input quality - fixed below.

### 3. Training-data coverage, before and after

| Operating state (rows) | Legacy I1 data (train/val/test) | **v2 data (train/val/test)** |
|---|---|---|
| empty zone (people_count == 0) | 0 / 0 / 0 | **1801 / 653 / 590** |
| stationary people with direction undefined (valid 0.0 speed) | 0 / 0 / 0 | **8484 / 2418 / 2423** |
| ANY core movement metric undefined (post warm-up) | 0 / 0 / 0 | **10035 / 2995 / 2912** |
| very low occupancy (people_count <= 2) | 0 / 0 / 0 | **3010 / 954 / 890** |
| stationary (people>0 and avg_speed < 1 px/s) | 14936 / 2611 / 2530 | **10180 / 2918 / 3057** |
| dense & slow (density >= 1.5 and speed < 30) | 19693 / 3764 / 3772 | **8671 / 2790 / 2893** |
| dangerous label with speed < 10 px/s | 9365 / 1048 / 2674 | **5849 / 1802 / 1990** |
| warm-up rows (t < 60 s) | 1740 / 360 / 420 | **1960 / 600 / 600** |
| rows / sequences | 37800 / 42 | **47400 / 158** |

### 4. What changed

**Data (`data/scenarios_v2.yaml`, generator changes backward-compatible).** `scenarios.yaml` and every legacy dataset are untouched; a test regenerates the H1 dataset and requires it
to be identical to the shipped CSV. New optional scenario fields: `max_density`, `min_speed_px_s` (plateaus), `movement_mode: stationary`, `metric_dropout_prob`. Movement semantics reproduce
E1 exactly (no people → all undefined; standing → speed 0.0 valid, direction undefined; dropout → undefined with people present). The six original scenarios keep their parameters, plateauing at
where each stopped within its **original 120 s horizon**; **13 new scenarios** cover empty zone, one stationary person, stationary low/moderate/elevated/dense-high/dense-critical, sparse moving,
moving dense high/critical, slow walk (safe), and dropout variants, plus 5 new transitions (sudden arrival, sudden departure, standing crowd forming, movement stopping as density rises). Danger is driven by
**density independently of speed**. New labels follow G1's *existing* density regime (SAFE ≈ 0.5, LOW ≈ 0.8, MODERATE ≈ 1.0-1.2, HIGH ≥ 1.5, CRITICAL ≥ 2.0 people/m²) - **provisional engineering assumptions, not validated thresholds**.
Dataset: `sample_data/synthetic_v2/synthetic_dataset.csv.gz` (47,400 rows, 158 sequences, 900 s, `row_stride=3` applied after future labels so they stay exact). Split: **stratified, sequence-level**
(every scenario contributes whole sequences to train, val and test; 98/30/30 sequences; 0 shared) - this also removes the old gap where a class was absent from the test split.
Generation is deterministic (seed 20260, split seed 11); no v2 train feature vector appears in the legacy test split (0 of 6,300 checked).

**Models retrained (both families - the probes showed both were affected).** H1 (`risk_model.pkl`, LR vs HGB sweep → HGB 31 leaves) and I1 current/+5m/+10m (HGB sweeps) on v2 with the same 76 features
and the same selection rule (safety-first, **validation only**; the test split was not used to choose anything). Candidates were frozen before the test evaluation; I did not tune after seeing test results. Honesty note:
I iterated on the *generator design* after seeing the **old** models' probe failures (probes are hand-built states, not test labels); the new models' probe and test results were first-try.
Previous artifacts are kept in `models/artifacts/archive_pre_reliability/` so the comparison stays reproducible.
Each artifact's `.meta.json` records model version (`h1-v2-hgb`, `i1-v2-{current,5m,10m}`), `dataset_version: synthetic_v2`, `feature_schema_version: f1-76`, seed 42, hyperparameters, row counts, sklearn 1.8.0, feature order, and
**`training_nan_patterns`**: every exact set of undefined inputs seen in training with its row count (24 patterns for H1/current/+5m, 13 for +10m).

**K1 / L1 safety policy (deterministic, documented, PROVISIONAL - no new emergency threshold invented).**
- *Input support.* L1 classifies why inputs are undefined (`no_people` / `no_movement_observed` / `movement_unavailable` / `warmup_history`) and checks whether the models saw that **exact** pattern in ≥ 30 training rows (`inference.min_pattern_support_rows`). Undefined-but-trained-for = **supported** (reliable); otherwise **out-of-distribution**.
- *Independent rule.* `diversion.source_capacity_trigger_utilization = 1.0`: a zone with `people_count ≥` its **configured capacity** is a diversion source whatever any model says - the same "at capacity" line K1 already uses to refuse a destination. A dense standing crowd therefore can never be hidden by undefined movement metrics. `None` disables it (previous behaviour). Occupancy below capacity does **not** trigger by itself.
- *Provenance.* Every recommendation lists `trigger_basis` (`model_current_risk`, `model_future_risk_5m/10m` are model-driven; `surge_flag`, `bottleneck_pressure`, `capacity_rule` are model-independent) and a `reliability`: **`provisional`** iff the source's inputs are out-of-distribution **and** every trigger is model-driven - reported (destination still computed, warning emitted, `reliability_note` required by the validator) but not a supported conclusion; **`reliable`** otherwise (a model-independent trigger fired, or inputs are supported).
- *Destinations.* A neighbour with out-of-distribution inputs is rejected ("its predicted risk cannot be used as evidence that it is safe"); unknown is never treated as safe. Nothing is silently suppressed.
- *JSON additions (schema stays `l1.0`, additive):* zone `input_quality.{undefined_state, pattern_supported, pattern_training_rows, reliable}`, top-level `unsupported_input_zones`, recommendation `trigger_basis`, `reliability`, `reliability_note`, `source_inputs_reliable`. All earlier diagnostics (`input_quality`, `incomplete_input_zones`, `source_inputs_complete`, warnings) are retained.

### 5. Results - old vs new, same metrics

Definitions: **dangerous recall** = true HIGH/CRITICAL predicted HIGH/CRITICAL ÷ true HIGH/CRITICAL; **false-safe** = true HIGH/CRITICAL predicted SAFE/LOW ÷ true HIGH/CRITICAL (missed danger);
**false-alarm** = true SAFE/LOW predicted HIGH/CRITICAL ÷ true SAFE/LOW; **escalation recall** = of rows not dangerous now that truly are dangerous at the horizon, the share predicted dangerous.
Cells are `old → new`. Full numbers, per-class metrics, confusion matrices and per-scenario results: `models/artifacts/reliability_report.json` (`python scripts/evaluate_reliability.py`).

**V2 test split** (9000 current-horizon rows; held out from the new models, never seen by the old ones; every scenario present):

| Model | Accuracy | Macro-F1 | Dangerous recall | False-safe | False-alarm | Escalation recall | Log loss |
|---|---|---|---|---|---|---|---|
| H1 | 0.616 → 0.997 | 0.489 → 0.997 | 0.934 → 0.998 | 0.053 → 0.002 | 0.424 → 0.001 | — | 2.300 → 0.016 |
| I1_current | 0.738 → 0.997 | 0.659 → 0.997 | 0.735 → 0.998 | 0.264 → 0.002 | 0.099 → 0.001 | — | 1.987 → 0.017 |
| I1_5m | 0.437 → 0.801 | 0.408 → 0.794 | 0.672 → 0.780 | 0.291 → 0.185 | 0.156 → 0.069 | 0.029 → 0.066 | 3.289 → 0.633 |
| I1_10m | 0.349 → 0.637 | 0.289 → 0.618 | 0.743 → 0.623 | 0.209 → 0.308 | 0.519 → 0.172 | 0.420 → 0.169 | 4.030 → 1.505 |

**Legacy test splits** (the original distributions the old models were trained for):

| Model | Accuracy | Macro-F1 | Dangerous recall | False-safe | False-alarm | Escalation recall | Log loss |
|---|---|---|---|---|---|---|---|
| H1 | 0.952 → 0.948 | 0.771 → 0.763 | 0.930 → 0.939 | 0.050 → 0.061 | 0.000 → 0.004 | — | 0.461 → 0.395 |
| I1_current | 0.982 → 0.649 | 0.787 → 0.487 | 0.971 → 0.790 | 0.005 → 0.003 | 0.000 → 0.206 | — | 0.144 → 2.624 |
| I1_5m | 0.752 → 0.628 | 0.582 → 0.489 | 0.799 → 0.790 | 0.183 → 0.128 | 0.003 → 0.048 | 0.125 → 0.417 | 2.229 → 1.698 |
| I1_10m | 0.564 → 0.597 | 0.455 → 0.398 | 0.643 → 0.788 | 0.287 → 0.211 | 0.000 → 0.158 | 0.107 → 0.472 | 3.218 → 2.238 |

**Per scenario (v2 test)** - the states that were broken:

| Scenario (label) | Test rows | Metric | H1 old → new | I1-current old → new |
|---|---|---|---|---|
| EMPTY_ZONE (SAFE) | 300 | false alarm | 0.000 → 0.000 | 0.000 → 0.000 |
| SINGLE_STATIONARY_PERSON (SAFE) | 300 | false alarm | 0.997 → 0.000 | 0.000 → 0.000 |
| STATIONARY_LOW_OCCUPANCY (SAFE) | 300 | false alarm | 0.997 → 0.000 | 0.000 → 0.000 |
| STATIONARY_MODERATE_DENSITY (LOW) | 300 | false alarm | 0.997 → 0.000 | 1.000 → 0.000 |
| SLOW_WALK_SAFE (SAFE) | 300 | false alarm | 0.003 → 0.000 | 0.000 → 0.000 |
| NORMAL_STABLE (SAFE) | 300 | false alarm | 0.000 → 0.000 | 0.000 → 0.000 |
| STATIONARY_DENSE_HIGH (HIGH) | 300 | missed danger | 0.000 → 0.000 | 0.883 → 0.000 |
| STATIONARY_DENSE_CRITICAL (CRITICAL) | 300 | missed danger | 0.000 → 0.000 | 0.997 → 0.000 |
| MOVING_DENSE_HIGH (HIGH) | 300 | missed danger | 0.003 → 0.000 | 0.030 → 0.000 |
| DROPOUT_DENSE_HIGH (HIGH) | 300 | missed danger | 0.000 → 0.000 | 0.003 → 0.000 |

**Complete vs incomplete inputs (v2 test)** - 3142 of 9000 test rows have undefined inputs:

| Model | Inputs | Rows | Accuracy old → new | Missed danger (false-safe) old → new |
|---|---|---|---|---|
| H1 | complete | 5858 | 0.721 → 0.997 | 0.073 → 0.002 |
| H1 | **some undefined** | 3142 | 0.422 → 0.998 | 0.003 → 0.002 |
| I1_current | complete | 5858 | 0.880 → 0.997 | 0.017 → 0.002 |
| I1_current | **some undefined** | 3142 | 0.472 → 0.998 | 0.857 → 0.002 |

**Probe states** (13 controlled states; expected classes are provisional policy). Within expected set: old H1 10, I1-current 10, +5m 12, +10m 8 → new **13, 13, 13, 13** (all 13/13).

| Probe state | Expected | H1 old → new | I1-current old → new | I1 +5m old → new | I1 +10m old → new |
|---|---|---|---|---|---|
| empty_zone | LOW/SAFE | SAFE → **SAFE** | SAFE → **SAFE** | LOW → **SAFE** | CRITICAL → **SAFE** |
| one_stationary_person | LOW/SAFE | HIGH → **SAFE** | SAFE → **SAFE** | LOW → **SAFE** | CRITICAL → **SAFE** |
| stationary_low_occupancy | LOW/SAFE | HIGH → **SAFE** | SAFE → **SAFE** | LOW → **SAFE** | CRITICAL → **SAFE** |
| stationary_moderate | LOW/MODERATE/SAFE | HIGH → **LOW** | HIGH → **LOW** | HIGH → **LOW** | HIGH → **LOW** |
| dense_stationary_high | CRITICAL/HIGH | HIGH → **HIGH** | LOW → **HIGH** | HIGH → **HIGH** | CRITICAL → **HIGH** |
| dense_stationary_critical | CRITICAL/HIGH | HIGH → **CRITICAL** | LOW → **CRITICAL** | HIGH → **CRITICAL** | CRITICAL → **CRITICAL** |
| normal_moving | LOW/SAFE | SAFE → **SAFE** | SAFE → **SAFE** | SAFE → **SAFE** | LOW → **LOW** |
| moving_dense_high | CRITICAL/HIGH | HIGH → **HIGH** | HIGH → **HIGH** | HIGH → **HIGH** | HIGH → **HIGH** |
| low_speed_dangerous_congestion | CRITICAL/HIGH | HIGH → **HIGH** | HIGH → **HIGH** | HIGH → **HIGH** | HIGH → **HIGH** |
| low_speed_safe | LOW/SAFE | SAFE → **SAFE** | SAFE → **SAFE** | LOW → **SAFE** | LOW → **SAFE** |

**Flagged inputs.** Of 9000 v2-test rows, 3142 have undefined inputs; **53** of them hit a pattern with < 30 training rows under the new models
(19 distinct patterns in test, 24 in training) - so the out-of-distribution guard remains meaningful (it fired on warm-up transients in the real-video run).

**Latency / size.** Bundle load 0.2 s → 0.467 s; single-sample prediction (ms) H1 5.0 → 4.3,
I1-current 5.4 → 4.1, +5m 5.3 → 5.2, +10m 5.2 → 6.0;
sizes (KB) H1 1461.6 → 978.4, current 1507.5 → 1896.0, +5m 1884.9 → 1005.3, +10m 1876.4 → 1878.6. Timings are single-run on one CPU core - indicative only.

**Confusion matrices (v2 test).**
- I1_current `old` rows=true, cols=predicted (SAFE, LOW, MODERATE, HIGH, CRITICAL): [3001,2,0,5,0] ; [2,595,1,389,0] ; [0,2,365,360,192] ; [5,579,2,1980,3] ; [2,492,4,322,697]
- I1_current `new` rows=true, cols=predicted (SAFE, LOW, MODERATE, HIGH, CRITICAL): [3007,0,0,1,0] ; [2,982,0,3,0] ; [0,2,913,4,0] ; [3,2,1,2563,0] ; [2,0,2,2,1511]
- I1_5m `old` rows=true, cols=predicted (SAFE, LOW, MODERATE, HIGH, CRITICAL): [598,1024,0,101,0] ; [102,221,17,252,11] ; [0,79,154,200,170] ; [213,390,6,1219,46] ; [1,289,108,370,429]
- I1_5m `new` rows=true, cols=predicted (SAFE, LOW, MODERATE, HIGH, CRITICAL): [1585,0,0,121,17] ; [81,469,30,23,0] ; [0,64,467,11,61] ; [380,101,10,1372,11] ; [86,0,100,97,914]
- I1_10m `old` rows=true, cols=predicted (SAFE, LOW, MODERATE, HIGH, CRITICAL): [181,193,0,97,329] ; [97,9,49,59,86] ; [0,15,64,157,64] ; [186,142,17,493,162] ; [0,6,60,234,300]
- I1_10m `new` rows=true, cols=predicted (SAFE, LOW, MODERATE, HIGH, CRITICAL): [683,2,0,83,32] ; [47,163,16,74,0] ; [0,50,165,39,46] ; [351,82,18,526,23] ; [60,0,93,74,373]

### 6. Trade-offs and regressions - reported, not hidden

- **Current-risk models:** large gains on the states that were broken (H1 false alarms 0.424 → 0.001; I1-current missed danger 0.264 → 0.002). With undefined inputs the **old I1-current missed 85.7% of dangerous rows**.
- **I1 +10m is worse on missed danger** on v2-test: false-safe 0.209 → 0.308, dangerous recall 0.743 → 0.623. The old model's higher recall was bought with a 0.519 false-alarm rate (it called nearly everything dangerous, e.g. empty zones); the new one's is 0.172. This is a real trade-off, not a free win, and I did not tune it against the test set.
- **Escalation recall is still low:** +5m 0.029 → 0.066, +10m 0.420 → 0.169 on v2-test (legacy test: 0.125 → 0.417 and 0.107 → 0.472, with more false alarms). Do **not** present I1 as a reliable early-warning system.
- **Legacy I1 test regressed** (current-risk accuracy 0.982 → 0.649; false alarms 0.000 → 0.206; dangerous recall 0.971 → 0.790). A pre-declared time split shows where: inside the generator's **original 120 s envelope** there is no regression (accuracy 0.992 → 0.994, missed danger 0.017 → 0.000), whereas beyond it the legacy labels are inconsistent with density - 667 of 2182 SAFE/LOW-labelled rows (31%) sit at ≥ 1.5 people/m², up to 325 people in 50 m², which a density-aware model disagrees with. This is *consistent with* label noise in the legacy extrapolation; it does not prove it, and it is a genuine loss of fit to the legacy distribution.
| Model | Rows | False-alarm old → new | Missed-danger old → new | Accuracy old → new |
|---|---|---|---|---|
| I1_current t ≤ 120 s (inside the generator's original envelope) | 847 | 0.000 → 0.002 | 0.017 → 0.000 | 0.992 → 0.994 |
| I1_current t > 120 s (extrapolated region) | 5453 | 0.000 → 0.252 | 0.004 → 0.003 | 0.981 → 0.595 |
| I1_5m t ≤ 120 s (inside the generator's original envelope) | 847 | 0.000 → 0.091 | 0.083 → 0.083 | 0.949 → 0.902 |
| I1_5m t > 120 s (extrapolated region) | 3353 | 0.004 → 0.030 | 0.200 → 0.136 | 0.702 → 0.559 |
| I1_10m t ≤ 120 s (inside the generator's original envelope) | 847 | 0.000 → 0.211 | 0.316 → 0.276 | 0.574 → 0.651 |
| I1_10m t > 120 s (extrapolated region) | 1253 | 0.000 → 0.123 | 0.268 → 0.166 | 0.558 → 0.561 |
- **H1 on its legacy test** is essentially unchanged (accuracy 0.952 → 0.948, false-safe 0.050 → 0.061 on 12 sequences - within the noise of that small split).

### 7. Real-video smoke test (bounded; NOT real-world validation)

`python scripts/run_pipeline.py --video sample_data/test_video.mp4 --venue config/venues/example_venue.yaml --max-frames 24 --validate` (real YOLOv8n weights fetched at run time; 24 frames, 3 zones, ≤ 1 person):
all frames schema-valid, strict JSON. **Recommendations: 69 → 1** (all 69 previously came from zones with undefined inputs, including EMERGENCY for empty zones). Last frame: every zone SAFE at all three horizons; empty zones are `no_people`, *supported*.
Frames 0 and 21 flag unsupported zones (warm-up patterns seen in 6 and 0 training rows) - correctly, with no alarm. The one remaining recommendation (frame 20, t = 1.67 s, `during_warmup: true`) is `main_area` triggered **only** by `model_future_risk_10m = HIGH` with current and +5m SAFE and one person just detected:
a residual +10m false alarm during warm-up. It is `reliable` by the policy (inputs supported) - an honest limit of the weakest model, not suppressed.

### 8. Reproduce

```bash
python scripts/generate_synthetic_data.py --scenarios data/scenarios_v2.yaml --output-dir sample_data/synthetic_v2 --split-seed 11 --gzip
python scripts/analyze_training_coverage.py --dataset sample_data/synthetic_v2/synthetic_dataset.csv.gz --output sample_data/synthetic_v2/coverage_report.json
python scripts/train_risk_model.py --dataset sample_data/synthetic_v2/synthetic_dataset.csv.gz --output-dir models/artifacts --version-tag=-v2 --dataset-version synthetic_v2
python scripts/train_future_risk.py --dataset sample_data/synthetic_v2/synthetic_dataset.csv.gz --output-dir models/artifacts --version-prefix i1-v2 --dataset-version synthetic_v2
python scripts/evaluate_reliability.py      # old (archive_pre_reliability) vs new -> models/artifacts/reliability_report.json
python -m pytest tests/test_reliability.py -v
```

### 9. Remaining limitations

- Synthetic only. Labels of the new scenarios, the density→label regime, the probe expectations, the capacity rule and the support threshold (30 rows) are **provisional engineering assumptions**.
- I1 escalation recall remains low, +10m missed danger is higher than before (above), and a +10m false alarm can still occur during warm-up.
- Real detectors jitter: E1 reports *exactly* zero displacement as undefined direction, but a real standing crowd produces small random displacements (low speed, **high direction variance**). The v2 data models exact-zero standing crowds and dropout, not detector jitter; behaviour on jittery standing crowds is **untested** and could resemble "conflicting flow".
- **The speed shortcut is reduced, not eliminated.** `P(stationary | SAFE)` rose from 0.00 to 0.25, but J1 explanations of a dense standing crowd still rank movement features (e.g. `w60_speed_mean`, flow slopes) among the top supporting evidence, i.e. the models still treat "nobody is moving" as partial evidence of danger alongside occupancy. Explanations describe model reasoning; they are not proof that the reasoning is right.
- All zones in training have area 50 m² (`people_count` and `density` co-vary); other zone sizes are not represented in training.
- K1 remains direct-neighbour, configured-capacity, no partial splits; the capacity rule uses `people_count` from the tracker (people outside every zone are not counted).
- Calibration was not assessed; probabilities are not claimed to be calibrated.

## Next Stage

Open items rather than a fixed phase: validate on real or independently produced crowd data (including jittery standing crowds), consider recalibrating/regularising the +10m model, and only then
build a dashboard/API layer on top of the JSON output.
