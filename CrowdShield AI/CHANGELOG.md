# Changelog

All notable changes to CrowdShield AI (Milestone 1) are recorded here,
one entry per completed stage.

## [L1-fix] — Post-L1 ML Reliability Fix (training/inference distribution mismatch)

**Status:** Complete. Scope: ML reliability only (data, retraining, undefined-input handling, K1/L1 safety policy, tests, docs).

### Root cause (confirmed from code and data)
- The legacy synthetic data never contained empty zones or undefined movement metrics after warm-up (0 of 35,280 post-warm-up rows; min people_count 13/16),
  tied low speed to danger (P(stationary | SAFE) = 0; every stationary H1 row is CRITICAL) and extrapolated trends for 900 s (up to 5,144 people in 50 m2).
- Probes through the real inference path showed BOTH families affected: H1 called 1-10 standing people HIGH; I1 +10m called an empty zone CRITICAL; I1-current called a
  dense standing crowd (undefined direction) LOW. With undefined inputs the old I1-current missed 85.7% of dangerous rows.
- For an empty zone with mature windows exactly 27 of 76 model inputs are undefined (13 for standing people; 33 at the first sample); see README for the provenance table.

### Added
- `data/scenarios_v2.yaml` (19 scenarios, 11 transitions); generator fields `max_density`, `min_speed_px_s`, `movement_mode`, `metric_dropout_prob`, `row_stride`,
  `split_strategy`, `dataset_version`; `split_dataset_stratified()` / `apply_split()`; `--gzip`. Legacy `scenarios.yaml` unchanged: the H1 dataset regenerates identically (tested).
- `sample_data/synthetic_v2/` (47,400 rows, 158 sequences, stratified sequence-level split 98/30/30) and coverage reports for the legacy and v2 data.
- `evaluation/probes.py`, `scripts/analyze_training_coverage.py`, `scripts/evaluate_reliability.py`, `models/artifacts/reliability_report.json`.
- K1: `source_triggers()`, trigger provenance (`trigger_basis`), `reliability` (reliable/provisional + note), model-independent `capacity_rule`
  (`diversion.source_capacity_trigger_utilization`, default 1.0 = at configured capacity; None disables), `destination_requires_reliable_inputs`.
- L1: per-zone `input_quality` (undefined_state, pattern_supported, pattern_training_rows, reliable), `unsupported_input_zones`, recommendation `trigger_basis`/`reliability`/
  `reliability_note`/`source_inputs_reliable`; `inference.min_pattern_support_rows`. Earlier diagnostics retained. Schema stays `l1.0` (additive).
- Model metadata: `dataset_version`, `feature_schema_version`, `training_seed`, `training_rows`, `training_nan_patterns` (every exact set of undefined inputs seen in training).
- `tests/test_reliability.py` (28 tests).

### Changed
- H1 and I1 (current, +5m, +10m) retrained on v2: same 76 features, same safety-first VALIDATION-only selection, test not used for choices. Versions `h1-v2-hgb`, `i1-v2-*`.
  Previous artifacts kept in `models/artifacts/archive_pre_reliability/`.
- `build_feature_row`: +-Infinity is now treated as undefined (NaN), never passed to a model; valid zeros untouched.
- Old L1 test `test_real_models_flag_empty_zones_as_incomplete_inputs` strengthened (assertions added, none removed). `tests/test_decision_engine.py` unchanged and passing.

### Results (synthetic; details and caveats in README)
- v2 test: H1 accuracy 0.616 -> 0.997 (false alarms 0.424 -> 0.001); I1-current 0.738 -> 0.997 (missed danger 0.264 -> 0.002); probes 13/13 at every horizon (old 8-12/13).
- Real video (24 frames): recommendations 69 -> 1.

### Trade-offs / regressions (reported, not hidden)
- I1 +10m: missed danger worsened 0.209 -> 0.308 (dangerous recall 0.743 -> 0.623) while false alarms fell 0.519 -> 0.172. Escalation recall remains low (+5m 0.066, +10m 0.169 on v2-test).
- Legacy I1 test regressed (accuracy 0.982 -> 0.649). A pre-declared time split shows no regression inside the original 120 s envelope (0.992 -> 0.994) and label/density
  inconsistency beyond it (667 of 2,182 SAFE/LOW rows at >= 1.5 people/m2); consistent with, not proof of, legacy label noise.
- One residual +10m false alarm during warm-up in the real-video smoke test.

### Known limitations
- Speed shortcut reduced (P(stationary | SAFE) 0.00 -> 0.25), not eliminated: J1 shows movement features still among the top evidence for a dense standing crowd.
- Synthetic only; new labels, probe expectations, the capacity rule and the support threshold are provisional. Detector jitter on standing crowds is untested. Single zone area (50 m2) in training.
  K1 still direct-neighbour, configured-capacity only. Calibration not assessed.

## [L1] — End-to-End Inference Pipeline

**Status:** Complete

### Added
- `inference/pipeline.py` — `InferencePipeline`: orchestrates detection -> tracking -> zones/metrics -> temporal ->
  H1/I1 -> J1 -> K1 into one JSON-safe result per processed frame; `from_config()` loader (components injectable);
  `build_feature_row()` (E1/F1 -> model feature row); `PipelineError`.
- `inference/schema.py` — schema constants, `to_json_safe()` (NumPy/NaN/Infinity safe), `validate_result()`.
- `scripts/run_pipeline.py` — CLI (`--video`, `--venue`, `--max-frames`, `--output` file or `-`, `--explain`, `--validate`);
  exit codes 0/1/2.
- `config/schema.py` — `InferenceConfig` + `AppConfig.inference` (same config system, defaults only).
- `tests/test_pipeline.py` — 45 tests. `requirements.txt` — `lap>=0.5.12` declared.

### Design decisions
- Integration only: no model/threshold/K1-policy change; zone assignment is not duplicated (E1 already does it).
- `null` = unavailable, never zero/safe; unavailable current predictions are excluded from K1, not treated as safe.
- Distinct states: `warming_up`, `degraded`, per-zone `ok/warmup/unavailable/disabled/no_data`.
- Failure handling: one bad detection/frame degrades that frame only; `max_consecutive_failures` aborts with a clear error;
  missing models/detector fail clearly (models are checked before the detector so tests/CLI fail fast without downloads).
- Explanations are optional (`off|alerts|all`) and isolated: an explainer failure becomes a warning.

### Found and handled (not fixed - needs a decision)
- **Out-of-distribution inputs.** Zones where nobody moves (empty, or stationary crowd) have undefined speed/direction
  metrics (27/76 model inputs NaN), a state absent from the synthetic training data (0/35,280 rows after warm-up). An empty
  zone is predicted +10m `CRITICAL`; filling the NaNs with SAFE medians (diagnostic only) gives `SAFE`, so it is an artifact.
  Reproduced on the real sample video. L1 reports it (`input_quality`, `incomplete_input_zones`,
  `source_inputs_complete`, warning, CLI summary) instead of imputing or silently dropping zones (a stationary dense crowd
  would also be dropped). Follow-up: retrain I1 with empty/stationary scenarios and/or require `people_count > 0` for K1 sources.

### Fixed
- `lap` (ByteTrack dependency) was only auto-installed lazily by Ultralytics and failed in a clean environment (11 tracker
  tests failed); now declared in `requirements.txt`.
- A CLI test initially triggered a YOLO weight download; `from_config` now loads models before constructing the detector.

### Verified
- Full suite and fresh-extraction results are recorded in the release report; real-video smoke test: 24 frames of
  `sample_data/test_video.mp4`, all frames schema-valid, strict JSON.

### Known Limitations
- Synthetic-only models; I1 escalation recall ~12.5% (+5m) / ~10.7% (+10m); K1 direct-neighbour, configured-capacity
  only, no partial splits; not a certified evacuation system. Not tested: live streams, GPU, Windows, other videos,
  whole-pipeline throughput.

## [K1] — Crowd Diversion / Venue Decision Engine

**Status:** Complete

### Added
- `decision/zone_graph.py` — `ZoneGraph`: explicit configured adjacency
  (never geometry-inferred), directed per-zone neighbors, deterministic
  declared-order candidates, `routing_cost` (default 1.0, `inf` for
  non-neighbors), defensive unknown-neighbor validation.
- `decision/diversion_engine.py` — `DiversionEngine`: `is_diversion_source`
  triggers, two-phase destination selection (hard rejection, then
  deterministic weighted scoring), `evaluate_candidates`,
  `recommend_for_source`, and greedy capacity-reserving `recommend_all`.
- `decision/schemas.py` — `ZoneState`, `ScoreBreakdown`,
  `RejectedCandidate`, `DiversionRecommendation`, ordinal helpers.
- `decision/state_builder.py` — E1/F1/H1/I1 -> `ZoneState` field mapping.
- `config/schema.py` — `ZoneConfig` gained `capacity`,
  `preferred_utilization_threshold`, `neighbors`, `neighbor_costs`;
  `VenueConfig` neighbor validation; new `DiversionConfig`;
  `AppConfig.diversion`. `example_venue.yaml` updated;
  `config/venues/k1_demo_venue.yaml` added.
- `scripts/run_diversion_demo.py` and `tests/test_decision_engine.py`
  (31 tests).

### Design decisions
- No new ML model; K1 reuses H1/I1 risk labels and D1/E1/F1 signals.
- Safety-first: hard rejection is never blended into scoring; emergency
  mode (CRITICAL source) changes only the scoring weights, never the
  rejection filters. A destination with no configured capacity is
  rejected rather than assumed safe.
- If all neighbors are rejected the result is NO_SAFE_DESTINATION, never a
  forced destination.
- Deterministic: score, then alphabetical `zone_id`. Multi-source:
  most severe source first, each accepted diversion reserves the
  source's `people_count` at its destination.
- Weights/thresholds are documented engineering priorities, not tuned.

### Process note
The development sandbox reset several times mid-phase, discarding files
written outside the persisted ZIP; the K1 core was rebuilt from the
surviving schema/config work and re-verified by automated tests rather
than relying on earlier ad-hoc manual results.

### Verified
- 31 K1 tests pass; full suite 277 passed, 5 skipped.
- Demo output is computed by the real engine from the real venue config.

### Known Limitations
- Direct-neighbor diversion only; no multi-hop routing.
- Capacity/adjacency/cost exist only as configured; not a certified
  evacuation system and not validated on real incidents.
- Full `people_count` reservation per diversion (conservative).
- Inherits I1's low escalation recall (~12.5% at +5m, ~10.7% at +10m)
  for any future-risk-based trigger or rejection.
- No live-video integration test for K1.

## [J1] — Explainability

**Status:** Complete

### Added
- `models/explainability.py` — `RiskExplanation`, `FeatureContribution`,
  `ModelExplainer` (occlusion/ablation local explanations, built once
  per model and reused), `MultiHorizonExplainer` (bundles current/+5m/
  +10m explainers, mirrors `MultiHorizonRiskModel`'s "load once" design),
  `format_explanation()` (deterministic human-readable formatter),
  `human_readable_feature_name()` (parses F1's `w{N}_{suffix}` naming
  pattern rather than a hand-maintained 76-entry lookup table).
- `scripts/run_explainability_demo.py` — loads packaged models, prints
  representative current/+5m/+10m explanations, includes the I1
  escalation-recall disclaimer.
- Test suite: `tests/test_explainability.py` (24 tests) plus a new
  regression test in `tests/test_risk_model.py` for the missing-class
  `predict_proba()` robustness fix (below).

### Method investigation and decision
- **SHAP evaluated first, not skipped**: `shap.TreeExplainer` verified
  correct (additivity check reproduces the real predicted class via
  argmax on real test rows) and fast (~2.7ms/row) against the exact
  pinned `HistGradientBoostingClassifier` (sklearn 1.8.0). **Rejected
  anyway** — `pip install shap` pulls in `numba` + `llvmlite` (206MB of
  an LLVM JIT toolchain) purely to explain a small tabular model,
  conflicting with the project's established lightweight principle.
- **Chosen instead: occlusion (ablation) explanation** — for each
  feature, revert it to the training set's median and measure the
  confidence drop for the predicted class via `predict_proba()`.
  Grounded directly in real model behavior (not internal tree structure
  that could drift across sklearn versions — the same category of risk
  behind H1's earlier portability bug), fully deterministic, zero new
  dependencies. Documented as coarser than SHAP (ignores multi-feature
  interactions), not oversold as equivalent rigor.

### Fixed — two real issues caught during J1 development
1. **Explanation latency inefficiency** (mirrors I1's earlier
   `MultiHorizonRiskModel` fix): the first working `explain()` built the
   perturbation matrix via a 76-iteration `pandas.DataFrame.loc[i, ...]`
   loop and called `predict()`/`predict_proba()`/`predict_risk_score()`
   separately (the latter re-calling `predict_proba()` internally).
   Measured 62.7ms/explanation. Fixed via `numpy.tile`/`fill_diagonal`
   (vectorized matrix construction) and a single `predict_proba()` call
   per explanation, deriving class (via `idxmax`, verified equivalent to
   `predict()`) and risk score (via the shared `risk_score_from_proba()`)
   from that one result. Fixed: 26ms — a genuine ~2.4x improvement.
2. **`RiskModel.predict_proba()` robustness gap**: found while building
   a small test fixture — if training data happened to be missing one of
   the 5 classes entirely (a real risk already documented in H1/I1's
   small-split coverage-gap findings), `pipeline.classes_` only has 4
   entries and the original `raw_df[LABELS_ORDERED]` direct indexing
   raised `KeyError`. Fixed via
   `.reindex(columns=LABELS_ORDERED, fill_value=0.0)` — correctly reports
   `0.0` probability for an unseen class instead of crashing. Not
   explainability-specific; a genuine `RiskModel` fix with its own
   regression test.
3. **NaN feature display**: the formatter originally printed the literal
   text `"nan (below typical ...)"` for features missing due to F1's
   warm-up convention — nonsensical, since "above/below typical" is
   undefined for a missing value. Fixed to
   `"not available yet (insufficient rolling-window history)"`.

### Verified
- Explanations are grounded in the actual trained model: predicted
  class/probability/risk score from `explain()` match `RiskModel`'s own
  `predict()`/`predict_proba()`/`predict_risk_score()` exactly, for
  every horizon.
- Deterministic for identical input.
- No excluded metadata/target columns ever appear in
  `feature_contributions`.
- Formatter output never contains fabricated psychological language
  ("panic," "scared," "afraid," "terrified," "chaos" — explicitly
  tested).
- `MultiHorizonExplainer` correctly produces independent explanations
  per horizon (current/+5m/+10m can and do predict different classes for
  the same input).
- Save → load → explain round-trip produces identical explanations.
- 246 tests passing (25 new for J1 — 24 in `test_explainability.py` + 1
  robustness regression in `test_risk_model.py`), 5 integration tests
  skipped by default (unrelated, unchanged from I1).

### Known Limitations
- Occlusion contributions ignore feature interactions beyond the single
  perturbed feature — coarser than SHAP's exact attribution, by design.
- Reference ("typical") value is a single global median, not conditioned
  on zone/scenario context.
- Explanation latency (~26ms) exceeds prediction latency (~8ms) — fine
  for on-demand use, not a per-frame hot-loop operation.
- **I1's escalation recall remains low (~12.5% at +5m, ~10.7% at +10m)
  — carried forward explicitly, unchanged and un-improved by J1.** A
  well-grounded explanation of a future prediction does not make that
  prediction more reliable; J1 explains model reasoning, it does not
  validate or improve I1's forecasting accuracy.
- No alternative attribution method available if occlusion's
  single-feature-perturbation assumption proves inadequate later.

## [I1] — 5-Minute / 10-Minute Future-Risk Prediction

**Status:** Complete

### Added
- `models/future_risk.py` — `HORIZON_COLUMNS`/`HORIZON_SECONDS`,
  `prepare_future_features()` (horizon-truncation-aware feature/target
  prep, reusing `models.features` unchanged), `dangerous_probability()`,
  `MultiHorizonRiskModel` (bundles current/+5m/+10m `RiskModel`
  instances, loads once, predicts all three horizons per call).
- `models/training_utils.py` — extracted from H1's
  `scripts/train_risk_model.py` (leakage audit, HGB/LR pipeline
  builders, latency benchmark, safety-first `select_best`, temp-artifact
  cleanup) so I1's training script doesn't duplicate them. H1's script
  re-verified to produce byte-identical output after the extraction.
- `scripts/train_future_risk.py` — trains current/+5m/+10m
  `HistGradientBoostingClassifier` models (small `max_leaf_nodes` sweep),
  runs an independent future-label provenance audit, evaluates with
  H1's metrics plus the new `escalation_recall`, benchmarks combined
  vs. current-only inference latency, saves all three models with
  extended metadata.
- `evaluation/metrics.py::escalation_recall()` — of samples that were
  SAFE/LOW/MODERATE now but genuinely became HIGH/CRITICAL at the
  horizon, how often was the escalation caught in advance. NaN (not
  fabricated) when there are no genuine escalation cases to measure.
- `scripts/generate_synthetic_data.py` — additive `--duration`,
  `--realizations-per-scenario`, `--realizations-per-transition`
  overrides (backward compatible; default behavior unchanged).
- `sample_data/synthetic_long/` — I1's 900s-duration dataset (37,800
  rows, 42 sequences), generated via G1's *existing* generator with no
  generator code changes — only different CLI parameters.
- Test suite: `tests/test_future_risk.py` (18 tests) and
  `tests/test_future_risk_integration.py` (9 tests, against the actual
  packaged I1 artifacts).

### Design decisions
- **Approach A (direct future-label classification)** with three
  independent `HistGradientBoostingClassifier` models sharing H1's exact
  feature representation — no LSTM/GRU/Transformer, per spec.
- **G1's existing future-label mechanism reused unchanged** — inspected
  first, found already leakage-safe by construction (per-sequence
  `.shift()`, never crosses sequence boundaries, truncated rows already
  `NaN`). I1 added zero new label-generation or feature-exclusion logic.
- **New dataset generated with the same generator**, only a longer
  `duration_s` (900s, via new CLI overrides) — G1's shipped 120s dataset
  intentionally left untouched; H1's own `risk_model.pkl` is never
  retrained or overwritten by I1.
- **`risk_score_from_proba()` factored out of `RiskModel`** so H1 and I1
  share one scoring implementation — no second risk-score formula exists.
- **Only HistGradientBoostingClassifier evaluated** (not Logistic
  Regression) — per spec's explicit "start with the H1 model family"
  guidance for I1's initial future-risk baseline.

### Fixed — real inefficiency caught during I1 benchmarking
- `MultiHorizonRiskModel.predict()`'s first working version called
  `predict()`, `predict_proba()`, AND `predict_risk_score()` separately
  per model — but `predict_risk_score()` itself calls `predict_proba()`
  internally, so probabilities were computed twice (plus a redundant
  `predict()` call) per model. Measured: 69.68ms combined latency.
  **Fixed** by calling `predict_proba()` exactly once per model and
  deriving both `risk_class` (via `proba.idxmax(axis=1)`) and
  `risk_score` (via the shared `risk_score_from_proba()`) from that
  single result. Verified exactly equivalent to the original separate
  calls (0 mismatches over 500 real predictions,
  `test_multi_horizon_idxmax_matches_pipeline_predict`). Fixed latency:
  **28.84ms** — a genuine ~2.4x improvement.

### Verified
- Future-label provenance: sampled rows' `future_5min_label`/
  `future_10min_label` match the actual `label` at the real row 300s/
  600s later in the SAME sequence (independent spot-check beyond G1's
  own construction guarantee).
- Sequence-boundary leakage: rows within 300s/600s of a sequence's own
  end correctly have `NaN` future labels — never borrowed from another
  sequence.
- Feature/target leakage: `X` never contains `label`,
  `future_5min_label`, or `future_10min_label`, for any horizon.
- Feature-ordering compatibility: models correctly select their own
  trained feature columns regardless of input column order.
- Save/load round-trip with extended metadata (`horizon`,
  `horizon_seconds`, etc.) — identical predictions, metadata intact.
- Risk-score bounds `[0, 100]`, probabilities sum to 1, determinism —
  all reconfirmed for every horizon.
- Results reported honestly, including the weak ones: current-state
  accuracy 0.9822 → +5m 0.7521 → +10m 0.5643 (expected degradation
  pattern, not hidden).
- **Escalation recall (12.5% at +5m, 10.7% at +10m) is far lower than
  dangerous recall (79.9%/64.3%)** — investigated and reported as the
  single most important I1 finding: the model recognizes persistence of
  existing danger far better than it anticipates a new escalation.
- 221 tests passing (27 new for I1), 5 integration tests skipped by
  default (unrelated to I1, unchanged from H1).

### Known Limitations
- Trained only on synthetic data — same caveat as H1, extended to the
  future horizon.
- Escalation recall is low (~11-13%) — the most important limitation;
  this baseline is not a reliable early-warning signal for genuinely new
  escalations yet.
- Combined 3-model inference latency (~29ms) exceeds the spec's "a few
  milliseconds" aspiration, though still fast enough for the ~1Hz target.
- I1's 900s dataset is a different synthetic dataset than H1's shipped
  120s dataset — I1's own "current" horizon metrics are not directly
  comparable to H1's reported numbers.
- Small val/test split sizes again produced some per-class coverage
  gaps, consistent with the pattern H1 already documented.
- No SHAP/explainability integration yet — explicitly deferred to a
  later phase per spec.

## [H1 portability fix] — Cross-Machine Model Serialization Fix

**Status:** Complete

### Root cause
Local validation on a real Windows machine surfaced 4 test failures in
`tests/test_packaged_model.py`: `pickle.load(...)` raised
`ModuleNotFoundError: No module named '_loss'` when loading the shipped
`models/artifacts/risk_model.pkl`, alongside an `InconsistentVersionWarning`
(trained under scikit-learn 1.8.0, running under 1.9.0). Investigation
confirmed: `HistGradientBoostingClassifier` (H1's selected model) holds a
reference to a private internal loss-function object
(`sklearn._loss.loss.HalfMultinomialLoss`) whose module path is not
stable across sklearn versions. `requirements.txt` only pinned
`scikit-learn>=1.4` (a lower bound, no upper bound), so a fresh install
on another machine could resolve a different sklearn version than the
one used to train the shipped artifact, breaking `pickle`'s class
resolution on load. `precision_score`/`recall_score`/`f1_score`/`log_loss`
were unaffected — this is specifically a `pickle`/model-serialization
issue, not a metrics-computation issue.

### Fix
1. **`requirements.txt`**: `scikit-learn>=1.4` → `scikit-learn==1.8.0`
   (exact pin, deliberately different from every other dependency's loose
   lower-bound policy — documented inline as an intentional exception,
   since scikit-learn does not guarantee pickle/joblib compatibility
   across versions).
2. **`models/risk_model.py`**: `RiskModel.save()`/`.load()` switched from
   raw `pickle` to `joblib` (sklearn's own recommended tool for this,
   already a transitive dependency — no new dependency introduced), and
   split into two files: the joblib-serialized `Pipeline`
   (`risk_model.pkl`) and a plain-JSON metadata sidecar
   (`risk_model.pkl.meta.json`) recording the exact training-time
   scikit-learn version. `load()` reads the JSON sidecar first (always
   succeeds, no pickle involved) and compares versions; a mismatch emits
   an immediate `RuntimeWarning`, and if the subsequent pipeline load
   fails, the error is re-raised as a `RuntimeError` naming both versions
   and the exact fix command instead of a bare traceback.
3. **`scripts/train_risk_model.py`**: temporary candidate-model cleanup
   updated to also remove the new metadata sidecar files (was leaving
   `_tmp_*.pkl.meta.json` files behind).
4. **Model artifact regenerated** under the now-pinned scikit-learn
   1.8.0 — identical training data, identical selection logic, identical
   result: `HistGradientBoosting(max_leaf_nodes=31)` selected, same test
   metrics (accuracy 0.9521, dangerous_recall 0.9296, false_safe_rate
   0.0505, log_loss 0.4611) — only the serialization format changed.

### Preserved (unchanged)
Model-selection logic, the selected `HistGradientBoosting` model (no
technical reason found to switch away from it — the issue was
serialization strategy + dependency pinning, not the model choice),
feature names/ordering, risk-score formula, safety-first selection
criteria, all leakage protections, and every pre-existing H1 test
(unmodified, all still passing against the new serialization).

### Added — regression tests
- `test_save_writes_metadata_sidecar_with_sklearn_version` — sidecar
  written correctly with the installed sklearn version.
- `test_load_missing_metadata_sidecar_raises_clear_error` — clear
  `FileNotFoundError` if the sidecar is missing.
- `test_load_warns_on_sklearn_version_mismatch` — a tampered sidecar
  version reliably triggers a `RuntimeWarning` naming the mismatch.
- `test_load_wraps_actual_load_failure_with_clear_message` — a corrupted
  pipeline file produces a `RuntimeError` naming both sklearn versions
  and the exact `pip install` remediation, not a raw traceback.
- `test_packaged_model_has_metadata_sidecar_matching_installed_sklearn`
  — the actual shipped artifact's recorded version matches what's
  installed in the packaging environment.

### Honest verification scope
This sandbox has exactly one scikit-learn installation, so the original
`ModuleNotFoundError('_loss')` failure mode itself could not be literally
reproduced here (that requires two different sklearn installations — the
condition the user's Windows machine actually hit). What **was**
verified: (1) the version-mismatch **detection** mechanism fires
correctly on a tampered metadata file, (2) a genuine load failure
(simulated by corrupting the pipeline file's bytes) is wrapped into a
clear, actionable error rather than propagating raw, and (3) the primary
fix — the exact version pin — eliminates the root cause for any future
install that follows `requirements.txt` as shipped. This is stated
explicitly rather than claiming a cross-platform test that wasn't
actually run.

### Verified
- 194 tests passing (5 new for this fix), 5 integration tests skipped by
  default (unrelated, unchanged).
- `model_report.json` metrics identical before/after the fix — confirms
  the change is serialization-only, not a model-quality change.
- Fresh ZIP extraction: full suite passes; packaged model loads and
  predicts; predictions/risk-scores byte-identical between the
  pre-packaging model and the loaded artifact from the fresh extraction.

## [H1] — Baseline Risk Model Training

**Status:** Complete

### Added
- `models/features.py` — `LABELS_ORDERED`, `ORDINAL_MAP`,
  `DANGEROUS_LABELS`, `EXCLUDED_COLUMNS` (single source of truth for
  leakage prevention), `get_feature_columns()`, `prepare_features()`.
- `models/risk_model.py` — `RiskModel`: `fit()`/`predict()`/
  `predict_proba()`/`predict_risk_score()`/`evaluate()`/`save()`/`load()`
  around an sklearn `Pipeline`; reindexes `predict_proba()` into the fixed
  ordinal `LABELS_ORDERED` column order (sklearn's internal order is
  alphabetical, not ordinal).
- `evaluation/metrics.py` — `dangerous_recall`, `dangerous_precision`,
  `false_safe_rate`, `ordinal_mae`, `compute_all_metrics()`.
- `scripts/train_risk_model.py` — trains Logistic Regression (median
  imputation + scaling) and `HistGradientBoostingClassifier` (native NaN
  support, no imputer), minimal hyperparameter sweeps, leakage audit,
  safety-first model selection, single test-set evaluation, permutation
  feature importance, inference-latency benchmark, saves the selected
  model + a full JSON report.
- Test suite: `tests/test_metrics.py` (13), `tests/test_risk_model.py`
  (15), `tests/test_split_integrity.py` (6, against the actual packaged
  G1 dataset), `tests/test_risk_model_scenarios.py` (7, manual scenario
  sanity checks per spec section 26), `tests/test_packaged_model.py` (4,
  against the actual shipped model artifact).
- `scikit-learn>=1.4` added to `requirements.txt`.

### Design decisions
- **HistGradientBoostingClassifier chosen over RandomForest** specifically
  because it handles F1's warm-up NaN values natively — a documented,
  non-arbitrary reason, not a coin flip.
- **`flow_direction_deg` excluded from features** (spec's "Option B")
  rather than sin/cos-encoded — `direction_variance` already captures the
  risk-relevant signal, and the absolute direction has no inherent
  meaning in a single synthetic zone.
- **Safety-first model selection**: an explicit lexicographic sort by
  (dangerous_recall desc, macro_f1 desc, false_safe_rate asc, latency
  asc, size asc, accuracy desc) — never selects on accuracy alone.
- **Uses G1's existing sequence-based `split` column exactly as shipped**
  — no row reshuffling, no new random split.
- Permutation importance (not native `.feature_importances_`) for the
  selected HGB model, since HGB doesn't expose native importances the way
  RandomForest does.

### Fixed — real bug caught during development
- **`sklearn.metrics.log_loss` silently mis-scores probabilities** when
  given a non-lexicographic `labels=` order: it emits a `UserWarning` but
  does NOT reorder the probability columns to match, silently producing
  wrong values (verified: expected 0.29 on a hand-built example, sklearn
  returned 2.30 when naively passed the ordinal label order).
  `precision_score`/`recall_score`/`f1_score` were independently verified
  to correctly respect a custom `labels=` order — only `log_loss` had
  this issue. Fixed in `evaluation/metrics.py::compute_all_metrics` by
  reordering probability columns into lexicographic order before calling
  `log_loss`; locked in with a regression test
  (`test_log_loss_correct_with_ordinal_label_order`).

### Verified
- Leakage audit: no excluded metadata/target column ever appears in
  `feature_columns`; train/val/test sequence IDs fully disjoint
  (re-verified against the actual packaged dataset, not trusted blindly
  from G1's own claim).
- `predict_proba()` always returns columns in `LABELS_ORDERED` order and
  sums to 1.0.
- `predict_risk_score()` bounded `[0, 100]`; formula verified directly
  (all-CRITICAL proba → 100.0, all-SAFE proba → 0.0).
- Save → load round-trip produces byte-identical predictions,
  probabilities, and risk scores.
- Same training config produces reproducible predictions.
- Manual scenario sanity: `CRITICAL_ESCALATION` has the highest mean risk
  score among all 6 scenarios; `CONFLICTING_FLOW` scores meaningfully
  above `NORMAL_STABLE` despite only moderate density — confirming
  direction-based risk isn't masked by a density-only view.
- Validation comparison: `HistGradientBoosting(max_leaf_nodes=31)`
  selected — dangerous_recall 0.9862, false_safe_rate 0.0052, both best
  among 5 candidates (3 Logistic Regression `C` values, 2 HGB
  configurations).
- Test-set (evaluated once): accuracy 0.9521, weighted F1 0.9573,
  dangerous_recall 0.9296, dangerous_precision 1.0, false_safe_rate
  0.0505, log_loss 0.4611 (post-fix, sane value).
- **Investigated finding, documented not hidden:** test macro F1 (0.7711)
  is deflated by the 12-sequence test split having zero MODERATE-labeled
  rows by chance — confirmed via weighted F1 and dangerous-class metrics
  remaining strong and consistent with validation; not re-split (per
  spec, G1's shipped split is used as-is) — flagged as a reason to
  increase G1 realization counts in a future revision.
- Feature importance: `direction_variance` dominates by an order of
  magnitude — an observed result consistent with G1's
  `CONFLICTING_FLOW` design goal, not a forced outcome.
- 189 tests passing (45 new for H1), 5 integration tests skipped by
  default (unrelated to H1, unchanged from G1).

### Known Limitations
- Trained only on synthetic data — no real-world crowd-safety validation.
- Test macro F1 deflated by a zero-MODERATE-coverage split artifact (see
  above) — a legitimate reason to revisit G1's realization counts later,
  not implemented in this locked phase.
- HGB's ~7.5ms single-sample latency exceeds Logistic Regression's ~3ms
  — acceptable for CrowdShield's ~1Hz inference rate, worth monitoring if
  latency requirements tighten.
- No calibration curve added (log_loss alone was judged sufficient
  signal, per spec guidance not to add calibration without demonstrated need).
- No ordinal-specific classifier — standard multiclass, per spec, with
  `ordinal_mae` as the ordinal-awareness metric.

## [G1] — Synthetic Crowd-Risk Scenario Generator

**Status:** Complete

### Added
- `data/schemas.py` — `ScenarioConfig`, `TransitionConfig`,
  `GeneratorConfig` (pydantic, mirrors `config/schema.py`'s pattern;
  deliberately separate from `AppConfig` since this describes dataset
  generation, not the real-time pipeline).
- `data/loader.py` — YAML loader for `data/scenarios.yaml`, mirrors
  `config/loader.py`.
- `data/scenarios.yaml` — the 6 scenario definitions (`NORMAL_STABLE`,
  `GRADUAL_BUILDUP`, `RAPID_SURGE`, `BOTTLENECK`, `CONFLICTING_FLOW`,
  `CRITICAL_ESCALATION`) and 6 scripted transition pairs; every numeric
  value documented as an explicit engineering assumption, not a
  validated threshold.
- `data/synthetic_generator.py` — `generate_sequence()`,
  `generate_dataset()`, `split_dataset()`, `save_dataset()`,
  `load_dataset()`. Synthesizes a real `features.crowd_metrics.ZoneMetrics`
  per timestamp and feeds it through a real
  `features.temporal.TemporalFeatureBuilder` — reuses the actual E1/F1
  code path rather than duplicating slope/rate/window math.
- `scripts/generate_synthetic_data.py` — CLI to generate + split + save
  the dataset.
- `scripts/analyze_synthetic_data.py` — CLI for class-distribution and
  correlation sanity reporting, plus optional (matplotlib, dev-only)
  scenario plots. Outside the runtime pipeline, not a dashboard.
- Test suite: `tests/test_synthetic_generator.py` (24 fast unit tests).

### Design decisions
- **Feature-level only** — no fake video, YOLO, or ByteTrack; generation
  happens directly at the E1/F1 output shape.
- **Reuses real F1 code** for all rolling-window/slope/rate/bottleneck-
  pressure computation — guarantees synthetic and real data share
  identical formulas and schema, zero duplicated logic.
- **Controlled class overlap via two mechanisms**: per-timestep Gaussian/
  Poisson noise, and — the primary mechanism — scripted transitions where
  the label switches instantly at `transition_start_s` but the underlying
  feature values keep blending smoothly for `transition_duration_s`
  afterward, so features briefly still resemble the previous class after
  the label has changed. Not tuned to maximize class separability.
- **Sequence-based train/val/test split** — never splits a sequence's
  rows across splits (would leak temporally adjacent data).
- **Future labels via pure column-additive shift** (`pandas.shift()` on
  the label column within each sequence) — mathematically cannot affect
  any existing feature column, verified by a regression test.
- Deterministic seeding throughout: every sequence's seed derives from
  `base_seed` + a fixed counter, transition timing from a separate seeded
  RNG — never wall-clock time.

### Verified
- Same seed → byte-identical dataset; different seed → different
  realization.
- Every scenario produces valid, bounded data (`density>=0`, `speed>=0`,
  `direction∈[0,360)`, `direction_variance∈[0,1]`, counts `>=0`).
- Scenario semantics: `GRADUAL_BUILDUP` density rises over the sequence;
  `BOTTLENECK`'s 15s net flow rate is positive on average;
  `CONFLICTING_FLOW`'s direction variance exceeds `NORMAL_STABLE`'s;
  `RAPID_SURGE` eventually trips `surge_flag`.
- All 5 labels appear; no label exceeds 50% or falls below 5% of rows
  (actual distribution: SAFE 15.9%, LOW 20.0%, MODERATE 16.6%, HIGH
  29.8%, CRITICAL 17.7%).
- Every generated sequence has a unique `sequence_id`.
- Transition label switches exactly at `transition_start_s`; the
  single-step feature jump at that boundary is confirmed much smaller
  than the full scenario-to-scenario gap (smooth blending, not a jump).
- Split integrity: no `sequence_id` appears in more than one split;
  every sequence is assigned; split is deterministic for a fixed seed;
  bad fractions (not summing to 1.0) are rejected.
- Future-label shift verified exactly correct on a 700s test sequence,
  including correctly reflecting a *different* regime across a scripted
  transition; confirmed `NaN` (not fabricated) near the sequence end and
  for the entire dataset at the default 120s duration.
- No-leakage regression guard: regenerating without ever computing
  future-label columns produces identical feature columns.
- Manual inspection of `scripts/analyze_synthetic_data.py` plots:
  `CONFLICTING_FLOW` clearly shows the highest direction_variance
  despite only moderate density — a direct visual confirmation that
  "danger doesn't depend solely on density" as intended.
- Default dataset (84 sequences, 10,080 rows) generates in ~2.2s,
  CPU-only, no torch/YOLO/OpenCV — ~10MB in memory, 12MB CSV on disk.
- 144 tests passing (24 new for G1), 5 integration tests skipped by
  default (unrelated to G1, unchanged from F1).

### Known Limitations
- Synthetic data is engineering data, not validated crowd-safety ground
  truth (see disclaimer in README).
- Scenario parameters are explicit assumptions, not calibrated against
  real incident data.
- At the recommended 120s sequence duration, future 5-min/10-min labels
  are `NaN` for virtually every row — correct, but means longer
  `duration_s` is needed before this dataset is useful for forecasting-
  model training.
- Each sequence models exactly one synthetic zone — no multi-zone
  interaction within a sequence.
- `flow_direction_deg` blends linearly during a transition rather than
  via circular interpolation (acceptable since `direction_variance`, not
  raw direction, is the risk-relevant feature).
- No risk model exists yet to consume this dataset (Phase H1).

## [F1] — Rolling Temporal Features + Growth/Trend Detection

**Status:** Complete

### Added
- `features/temporal.py` — `WindowFeatures` (frozen dataclass: per-window
  `sample_count`, `coverage_s`, `is_mature`, density/speed/direction-
  variance rolling stats, slopes, flow rates, `bottleneck_pressure`),
  `TemporalFeatures` (per-zone container: one `WindowFeatures` per window
  + `surge_flag`), `ZoneTemporalTracker` (bounded per-zone history +
  window computation), `TemporalFeatureBuilder` (multi-zone orchestrator
  consuming E1's `ZoneMetrics` batch), and `_linear_regression_slope()`
  (lightweight least-squares helper, no numpy dependency).
- `config/schema.py::TemporalConfig` — `windows_seconds` (schema-enforced
  to exactly `[5, 15, 30, 60]`) and `density_growth_threshold` (surge
  trigger). `AppConfig` gained a cross-field validator requiring
  `analytics.feature_history_seconds >= max(temporal.windows_seconds)`.
- `scripts/run_temporal_demo.py` — console-only per-zone rolling-window
  snapshot printer.
- Test suite: `tests/test_temporal.py` (31 fast unit tests) and an opt-in
  `tests/test_temporal_integration.py` (full real A1-F1 pipeline on the
  real sample video, gated behind `CROWDSHIELD_RUN_INTEGRATION=1`).

### Design decisions
- **Exactly 4 windows (5/15/30/60s), config-enforced** — no arbitrary
  combinations, per the Milestone 1 spec.
- **Boundary rule: an observation exactly `window_seconds` old is
  included** (`<=`, not `<`) — consistent with D1's "boundary counts as
  inside" convention.
- **Bounded memory**: one `deque` per zone, capped at the largest window,
  evicted on every new observation — verified to stay bounded even after
  1000s of continuous input.
- **Slopes via lightweight least-squares regression** over actual
  timestamps (never `last - first`, never an assumed fixed FPS) — `None`
  when fewer than 2 valid points or all timestamps are identical.
- **Flow rates use the window's actual covered time span as the
  denominator**, not the nominal `window_seconds` — prevents diluting a
  rate during warm-up.
- **`bottleneck_pressure`** is `max(0, cumulative_inflow - cumulative_outflow)`
  within the window — documented explicitly as a crowd-flow proxy, not a
  physically measured pressure. Available from a single sample (unlike
  the rate fields), since it's a cumulative sum, not a rate.
- **`surge_flag`** compares only the fastest-reacting 5s window's
  `density_slope` against the configured threshold — deliberately not a
  longer window, which would dilute a sudden local spike. Documented as
  an engineering trigger, not a validated safety threshold.
- **`sample_count` / `coverage_s` / `is_mature`** exposed on every window
  so a consumer can judge maturity — never silently padded with
  fabricated data during warm-up.

### Verified
- Boundary inclusion/exclusion at exactly `window_seconds`.
- Correct rolling sample counts across all 4 windows on a hand-computed
  irregular sequence.
- Old observations evicted; history bounded after 1000s of continuous
  input.
- Density slope: known increasing/decreasing/stable sequences produce the
  expected sign and magnitude; regression helper verified directly
  against a known line (`y = 2x + 1 -> slope == 2.0`).
- Speed slope: known slowing sequence produces the expected negative
  slope; `None` with zero valid speed observations.
- Direction-variance slope: known increasing-dispersion sequence produces
  positive slope.
- Inflow/outflow/net-flow rates against known transition counts and
  elapsed times; `None` with a single observation (rate undefined).
- Bottleneck pressure: correct rolling calculation, floored at 0, and
  available from a single sample.
- Surge flag: true above threshold, false below, false during warm-up
  when the 5s slope isn't available yet.
- Warm-up: incomplete windows correctly marked `is_mature=False` with
  accurate `coverage_s`; a window reaching full coverage flips to mature.
- Irregular timestamps (`t=0, 0.7, 1.9, 3.2`): both slope and rate
  computed correctly against true elapsed time, not an assumed interval.
- Empty history: no crash, every statistic `None`, `surge_flag=False`.
- `TemporalFeatureBuilder` correctly routes each zone's metrics to its
  own tracker and silently skips unknown zone IDs.
- Real-pipeline integration test: 5s window matures early, 15s window
  matures by clip end, 30s/60s windows correctly remain immature for the
  entire ~20s clip (positively asserted, not just "didn't crash"); F1's
  internal history stays bounded throughout.
- Incremental overhead measured in a single consolidated process run
  (avoids cross-process noise seen in earlier phases): D1+E1 ~0.04ms,
  F1 ~0.60ms, both tiny relative to YOLO's ~75ms/frame.
- 120 tests passing, 5 integration tests skipped by default (all verified
  passing when explicitly enabled).

### Known Limitations
- `surge_flag`/`density_growth_threshold` are prototype engineering
  triggers, not clinically/physically validated thresholds.
- Speed-derived slope remains pixel-space (px/s²) — inherited from E1's
  lack of camera calibration.
- The ~20s bundled test clip is too short to ever mature the 30s/60s
  windows — confirmed behavior, not a workaround.
- Assumes non-decreasing timestamps from the caller; out-of-order
  timestamps are not defended against.
- No synthetic risk-model training data or ML model yet (Phase G/H).

## [E1] — Per-Zone Crowd Metrics

**Status:** Complete

### Added
- `features/crowd_metrics.py` — `ZoneMetrics` (frozen dataclass:
  `people_count`, `density`, `avg_speed_px_s`, `speed_variance`,
  `flow_direction_deg`, `direction_variance`, `inflow_count`,
  `outflow_count`), `CrowdMetricsCalculator` (stateless per-snapshot
  computation, reuses D1's `ZoneTransitionTracker` for inflow/outflow),
  and standalone helpers `track_speed_px_s()`, `track_direction_deg()`,
  `circular_mean_and_variance()`.
- `scripts/run_metrics_demo.py` — console-only per-zone metric snapshot
  printer (no new video overlay; E1 metrics are numeric, D1's zone demo
  remains the visual tool).
- Test suite: `tests/test_crowd_metrics.py` (30 fast unit tests) and an
  opt-in `tests/test_crowd_metrics_integration.py` (real pipeline on the
  real sample video, gated behind `CROWDSHIELD_RUN_INTEGRATION=1`).

### Design decisions
- **Pixel space, not real-world speed:** every speed value is named
  `avg_speed_px_s` (pixels/second), never `m/s` — no camera calibration
  exists. `density` (`people_count / area_m2`) is the only metric with a
  real-world unit, since `area_m2` is an explicit D1 config value.
  `test_speed_never_mislabeled_as_mps` guards the field name itself.
- **Circular statistics for direction**, not arithmetic averaging —
  correctly resolves angle wrap-around (359°+1° → ~0°, not the naive
  arithmetic mean of 180°) and correctly signals dispersion: two exactly
  opposing tracks give `direction_variance ≈ 1.0`, the signal to distrust
  `flow_direction_deg` in that case.
- **Documented empty/single-observation conventions** rather than
  fabricated values: 0 valid speed/direction observations → `None`
  (no data); exactly 1 → `0.0` variance (trivially zero spread); a track
  with <2 trajectory points contributes no speed/direction at all.
- **Inflow/outflow are single-snapshot counts**, not accumulated rates —
  only an actual zone change counts as a transition (a person staying in
  one zone for 20 frames generates exactly 1 inflow event, not 20).
  Rolling-rate accumulation and bottleneck pressure are deferred to F1,
  keeping E1 stateless (aside from the already-bounded
  `ZoneTransitionTracker`) and F1 temporal.
- `CrowdMetricsCalculator.compute_snapshot()` takes `trajectories` as a
  plain dict, not a `PersonTracker` reference — keeps the metrics layer
  decoupled from vision/tracker internals and trivially unit-testable.

### Fixed
- Caught during development: the circular wrap-around case could round to
  exactly `360.0` instead of `0.0` at the `[0, 360)` boundary due to
  float precision in `atan2`. Both `track_direction_deg()` and
  `circular_mean_and_variance()` now explicitly clamp `360.0 -> 0.0`.

### Verified
- People count / density formula correct against known inputs.
- Speed: known-displacement test, insufficient-history → `None`,
  stationary → `~0.0`, zone-average, population-variance with known
  values, single-observation convention, zero-observation `None`.
- Direction: all 4 cardinal angles verified under the documented
  (clockwise, y-down) convention; stationary/insufficient-history `None`.
- Circular statistics: wrap-around near 0° (not 180°), aligned tracks
  low variance, opposing/fully-dispersed tracks variance ≈ 1.0, single
  angle trivially aligned, empty input → `(None, None)`.
- Inflow/outflow: entry, exit, zone-to-zone (both sides increment),
  no-duplicate-transitions while remaining in one zone for 20 frames.
- Empty and disabled zones produce correct documented defaults; disabled
  zones produce no entry in the snapshot dict at all.
- Manual inspection of `scripts/run_metrics_demo.py` console output on
  the real clip: density scales correctly with people/area; a real
  2-person moment shows `dir_var=0.19` (moderate dispersion), a genuine
  demonstration of the circular dispersion metric on real footage.
- Zone-assignment + metrics overhead measured at roughly -4ms relative to
  C1 baseline — within this sandbox's single-core measurement noise,
  consistent with plain-Python arithmetic being cheap relative to YOLO's
  ~65-70ms/frame cost.
- 89 tests passing, 4 integration tests skipped by default (all verified
  passing when explicitly enabled).

### Known Limitations
- Speed is pixel-space only — cannot be converted to real-world units
  without camera calibration (not yet implemented).
- Inflow/outflow are single-snapshot counts; rolling rate / accumulated
  bottleneck pressure is Phase F1.
- Bundled test clip is sparse (max ~2 people) — output is real, not
  fabricated, but not validation of crowd-risk-relevant behavior at real
  density.
- `direction_variance` near 1.0 signals `flow_direction_deg` is
  unreliable, but the field still returns a numeric value rather than
  `None` in that case — callers must check variance before trusting it.

## [D1] — Zone Configuration + Point-in-Polygon Assignment

**Status:** Complete

### Added
- `features/zones.py` — `point_in_polygon()` (plain-Python ray-casting,
  no Shapely/GeoPandas), `Zone` dataclass, `ZoneManager`
  (`point -> zone_id | None`, `track -> zone_id | None`, polygons
  preprocessed once at construction), and `ZoneTransitionTracker` (bounded,
  session-scoped `previous_zone -> current_zone` reporting for future
  inflow/outflow work — no metrics computed yet).
- `config/schema.py::ZoneConfig` — added `enabled: bool` field and
  documented the pixel-coordinate convention explicitly.
- `config/schema.py::VenueConfig` — added a validator rejecting duplicate
  zone IDs within a venue.
- `config/venues/example_venue.yaml` — replaced the single placeholder
  zone with three demo zones (`entrance`, `main_area`, `exit`) as
  non-overlapping vertical bands across the bundled test video's frame,
  clearly commented as illustrative, not a real venue calibration.
- `scripts/run_zone_demo.py` — annotated demo video showing zone polygon
  outlines, track boxes/IDs, and each track's assigned zone.
- `vision/benchmark.py::benchmark_tracker_vs_zones` — measures zone
  assignment overhead on top of the C1 detector+tracker pipeline.
- Test suite: `tests/test_zones.py` (22 fast unit tests — point-in-polygon
  on simple/concave polygons, boundary behavior, overlapping/adjacent
  zones, disabled zones, track→zone mapping, config validation,
  `ZoneTransitionTracker`) and an opt-in
  `tests/test_zones_integration.py` (real detector + tracker + ZoneManager
  on the real sample video, gated behind `CROWDSHIELD_RUN_INTEGRATION=1`).

### Fixed
- `tests/test_smoke.py::test_venue_override_layers_correctly` updated to
  match the new 3-zone example venue (previously asserted the old
  single-zone `zone_a` placeholder).

### Verified
- Boundary points (edges and corners) consistently resolve as inside.
- Concave polygons correctly exclude points in the cut-out notch.
- Overlapping and boundary-adjacent zones deterministically resolve to
  the first-configured matching zone.
- Disabled zones are skipped without blocking zones beneath them.
- Config validation rejects polygons with <3 points, non-numeric
  coordinates, non-positive area, and duplicate zone IDs.
- Manual visual inspection of `sample_data/output/zones.mp4`: zone
  boundaries render at correct pixel positions; a tracked person's
  assigned-zone label updates correctly as they cross between zones.
- Zone assignment overhead measured at ~1.1ms on top of ~73ms
  detector+tracker latency — negligible as expected for plain-Python
  point-in-polygon on at most 1-2 active tracks per frame.
- 59 tests passing, 3 integration tests skipped by default (all verified
  passing when explicitly enabled).

### Known Limitations
- Example venue coordinates are illustrative only, not a real venue
  calibration — no perspective transformation exists yet.
- No density/flow/risk logic — `area_m2` is stored but unused until
  Phase E.
- Geometric validation does not verify polygons are simple
  (non-self-intersecting).
- `ZoneTransitionTracker`'s bounded eviction is insertion-order based,
  not a true LRU.

## [C1] — Multi-Object Tracking (ByteTrack)

**Status:** Complete

### Added
- `vision/tracker.py` — `PersonTracker`, the single boundary over
  Ultralytics' `BYTETracker`. Converts `Detection` objects to the minimal
  array interface ByteTrack expects via a private `_DetectionArray`
  adapter, and returns lightweight frozen `Track` objects
  (`track_id, x1, y1, x2, y2, confidence, age_frames, frames_seen, last_seen_s`).
- Bounded per-track trajectory buffer (`deque[(timestamp, center_x, center_y)]`),
  pruned every `update()` call against `analytics.trajectory_history_seconds`
  — pixel-space only, no meters conversion at this stage.
- `scripts/run_tracking_demo.py` — annotated demo video with bounding
  boxes, track ID, confidence, live people count/FPS overlay.
- `vision/benchmark.py::benchmark_detector_vs_tracker` — measures
  detector-only vs. detector+tracker latency to quantify tracking
  overhead.
- Test suite: `tests/test_tracker.py` (11 fast tests using the real
  ByteTrack implementation — ID persistence, new-track confirmation delay,
  multi-person distinctness, expiration, trajectory bounding, bbox
  conversion, session-scoped ID reset) and an opt-in
  `tests/test_tracker_integration.py` (real detector + real tracker on the
  real sample video, gated behind `CROWDSHIELD_RUN_INTEGRATION=1`).

### Fixed (B1 regression)
- **FPS semantics:** the bundled test video's `source_fps` (12) was lower
  than the default `target_fps` (15), which could make benchmark reports
  appear to process more unique frames/sec than the source contained.
  `FrameSource` now exposes `effective_target_fps = min(target_fps, source_fps)`,
  used internally for sampling, with the originally configured `target_fps`
  preserved as-is and a `WARNING` log when clamping occurs. Added
  `test_frame_source_clamps_target_fps_to_source_fps` and
  `test_frame_source_no_clamp_when_target_below_source` regression tests.
- Renamed `BenchmarkResult.effective_fps` → `measured_fps` to avoid
  confusion with the new `effective_target_fps` (sampling-rate concept vs.
  measured-throughput concept are different things).

### Verified
- Same person across consecutive frames retains the same track_id.
- New person entering gets a new track_id (respecting ByteTrack's
  real one-frame confirmation delay before a new track appears active).
- Track expires from active output once it stops matching within
  `tracking.max_age` frames.
- Multiple simultaneous people get distinct IDs.
- Trajectory buffer stores/prunes correctly and respects its configured
  maximum duration.
- Track ID counter resets to 0 for every new `PersonTracker` instance
  (session-scoped, non-persistent identity — privacy requirement).
- Manual visual inspection of `sample_data/output/tracked.mp4`: correct
  `ID:N` overlays following the same visible person; correct 0-count on
  empty frames.
- Tracking overhead measured at roughly -7ms to +3ms across repeated
  runs — within this sandbox's single-core measurement noise, consistent
  with ByteTrack (no neural network) being cheap relative to YOLO's
  ~90ms/frame cost.
- 37 tests passing, 2 integration tests skipped by default (both verified
  passing when explicitly enabled).

### Known Limitations
- Dense-crowd tracking remains unvalidated — the bundled clip shows at
  most 2 simultaneous people; no dense-crowd footage was available within
  this project's approved network domains at this stage. Flagged for
  revisit before final demo evaluation.
- Tracking overhead measurement is noise-dominated on the 1-core sandbox
  used for development; a cleaner measurement needs less-contended
  hardware or many averaged runs.
- Track IDs are expected to change across occlusion/crossing events —
  downstream crowd-flow features (Phase E) must not assume one ID = one
  person for an entire session.

## [B1] — Person Detection + Benchmarking

**Status:** Complete

### Added
- `vision/detector.py` — `PersonDetector`, a single-boundary wrapper around
  Ultralytics YOLO. Loads the model once, filters to the COCO `person`
  class, and returns lightweight frozen `Detection` objects
  (`x1, y1, x2, y2, confidence, class_id`) — no framework-specific types
  cross the `vision/` boundary. Bounding box convention (pixel coords,
  top-left origin) documented in-module for Phase C/D to build on.
- `vision/benchmark.py` — `FrameSource`, a streaming/generator-based video
  reader that samples frames to `video.target_fps` using timestamp
  comparison (not `sleep()`), and `benchmark_model()`, which measures cold
  start separately from warm steady-state inference latency and
  end-to-end frame latency.
- `scripts/run_detection_demo.py` — produces an annotated demo video
  (bounding boxes, confidence, live people count/FPS overlay).
- `scripts/run_benchmark.py` — CLI to compare YOLO model variants and
  write results to JSON.
- Real 20s pedestrian test clip at `sample_data/test_video.mp4`, sourced
  from the public `intel-iot-devkit/sample-videos` repo.
- Test suite: fast mock-based unit tests for the detector
  (`tests/test_detector.py`) and synthetic-video sampling tests
  (`tests/test_frame_source.py`); an opt-in real-model/video integration
  test (`tests/test_detector_integration.py`, gated behind
  `CROWDSHIELD_RUN_INTEGRATION=1` so the standard suite stays fast and
  network-free).
- `ultralytics` and `opencv-python-headless` added to `requirements.txt`.

### Benchmarked and selected
- Compared YOLOv8n vs YOLOv8s on the test clip (CPU, 1 core, 640px input,
  100 processed frames): nano reached 8.95 effective FPS vs. small's
  3.70 FPS, with identical average detections on this sparse clip and a
  3.5x smaller weight file. **YOLOv8n selected** as the default model per
  the "smallest model that provides sufficiently credible detection" rule.
- Model weights are not committed — Ultralytics downloads them
  automatically on first use.

### Verified
- Manual visual inspection of annotated output frames: correct bounding
  boxes, sensible confidence scores, no obvious false positives on
  clearly visible subjects; correctly reports 0 people on empty frames.
- 24 tests passing, 1 integration test skipped by default (both verified
  passing when explicitly enabled).

### Known Limitations
- Test clip is a sparse indoor pedestrian scene (max ~4 people at once),
  not a dense crowd — dense-crowd detection quality is unverified and
  flagged for revisit in Phase D/E.
- Benchmark numbers measured on a 1-core CPU sandbox; should be
  re-measured on target deployment hardware.
- No formal precision/recall — no ground-truth annotations exist for this
  footage; only qualitative sanity checks were performed.
- GPU path implemented and logged correctly but untested (no GPU in this
  environment).

## [A1] — Environment, Repository, Configuration, Logging

**Status:** Complete

### Added
- Project skeleton: `config/`, `vision/`, `features/`, `data/`, `models/`,
  `inference/`, `evaluation/`, `tests/` (placeholder packages for
  not-yet-implemented phases).
- Typed configuration schema (`config/schema.py`, pydantic v2) covering
  video, detection, tracking, analytics, risk, and venue/zone settings.
- YAML config loader (`config/loader.py`) with venue-override layering,
  loading configuration exactly once per call.
- `default.yaml` base config and `config/venues/example_venue.yaml`
  example venue override.
- Structured logging setup (`logging_setup.py`).
- Standard-library-only reproducibility seeding (`seed.py`).
- Entrypoint (`main.py`) wiring config + logging + seeding together.
- Smoke test suite (`tests/test_smoke.py`, 7 tests) covering config
  loading, venue overrides, directory structure, logging init, and a
  subprocess-isolated check that config loading never triggers heavy
  imports (torch/ultralytics/cv2/numpy/pandas).
- `requirements.txt` / `requirements-dev.txt` with a documented phased
  dependency strategy (only install what the current stage needs).

### Fixed (review round)
- **Config loading path:** `main.py` previously could load config twice
  when both `--config` and `--venue` were passed. Refactored `load_config`
  to accept both parameters directly and resolve to a single YAML read +
  merge + validate pass.
- **Pydantic v2 style:** switched `AppConfig` from the legacy inner `Config`
  class to `model_config = ConfigDict(frozen=True)`.
- **Dependency cleanliness:** confirmed A1 installs only `pydantic`,
  `PyYAML`, `pytest` — no CV/ML packages. Documented per-phase additions
  in `requirements.txt` header and README.
- **seed.py:** made standard-library-only (`random` only) for A1; NumPy
  seeding deferred to the phase where NumPy actually enters the pipeline.
- **`target_fps` semantics:** documented explicitly in `VideoConfig` and
  README that `target_fps` is the processing/inference rate, not the
  source video's native frame rate.
- **Heavy-import smoke test:** rewritten to run in an isolated subprocess
  that imports only `config.loader`/`config.schema`, so the check reflects
  what those modules themselves import rather than being sensitive to
  what else the growing test suite happens to load first.

### Verified
- `python main.py` (no args) → default config loads, ~0.22s startup.
- `python main.py --venue config/venues/example_venue.yaml` → venue
  override applied correctly (1 zone loaded).
- `python main.py --config config/default.yaml --venue config/venues/example_venue.yaml`
  → combined path works, single load.
- `pytest tests/test_smoke.py -v` → 7/7 passed in 0.34s.
- No heavy imports (torch/ultralytics/cv2/numpy/pandas) triggered by
  config loading.

### Known Limitations
- No detection/tracking/ML implemented yet (by design — Phase A scope only).
- Example venue zone coordinates/area are placeholders.
- Developed/tested on Python 3.12 in the current sandbox rather than the
  recommended 3.11; revisit before Phase B for Ultralytics/torch
  compatibility.
