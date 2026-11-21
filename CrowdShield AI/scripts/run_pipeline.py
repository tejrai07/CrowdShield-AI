"""L1 CLI: run the end-to-end CrowdShield pipeline on a video and write one
JSON object per processed frame (JSON Lines).

    python scripts/run_pipeline.py --video sample_data/test_video.mp4 \
        --venue config/venues/example_venue.yaml --max-frames 24 \
        --output sample_data/output/pipeline_output.jsonl --validate

Exit codes: 0 success | 1 runtime/pipeline failure or schema violation | 2 bad arguments, paths or config.
Use `--output -` for stdout (logs then go to stderr). YOLO weights are fetched by
Ultralytics on first run if absent; models must exist in --model-dir.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config.loader import load_config  # noqa: E402
from config.schema import AppConfig  # noqa: E402
from inference.pipeline import InferencePipeline, PipelineError  # noqa: E402
from inference.schema import validate_result  # noqa: E402
from logging_setup import setup_logging  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CrowdShield L1 end-to-end pipeline: video -> per-frame JSON",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--video", required=True, help="path to the input video")
    p.add_argument("--config", default=None, help="base config YAML (default: config/default.yaml)")
    p.add_argument("--venue", default="config/venues/example_venue.yaml", help="venue/zone-graph YAML")
    p.add_argument("--model-dir", default=None, help="override inference.model_dir (H1/I1 artifacts)")
    p.add_argument("--max-frames", type=int, default=None, help="stop after N processed frames (bounded run)")
    p.add_argument("--output", default="sample_data/output/pipeline_output.jsonl", help="JSONL output path, or '-' for stdout")
    p.add_argument("--explain", choices=["off", "alerts", "all"], default=None, help="override inference.explain_mode")
    p.add_argument("--validate", action="store_true", help="validate every result against the documented schema; exit 1 on violation")
    p.add_argument("--log-level", default="INFO")
    return p


def _resolve(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return REPO_ROOT / path


def run_cli(argv: list[str] | None = None, pipeline_factory: Callable[[AppConfig], InferencePipeline] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = setup_logging(args.log_level)
    to_stdout = args.output == "-"
    if to_stdout:
        for handler in logger.handlers:
            handler.setStream(sys.stderr)

    if args.max_frames is not None and args.max_frames < 0:
        print("error: --max-frames must be >= 0", file=sys.stderr)
        return 2
    video = _resolve(args.video)
    if not video.is_file():
        print(f"error: video not found: {video}", file=sys.stderr)
        return 2
    try:
        cfg = load_config(_resolve(args.config) if args.config else None, _resolve(args.venue) if args.venue else None)
        overrides = {k: v for k, v in (("model_dir", args.model_dir), ("explain_mode", args.explain)) if v is not None}
        if overrides:
            cfg = AppConfig.model_validate({**cfg.model_dump(), "inference": {**cfg.inference.model_dump(), **overrides}})
    except Exception as exc:  # missing files, YAML errors, pydantic validation
        print(f"error: invalid configuration: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    try:
        pipeline = (pipeline_factory or (lambda c: InferencePipeline.from_config(c, repo_root=REPO_ROOT)))(cfg)
        from vision.benchmark import FrameSource
        source = FrameSource(str(video), target_fps=cfg.video.target_fps)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (PipelineError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out = sys.stdout if to_stdout else None
    if not to_stdout:
        out_path = _resolve(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out = open(out_path, "w", encoding="utf-8")
    statuses: Counter = Counter()
    rec_statuses: Counter = Counter()
    low_reliability_recs = 0
    provisional_recs = 0
    frames = violations = 0
    try:
        for result in pipeline.run(source, max_frames=args.max_frames):
            line = json.dumps(result, allow_nan=False)  # raises if NaN/Infinity ever leaks
            out.write(line + "\n")
            frames += 1
            statuses[result["pipeline_status"]] += 1
            rec_statuses.update(r["status"] for r in result["diversion_recommendations"])
            low_reliability_recs += sum(1 for r in result["diversion_recommendations"] if not r["source_inputs_reliable"])
            provisional_recs += sum(1 for r in result["diversion_recommendations"] if r["reliability"] == "provisional")
            if args.validate:
                errors = validate_result(json.loads(line))
                if errors:
                    violations += 1
                    logger.error("frame %s violates schema: %s", result["frame_id"], errors[:3])
    except PipelineError as exc:
        print(f"error: pipeline failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if not to_stdout and out is not None:
            out.close()
    logger.info("processed %d frame(s) | pipeline_status=%s | recommendations=%s", frames, dict(statuses), dict(rec_statuses))
    if provisional_recs:
        logger.warning("%d recommendation(s) are PROVISIONAL: they rest only on model output for zones whose inputs are out-of-distribution "
                       "(reliability=provisional) - not supported conclusions", provisional_recs)
    elif low_reliability_recs:
        logger.info("%d recommendation(s) came from zones with out-of-distribution inputs but rest on a model-independent rule (reliability=reliable)", low_reliability_recs)
    if args.validate:
        logger.info("schema validation: %s", "FAILED on %d frame(s)" % violations if violations else "all frames valid")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(run_cli())
