"""B1 benchmark CLI: compares YOLO model variants on the test video and
prints a results table + writes JSON results for the README.

Usage:
    python scripts/run_benchmark.py --video sample_data/test_video.mp4 \
        --models yolov8n.pt yolov8s.pt --device cpu --target-fps 15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from logging_setup import setup_logging  # noqa: E402
from vision.benchmark import benchmark_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="CrowdShield B1 model benchmark")
    parser.add_argument("--video", default="sample_data/test_video.mp4")
    parser.add_argument("--models", nargs="+", default=["yolov8n.pt", "yolov8s.pt"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--output-json", default="sample_data/output/benchmark_results.json")
    args = parser.parse_args()

    logger = setup_logging("INFO")
    video_path = str(REPO_ROOT / args.video) if not Path(args.video).is_absolute() else args.video

    results = []
    for model in args.models:
        logger.info("Benchmarking %s ...", model)
        result = benchmark_model(
            video_path=video_path,
            model=model,
            confidence=args.confidence,
            device=args.device,
            image_size=args.image_size,
            target_fps=args.target_fps,
            max_frames=args.max_frames,
        )
        results.append(result.as_dict())
        logger.info("  -> %s", result.as_dict())

    output_path = Path(args.output_json)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    logger.info("Benchmark results written to: %s", output_path)

    header = f"{'Model':<12}{'Device':<8}{'Eff.FPS':<10}{'InferMs':<10}{'E2EMs':<10}{'AvgPeople':<10}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['model']:<12}{r['device']:<8}{r['measured_fps']:<10}"
            f"{r['avg_inference_latency_ms']:<10}{r['avg_end_to_end_latency_ms']:<10}"
            f"{r['avg_people_detected']:<10}"
        )


if __name__ == "__main__":
    main()
