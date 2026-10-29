#!/usr/bin/env python3
"""CLI utility to execute and save reproducible computer vision benchmarks."""

import argparse
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminals for Arabic characters
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from vehicle_intelligence.benchmark.runner import BenchmarkRunner
from vehicle_intelligence.core.logging import setup_logger
from vehicle_intelligence.detectors.mock_detector import MockVehicleDetector
from vehicle_intelligence.detectors.motion_detector import MotionVehicleDetector
from vehicle_intelligence.pipeline.media_loader import MediaLoader


def main():
    parser = argparse.ArgumentParser(description="Run vehicle intelligence benchmark suite.")
    parser.add_argument(
        "--detector",
        choices=["mock", "motion"],
        default="mock",
        help="Detector engine to benchmark ('mock' or 'motion')"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of iterations to benchmark (default: 50)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warm-up iterations (default: 10)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional path to a test image file. If omitted, synthetic frames are generated."
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="640x640",
        help="Synthetic frame resolution WxH (default: 640x640)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/benchmark_reports",
        help="Directory to save JSON and Markdown benchmark reports."
    )

    args = parser.parse_args()
    logger = setup_logger("benchmark_cli")

    logger.info(f"Initializing benchmark for detector='{args.detector}' with {args.iterations} iterations...")

    # Select detector
    if args.detector == "mock":
        detector = MockVehicleDetector()
    elif args.detector == "motion":
        detector = MotionVehicleDetector()
    else:
        raise ValueError(f"Unknown detector: {args.detector}")

    # Parse resolution
    w_str, h_str = args.resolution.lower().split("x")
    res = (int(w_str), int(h_str))

    test_image = None
    if args.image:
        logger.info(f"Loading custom test image: {args.image}")
        test_image = MediaLoader.load_image(args.image)

    runner = BenchmarkRunner(
        detector=detector,
        warmup_iterations=args.warmup,
        benchmark_iterations=args.iterations,
        input_resolution=res
    )

    report = runner.run(test_image=test_image, test_pipeline=True)
    json_path, md_path = runner.save_report(report, output_dir=args.output_dir)

    logger.info(f"Benchmark completed successfully! Benchmark ID: {report.benchmark_id}")
    logger.info(f"Detection Latency: {report.detection_metrics.mean_ms:.2f} ms ({report.detection_metrics.fps:.1f} FPS)")
    if report.pipeline_metrics:
        logger.info(f"Pipeline Latency:  {report.pipeline_metrics.mean_ms:.2f} ms ({report.pipeline_metrics.fps:.1f} FPS)")
    logger.info(f"Reports saved to:\n  - JSON: {json_path}\n  - Markdown: {md_path}")

    # Print summary to stdout
    print("\n" + BenchmarkRunner.generate_markdown_summary(report))


if __name__ == "__main__":
    main()
