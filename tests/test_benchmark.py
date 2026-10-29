"""Tests for benchmark runner and report generation."""

from pathlib import Path
from vehicle_intelligence.benchmark.runner import BenchmarkRunner
from vehicle_intelligence.detectors.mock_detector import MockVehicleDetector


def test_benchmark_runner_executes_and_saves_report(tmp_path: Path):
    detector = MockVehicleDetector()
    runner = BenchmarkRunner(
        detector=detector,
        warmup_iterations=3,
        benchmark_iterations=10,
        input_resolution=(320, 320)
    )

    report = runner.run(test_pipeline=True)

    assert report.iterations == 10
    assert report.warmup_iterations == 3
    assert report.detection_metrics.mean_ms > 0
    assert report.detection_metrics.fps > 0
    assert report.pipeline_metrics is not None
    assert report.pipeline_metrics.fps > 0

    json_path, md_path = runner.save_report(report, output_dir=tmp_path)
    assert json_path.exists()
    assert md_path.exists()

    md_text = md_path.read_text(encoding="utf-8")
    assert "Vehicle Intelligence Benchmark Report" in md_text
    assert "Detector Latency & Throughput" in md_text
    assert "Hardware Profile" in md_text
