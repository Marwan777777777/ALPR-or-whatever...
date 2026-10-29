"""Reproducible benchmarking engine measuring latency, throughput, and hardware utilization."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field

from vehicle_intelligence.core.domain import VehicleDetection
from vehicle_intelligence.core.interfaces import BaseVehicleDetector
from vehicle_intelligence.pipeline.engine import VehicleIntelligenceEngine


class LatencyMetrics(BaseModel):
    mean_ms: float
    std_ms: float
    min_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    fps: float


class HardwareInfo(BaseModel):
    os_name: str
    python_version: str
    cpu_architecture: str
    gpu_name: Optional[str] = None
    gpu_driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    total_gpu_memory_mb: Optional[float] = None


class BenchmarkReport(BaseModel):
    benchmark_id: str
    timestamp_utc: str
    detector_name: str
    iterations: int
    warmup_iterations: int
    resolution: str
    hardware: HardwareInfo
    detection_metrics: LatencyMetrics
    pipeline_metrics: Optional[LatencyMetrics] = None
    sample_detections_count: int


class BenchmarkRunner:
    """Executes reproducible latency and throughput benchmarks on image/video frames."""

    def __init__(
        self,
        detector: BaseVehicleDetector,
        warmup_iterations: int = 10,
        benchmark_iterations: int = 50,
        input_resolution: tuple[int, int] = (640, 640)
    ):
        self.detector = detector
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.resolution = input_resolution

    def _query_hardware(self) -> HardwareInfo:
        """Collects OS, CPU, and NVIDIA GPU telemetry if present."""
        gpu_name = None
        driver_ver = None
        cuda_ver = None
        gpu_mem = None

        # Attempt nvidia-smi query
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits"
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
            if res.returncode == 0 and res.stdout.strip():
                parts = [p.strip() for p in res.stdout.strip().split(",")]
                if len(parts) >= 3:
                    gpu_name = parts[0]
                    driver_ver = parts[1]
                    gpu_mem = float(parts[2])
        except Exception:
            pass

        return HardwareInfo(
            os_name=platform.platform(),
            python_version=platform.python_version(),
            cpu_architecture=platform.processor() or platform.machine(),
            gpu_name=gpu_name,
            gpu_driver_version=driver_ver,
            cuda_version=cuda_ver,
            total_gpu_memory_mb=gpu_mem,
        )

    def _calculate_metrics(self, latencies_ms: List[float]) -> LatencyMetrics:
        arr = np.array(latencies_ms)
        mean_ms = float(np.mean(arr))
        fps = (1000.0 / mean_ms) if mean_ms > 0 else 0.0

        return LatencyMetrics(
            mean_ms=round(mean_ms, 3),
            std_ms=round(float(np.std(arr)), 3),
            min_ms=round(float(np.min(arr)), 3),
            p50_ms=round(float(np.percentile(arr, 50)), 3),
            p95_ms=round(float(np.percentile(arr, 95)), 3),
            p99_ms=round(float(np.percentile(arr, 99)), 3),
            max_ms=round(float(np.max(arr)), 3),
            fps=round(fps, 1),
        )

    def run(
        self,
        test_image: Optional[np.ndarray] = None,
        test_pipeline: bool = True
    ) -> BenchmarkReport:
        """Executes the benchmark and returns a structured BenchmarkReport."""
        w, h = self.resolution
        img = test_image if test_image is not None else np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

        # 1. Warm-up phase
        for _ in range(self.warmup_iterations):
            _ = self.detector.detect(img)

        # 2. Benchmark detector alone
        detector_latencies: List[float] = []
        sample_detections: List[VehicleDetection] = []
        for _ in range(self.benchmark_iterations):
            t0 = time.perf_counter()
            dets = self.detector.detect(img)
            t1 = time.perf_counter()
            detector_latencies.append((t1 - t0) * 1000.0)
            if not sample_detections and dets:
                sample_detections = dets

        det_metrics = self._calculate_metrics(detector_latencies)

        # 3. Benchmark end-to-end pipeline (detection + tracking + plate + OCR)
        pipeline_metrics = None
        if test_pipeline:
            engine = VehicleIntelligenceEngine(vehicle_detector=self.detector)
            pipeline_latencies: List[float] = []
            for frame_idx in range(self.benchmark_iterations):
                t0 = time.perf_counter()
                _ = engine.process_frame(img, frame_id=frame_idx)
                t1 = time.perf_counter()
                pipeline_latencies.append((t1 - t0) * 1000.0)
            pipeline_metrics = self._calculate_metrics(pipeline_latencies)

        hardware = self._query_hardware()
        benchmark_id = f"bench_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        return BenchmarkReport(
            benchmark_id=benchmark_id,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            detector_name=self.detector.model_name,
            iterations=self.benchmark_iterations,
            warmup_iterations=self.warmup_iterations,
            resolution=f"{w}x{h}",
            hardware=hardware,
            detection_metrics=det_metrics,
            pipeline_metrics=pipeline_metrics,
            sample_detections_count=len(sample_detections),
        )

    def save_report(
        self,
        report: BenchmarkReport,
        output_dir: Path | str = "artifacts/benchmark_reports"
    ) -> tuple[Path, Path]:
        """Saves benchmark results to both JSON and Markdown formats."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / f"{report.benchmark_id}.json"
        md_path = out_dir / f"{report.benchmark_id}.md"

        # Save JSON
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

        # Save Markdown
        md_content = self.generate_markdown_summary(report)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return json_path, md_path

    @staticmethod
    def generate_markdown_summary(report: BenchmarkReport) -> str:
        hw = report.hardware
        det = report.detection_metrics
        pipe = report.pipeline_metrics

        md = f"""# Vehicle Intelligence Benchmark Report: `{report.benchmark_id}`

**Timestamp (UTC):** {report.timestamp_utc}  
**Detector:** `{report.detector_name}`  
**Test Resolution:** {report.resolution}  
**Iterations:** {report.iterations} (Warmup: {report.warmup_iterations})  

## Hardware Profile
| Property | Value |
|---|---|
| OS | {hw.os_name} |
| Python | {hw.python_version} |
| CPU | {hw.cpu_architecture} |
| GPU | {hw.gpu_name or 'N/A (CPU execution)'} |
| Driver Version | {hw.gpu_driver_version or 'N/A'} |
| GPU VRAM | {f'{hw.total_gpu_memory_mb:.0f} MB' if hw.total_gpu_memory_mb else 'N/A'} |

## Detector Latency & Throughput
| Metric | Latency (ms) |
|---|---|
| **Mean** | **{det.mean_ms} ms** |
| **Median (p50)** | {det.p50_ms} ms |
| **95th percentile (p95)** | {det.p95_ms} ms |
| **99th percentile (p99)** | {det.p99_ms} ms |
| Min / Max | {det.min_ms} ms / {det.max_ms} ms |
| Std Dev | {det.std_ms} ms |
| **Throughput** | **{det.fps} FPS** |

"""
        if pipe:
            md += f"""## End-to-End Pipeline Latency (Detect + Track + Plate + OCR)
| Metric | Latency (ms) |
|---|---|
| **Mean** | **{pipe.mean_ms} ms** |
| **Median (p50)** | {pipe.p50_ms} ms |
| **95th percentile (p95)** | {pipe.p95_ms} ms |
| **99th percentile (p99)** | {pipe.p99_ms} ms |
| **Pipeline Throughput** | **{pipe.fps} FPS** |
"""
        return md
