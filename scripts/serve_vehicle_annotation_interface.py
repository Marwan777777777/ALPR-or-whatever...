"""Mission 012: Vehicle Ground Truth Annotation Server.

Lightweight HTTP server providing:
- Safe image streaming from D:/TruckEntry via MediaLoader (bypassing Windows Unicode path issues)
- Anonymized sample indexing (zero exposure of legacy plate text)
- REST API for saving and loading independent vehicle ground truth annotations
"""

from __future__ import annotations

from datetime import datetime
import glob
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import os
from pathlib import Path
import re
import sys
from typing import Dict, List, Tuple
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
from vehicle_intelligence.pipeline.media_loader import MediaLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
INTERFACE_DIR = ROOT_DIR / "artifacts" / "missions" / "mission_012" / "annotation_interface"
ANNOT_DIR = ROOT_DIR / "artifacts" / "missions" / "mission_012" / "annotations"
GROUND_TRUTH_FILE = ROOT_DIR / "artifacts" / "missions" / "mission_012" / "vehicle_ground_truth.json"

GATE_DIR = Path("D:/TruckEntry")


def parse_timestamp_from_filename(filename: str) -> Tuple[datetime, float]:
    m = re.match(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_(\d{3})", filename)
    if not m:
        raise ValueError(f"Could not parse timestamp from {filename}")
    year, month, day, hour, minute, second, ms = [int(v) for v in m.groups()]
    dt = datetime(year, month, day, hour, minute, second, ms * 1000)
    return dt, dt.timestamp()


def categorize_time_of_day(dt: datetime) -> str:
    hour = dt.hour + dt.minute / 60.0
    if hour < 6.5:
        return "NIGHT_IR"
    elif hour < 17.5:
        return "DAYLIGHT"
    else:
        return "EVENING_IR"


def build_sample_manifest() -> List[Dict]:
    files = sorted(list(GATE_DIR.glob("*.jpg")))
    samples = []
    for idx, f in enumerate(files):
        fname = f.name
        dt, ts = parse_timestamp_from_filename(fname)
        lighting = categorize_time_of_day(dt)
        sample_id = f"sample_{idx + 1:03d}"
        anon_id = f"SAMPLE-{idx + 1:03d}"
        samples.append({
            "index": idx,
            "sample_id": sample_id,
            "anonymous_id": anon_id,
            "filename": fname,
            "timestamp": dt.strftime("%H:%M:%S"),
            "full_timestamp": dt.isoformat(),
            "epoch": ts,
            "lighting_condition": lighting,
            "image_url": f"/api/image?index={idx}",
            "width": 3840,
            "height": 2160,
        })
    return samples


SAMPLES_CACHE = build_sample_manifest()


class AnnotationServerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(INTERFACE_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/images":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            resp = {"total": len(SAMPLES_CACHE), "samples": SAMPLES_CACHE}
            self.wfile.write(json.dumps(resp, ensure_ascii=False).encode("utf-8"))
            return

        elif parsed.path == "/api/image":
            qs = parse_qs(parsed.query)
            idx_str = qs.get("index", ["0"])[0]
            try:
                idx = int(idx_str)
                if idx < 0 or idx >= len(SAMPLES_CACHE):
                    self.send_error(404, "Sample index out of range")
                    return
                sample = SAMPLES_CACHE[idx]
                file_path = GATE_DIR / sample["filename"]
                img = MediaLoader.load_image(file_path)
                if img is None:
                    self.send_error(500, "Failed to decode image")
                    return

                # Encode to JPEG
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    self.send_error(500, "Failed to encode JPEG")
                    return

                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(buf)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(buf.tobytes())
                return
            except Exception as e:
                self.send_error(500, str(e))
                return

        elif parsed.path == "/api/annotations/load":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            if GROUND_TRUTH_FILE.is_file():
                with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
                    content = f.read()
                self.wfile.write(content.encode("utf-8"))
            else:
                self.wfile.write(b"{}")
            return

        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/annotations/save":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
                ANNOT_DIR.mkdir(parents=True, exist_ok=True)
                GROUND_TRUTH_FILE.parent.mkdir(parents=True, exist_ok=True)

                # Save backup and ground truth
                with open(GROUND_TRUTH_FILE, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = ANNOT_DIR / f"annotations_{timestamp_str}.json"
                with open(backup_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "saved", "count": len(data)}).encode("utf-8"))
                return
            except Exception as e:
                self.send_error(500, str(e))
                return

        self.send_error(404, "Not Found")


def run_server(port: int = 8090):
    server_address = ("", port)
    httpd = HTTPServer(server_address, AnnotationServerHandler)
    print(f"Mission 012 Annotation Interface running at: http://localhost:{port}/")
    print(f"Serving {len(SAMPLES_CACHE)} gate captures from {GATE_DIR}")
    print("Press Ctrl+C to terminate server.")
    httpd.serve_forever()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
    run_server(port)
