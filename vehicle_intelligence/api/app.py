"""Thin REST API for Egyptian ALPR (Mission 6 Part D).
Exposes:
- GET /health: Health check status
- POST /v1/read-plate: Image upload returning List[ALPRPlateResult]
- Automatic Swagger /docs endpoint
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import JSONResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from vehicle_intelligence.core.domain import ALPRPlateResult
from vehicle_intelligence.pipeline.alpr_pipeline import EndToEndALPRPipeline

# Pipeline singleton instance
_pipeline: EndToEndALPRPipeline | None = None


def get_pipeline() -> EndToEndALPRPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = EndToEndALPRPipeline()
    return _pipeline


if HAS_FASTAPI:
    app = FastAPI(
        title="Vehicle Intelligence Egyptian ALPR Service",
        description="Edge AI REST API for Egyptian License Plate Recognition and Verification",
        version="1.0.0",
    )

    @app.get("/health")
    def health_check():
        return {
            "status": "healthy",
            "service": "vehicle-intelligence-alpr",
            "timestamp": time.time(),
        }

    @app.post("/v1/read-plate", response_model=List[ALPRPlateResult])
    async def read_plate(file: UploadFile = File(...)):
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image (JPEG, PNG).")

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")

        pipeline = get_pipeline()
        results = pipeline.read_plate_image(image=img, source_id=file.filename or "upload")
        return results

else:
    app = None


# Standalone runner fallback using Python standard library http.server if FastAPI is not installed
if __name__ == "__main__":
    import argparse
    import json
    from http.server import BaseHTTPRequestHandler, HTTPServer

    parser = argparse.ArgumentParser(description="Start ALPR HTTP API server.")
    parser.add_argument("--host", default="0.0.0.0", help="Host binding")
    parser.add_argument("--port", type=int, default=8000, help="Port binding")
    args = parser.parse_args()

    if HAS_FASTAPI:
        import uvicorn
        print(f"Starting FastAPI server on {args.host}:{args.port} (Interactive docs at http://{args.host}:{args.port}/docs)...")
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        class StandaloneHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "status": "healthy",
                        "service": "vehicle-intelligence-alpr",
                        "timestamp": time.time(),
                    }).encode("utf-8"))
                elif self.path in {"/docs", "/"}:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(b"<h1>Egyptian ALPR API</h1><p>GET /health</p><p>POST /v1/read-plate</p>")
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                if self.path == "/v1/read-plate":
                    content_length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(content_length)
                    nparr = np.frombuffer(body, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is None:
                        self.send_response(400)
                        self.end_headers()
                        return
                    pipeline = get_pipeline()
                    res = pipeline.read_plate_image(img, source_id="http_post")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(json.dumps([r.model_dump() for r in res], ensure_ascii=False).encode("utf-8"))
                else:
                    self.send_response(404)
                    self.end_headers()

        server = HTTPServer((args.host, args.port), StandaloneHandler)
        print(f"Starting standalone ALPR server on http://{args.host}:{args.port}...")
        server.serve_forever()
