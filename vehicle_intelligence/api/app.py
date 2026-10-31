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

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

HAS_FASTAPI = True

from vehicle_intelligence.core.domain import ALPRPlateResult
from vehicle_intelligence.pipeline.alpr_pipeline import EndToEndALPRPipeline

# Pipeline singleton instance
_pipeline: EndToEndALPRPipeline | None = None


def get_pipeline(
    det_imgsz: int | None = None,
    char_iou: float | None = None,
    agnostic_nms: bool | None = None,
) -> EndToEndALPRPipeline:
    global _pipeline
    if _pipeline is None:
        import os
        imgsz = det_imgsz or int(os.environ.get("ALPR_DET_IMGSZ", 640))
        iou = char_iou if char_iou is not None else float(os.environ.get("ALPR_CHAR_IOU", 0.45))
        agn = agnostic_nms if agnostic_nms is not None else (os.environ.get("ALPR_AGNOSTIC_NMS", "true").lower() in ("true", "1"))
        _pipeline = EndToEndALPRPipeline(
            det_imgsz=imgsz,
            char_iou=iou,
            agnostic_nms=agn,
        )
    return _pipeline


def reset_pipeline() -> None:
    """Resets the singleton instance (useful for test isolation)."""
    global _pipeline
    _pipeline = None


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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start ALPR HTTP API server.")
    parser.add_argument("--host", default="0.0.0.0", help="Host binding")
    parser.add_argument("--port", type=int, default=8000, help="Port binding")
    args = parser.parse_args()

    print(f"Starting FastAPI server on {args.host}:{args.port} (Interactive docs at http://{args.host}:{args.port}/docs)...")
    uvicorn.run(app, host=args.host, port=args.port)

