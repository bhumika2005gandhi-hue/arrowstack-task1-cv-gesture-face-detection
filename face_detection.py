"""
face_detection.py
=================
Face detection module for the Arrowstack AI Internship Task 1 project.

This module wraps OpenCV's Haar cascade and (optionally) DNN-based face
detectors behind a small, consistent interface so the rest of the
application does not depend on a specific backend.

Design goals
------------
* Backend-agnostic: callers use ``FaceDetector.detect(image)`` and receive
  a list of :class:`Detection` dataclasses. The backend can be swapped by
  changing one line in :class:`FaceDetector`.
* Graceful degradation: if the DNN model file is missing we automatically
  fall back to the Haar cascade, which ships with OpenCV and therefore
  always works.
* Reproducible: every detection carries a confidence score and a backend
  tag so results can be logged and compared during evaluation.

Author: Bhoomika Gandhi (Arrowstack Batch 2026, AI domain)
Ref ID: AS-2026-26671
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    """A single detection returned by any detector in this project."""

    x: int
    y: int
    w: int
    h: int
    confidence: float
    label: str
    backend: str

    @property
    def area(self) -> int:
        return int(self.w) * int(self.h)

    @property
    def center(self) -> Tuple[int, int]:
        return (int(self.x + self.w / 2), int(self.y + self.h / 2))

    def to_dict(self) -> dict:
        return {
            "x": int(self.x),
            "y": int(self.y),
            "w": int(self.w),
            "h": int(self.h),
            "confidence": float(self.confidence),
            "label": self.label,
            "backend": self.backend,
        }


class FaceDetector:
    """Detect frontal and profile faces in BGR images.

    Parameters
    ----------
    backend:
        ``"haar"`` (default, always available) or ``"dnn"`` (uses a Caffe
        model if present in ``model_dir``).
    model_dir:
        Directory that may contain ``res10_300x300_ssd_iter_140000.caffemodel``
        and ``deploy.prototxt``. If both files exist, the DNN backend can be
        used. If they are absent, the constructor transparently falls back
        to the Haar cascade.
    scale_factor, min_neighbors, min_size:
        Haar cascade tuning parameters. Defaults are conservative and work
        well on webcam frames at 640x480.
    """

    HAAR_FRONTAL = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    HAAR_PROFILE = cv2.data.haarcascades + "haarcascade_profileface.xml"

    def __init__(
        self,
        backend: str = "haar",
        model_dir: Optional[str] = None,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (48, 48),
    ) -> None:
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

        if not os.path.exists(self.HAAR_FRONTAL):
            raise FileNotFoundError(
                f"Haar cascade file missing: {self.HAAR_FRONTAL}. "
                "Your OpenCV install is incomplete."
            )

        self._frontal = cv2.CascadeClassifier(self.HAAR_FRONTAL)
        self._profile = cv2.CascadeClassifier(self.HAAR_PROFILE)

        self.backend = backend
        self._net = None
        if backend == "dnn":
            self._net = self._try_load_dnn(model_dir)
            if self._net is None:
                # Transparent fallback so the application still runs.
                self.backend = "haar"

    # ------------------------------------------------------------------
    # Backend loading
    # ------------------------------------------------------------------
    @staticmethod
    def _try_load_dnn(model_dir: Optional[str]) -> Optional[cv2.dnn.Net]:
        if not model_dir:
            return None
        weights = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        proto = os.path.join(model_dir, "deploy.prototxt")
        if not (os.path.exists(weights) and os.path.exists(proto)):
            return None
        try:
            return cv2.dnn.readNetFromCaffe(proto, weights)
        except cv2.error:
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Detection]:
        """Return a list of face detections for a BGR image."""
        if image is None or image.size == 0:
            return []
        if self.backend == "dnn" and self._net is not None:
            return self._detect_dnn(image, confidence_threshold)
        return self._detect_haar(image)

    def _detect_haar(self, image: np.ndarray) -> List[Detection]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        detections: List[Detection] = []

        for cascade, label in (
            (self._frontal, "face_frontal"),
            (self._profile, "face_profile"),
        ):
            boxes = cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            for (x, y, w, h) in boxes:
                detections.append(
                    Detection(
                        x=int(x),
                        y=int(y),
                        w=int(w),
                        h=int(h),
                        confidence=0.8,  # Haar cascade does not score; use a fixed conservative value
                        label=label,
                        backend="haar",
                    )
                )

        # Remove duplicate frontal/profile overlaps (keep the higher box).
        return FaceDetector._nms(detections, iou_threshold=0.3)

    def _detect_dnn(self, image: np.ndarray, confidence_threshold: float) -> List[Detection]:
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
            swapRB=False,
            crop=False,
        )
        self._net.setInput(blob)
        raw = self._net.forward()
        detections: List[Detection] = []
        for i in range(raw.shape[2]):
            confidence = float(raw[0, 0, i, 2])
            if confidence < confidence_threshold:
                continue
            x1 = int(raw[0, 0, i, 3] * w)
            y1 = int(raw[0, 0, i, 4] * h)
            x2 = int(raw[0, 0, i, 5] * w)
            y2 = int(raw[0, 0, i, 6] * h)
            detections.append(
                Detection(
                    x=max(x1, 0),
                    y=max(y1, 0),
                    w=max(x2 - x1, 1),
                    h=max(y2 - y1, 1),
                    confidence=confidence,
                    label="face_frontal",
                    backend="dnn",
                )
            )
        return detections

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _iou(a: Detection, b: Detection) -> float:
        ax2, ay2 = a.x + a.w, a.y + a.h
        bx2, by2 = b.x + b.w, b.y + b.h
        ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        union = a.area + b.area - inter
        return inter / float(union) if union > 0 else 0.0

    @staticmethod
    def _nms(detections: List[Detection], iou_threshold: float) -> List[Detection]:
        """Lightweight non-maximum suppression to drop duplicate detections."""
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: List[Detection] = []
        for det in detections:
            if any(FaceDetector._iou(det, k) > iou_threshold for k in kept):
                continue
            kept.append(det)
        return kept
