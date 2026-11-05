"""
utils.py
========
Shared helpers for drawing detections, saving snapshots, and rendering
on-screen HUD information for the gesture & face detection application.

Author: Bhoomika Gandhi (Arrowstack Batch 2026, AI domain)
Ref ID: AS-2026-26671
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import List

import cv2
import numpy as np

from face_detection import Detection
from gesture_recognition import Gesture


# Colour palette (BGR tuples for OpenCV).
COLOR_FACE = (255, 200, 0)        # warm yellow
COLOR_FACE_PROFILE = (0, 180, 255)  # orange
COLOR_GESTURE = (0, 255, 120)     # mint green
COLOR_HUD_BG = (40, 40, 40)
COLOR_HUD_TEXT = (240, 240, 240)
COLOR_WARN = (0, 80, 255)


def draw_faces(image: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """Draw bounding boxes and labels for face detections."""
    annotated = image.copy()
    for det in detections:
        color = COLOR_FACE if det.label == "face_frontal" else COLOR_FACE_PROFILE
        cv2.rectangle(annotated, (det.x, det.y), (det.x + det.w, det.y + det.h), color, 2)
        label = f"{det.label} {det.confidence:.2f}"
        _draw_label(annotated, label, (det.x, det.y - 8), color)
    return annotated


def draw_gestures(image: np.ndarray, gestures: List[Gesture]) -> np.ndarray:
    """Draw bounding boxes, gesture names, and confidence for each hand."""
    annotated = image.copy()
    for g in gestures:
        x, y, w, h = g.bbox
        cv2.rectangle(annotated, (x, y), (x + w, y + h), COLOR_GESTURE, 2)
        label = f"{g.name} ({g.hand}, {g.confidence:.2f})"
        _draw_label(annotated, label, (x, y - 8), COLOR_GESTURE)
    return annotated


def _draw_label(image: np.ndarray, text: str, origin, color) -> None:
    x, y = origin
    if y < 12:
        y = 12
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(
        image,
        (x, y - th - 4),
        (x + tw + 4, y + 4),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x + 2, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def draw_hud(image: np.ndarray, lines: List[str]) -> np.ndarray:
    """Render a simple HUD panel in the top-left corner."""
    annotated = image.copy()
    panel_w = 360
    panel_h = 24 * max(len(lines), 1) + 12
    cv2.rectangle(annotated, (8, 8), (8 + panel_w, 8 + panel_h), COLOR_HUD_BG, -1)
    for i, line in enumerate(lines):
        y = 28 + i * 24
        cv2.putText(
            annotated,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_HUD_TEXT,
            1,
            cv2.LINE_AA,
        )
    return annotated


def save_snapshot(image: np.ndarray, directory: str, prefix: str = "snap") -> str:
    """Save a timestamped PNG snapshot. Returns the absolute path."""
    os.makedirs(directory, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(directory, f"{prefix}_{ts}.png")
    cv2.imwrite(path, image)
    return path


class FpsCounter:
    """A tiny rolling-average FPS counter."""

    def __init__(self, window: int = 30) -> None:
        self._times: List[float] = []
        self._window = window

    def tick(self) -> None:
        now = time.time()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)

    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._times) - 1) / elapsed
