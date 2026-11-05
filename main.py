#!/usr/bin/env python3
"""
main.py
=======
Arrowstack AI Internship - Task 1
Computer Vision Gesture & Face Detection with OpenCV

Entry point for the real-time webcam demo. Press::

    q  - quit
    s  - save a snapshot to ./snapshots/
    h  - toggle HUD
    m  - toggle mirror

Examples
--------
Run the live webcam demo::

    python main.py

Run on a static image (no camera needed, useful for headless machines)::

    python main.py --image path/to/photo.jpg

Run on a folder of images and write annotated copies alongside::

    python main.py --batch path/to/folder --output annotated/

Author: Bhoomika Gandhi (Arrowstack Batch 2026, AI domain)
Ref ID: AS-2026-26671
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import List, Optional

import cv2
import numpy as np

from config import CONFIG, PROJECT_ROOT
from face_detection import Detection, FaceDetector
from gesture_recognition import Gesture, GestureRecognizer, is_mediapipe_available
from utils import FpsCounter, draw_faces, draw_gestures, draw_hud, save_snapshot


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
def setup_logging(log_file: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("arrowstack.task1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# ----------------------------------------------------------------------
# Core pipeline
# ----------------------------------------------------------------------
class VisionPipeline:
    """Wrap the detectors so the webcam loop, image, and batch paths
    share the same processing logic."""

    def __init__(
        self,
        face_detector: FaceDetector,
        gesture_recognizer: Optional[GestureRecognizer],
        logger: logging.Logger,
    ) -> None:
        self.face_detector = face_detector
        self.gesture_recognizer = gesture_recognizer
        self.logger = logger

    def process(self, frame: np.ndarray) -> tuple:
        """Return (annotated_frame, faces, gestures) for a BGR frame."""
        faces = self.face_detector.detect(frame)
        gestures: List[Gesture] = []
        if self.gesture_recognizer is not None:
            gestures = self.gesture_recognizer.detect(frame)
            frame = self.gesture_recognizer.draw(frame, gestures)

        annotated = draw_faces(frame, faces)
        annotated = draw_gestures(annotated, gestures)
        return annotated, faces, gestures

    def close(self) -> None:
        if self.gesture_recognizer is not None:
            self.gesture_recognizer.close()


# ----------------------------------------------------------------------
# Live webcam loop
# ----------------------------------------------------------------------
def run_webcam(pipeline: VisionPipeline, cfg: AppConfig, logger: logging.Logger) -> None:  # type: ignore[name-defined]
    cap = cv2.VideoCapture(cfg.camera_index, cv2.CAP_ANY)
    if not cap.isOpened():
        logger.error("Could not open webcam at index %d", cfg.camera_index)
        raise RuntimeError(f"Webcam index {cfg.camera_index} unavailable")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)

    fps = FpsCounter()
    show_hud = cfg.show_hud
    mirror = cfg.mirror_camera
    logger.info("Webcam session started. Press 'q' to quit, 's' to snapshot.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.warning("Empty frame received from camera.")
                break
            if mirror:
                frame = cv2.flip(frame, 1)

            annotated, faces, gestures = pipeline.process(frame)
            fps.tick()

            if show_hud:
                hud = [
                    f"Arrowstack AI Task 1  |  ref AS-2026-26671",
                    f"FPS: {fps.fps():.1f}   faces: {len(faces)}   hands: {len(gestures)}",
                    f"Press 'q' quit | 's' snapshot | 'h' HUD | 'm' mirror",
                ]
                if gestures:
                    hud.append("Gestures: " + ", ".join(f"{g.name}({g.hand})" for g in gestures))
                annotated = draw_hud(annotated, hud)

            cv2.imshow("Arrowstack AI Task 1 - Gesture & Face Detection", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(cfg.quit_key):
                logger.info("Quit key pressed; exiting.")
                break
            elif key == ord(cfg.snapshot_key):
                path = save_snapshot(annotated, cfg.snapshot_dir)
                logger.info("Snapshot saved: %s", path)
            elif key == ord("h"):
                show_hud = not show_hud
            elif key == ord("m"):
                mirror = not mirror
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ----------------------------------------------------------------------
# Static image / batch processing (for headless testing)
# ----------------------------------------------------------------------
def run_image(pipeline: VisionPipeline, image_path: str, output_path: Optional[str], logger: logging.Logger) -> dict:
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    frame = cv2.imread(image_path)
    if frame is None:
        raise IOError(f"Could not read image: {image_path}")

    annotated, faces, gestures = pipeline.process(frame)
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, annotated)
        logger.info("Annotated image written to %s", output_path)

    summary = {
        "image": image_path,
        "faces": [d.to_dict() for d in faces],
        "gestures": [g.to_dict() for g in gestures],
    }
    print(json.dumps(summary, indent=2))
    return summary


def run_batch(pipeline: VisionPipeline, input_dir: str, output_dir: str, logger: logging.Logger) -> List[dict]:
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(input_dir)
    os.makedirs(output_dir, exist_ok=True)

    results: List[dict] = []
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith(valid_exts):
            continue
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, os.path.splitext(fname)[0] + "_annotated.jpg")
        summary = run_image(pipeline, in_path, out_path, logger)
        results.append(summary)
    return results


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def build_pipeline(cfg, logger):  # type: ignore[name-defined]
    face = FaceDetector(
        backend=cfg.face_backend,
        model_dir=cfg.face_model_dir,
        scale_factor=cfg.face_scale_factor,
        min_neighbors=cfg.face_min_neighbors,
        min_size=cfg.face_min_size,
    )
    gesture: Optional[GestureRecognizer] = None
    if is_mediapipe_available():
        gesture = GestureRecognizer(
            max_num_hands=cfg.max_hands,
            min_detection_confidence=cfg.hand_min_detection_confidence,
        )
        logger.info("MediaPipe available; gesture recognition enabled.")
    else:
        logger.warning(
            "MediaPipe is not installed; gesture recognition disabled. "
            "Install with: pip install mediapipe"
        )
    return VisionPipeline(face, gesture, logger)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Arrowstack AI Task 1 - Gesture & Face Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image", help="Process a single image file instead of the webcam.")
    p.add_argument("--batch", help="Process every image in this folder.")
    p.add_argument("--output", help="Where to write annotated images (image/batch mode).")
    p.add_argument("--camera", type=int, default=CONFIG.camera_index, help="Webcam index.")
    p.add_argument("--no-gesture", action="store_true", help="Disable MediaPipe gesture recognition.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.camera != CONFIG.camera_index:
        CONFIG.camera_index = args.camera

    logger = setup_logging(CONFIG.log_file)
    logger.info("Starting Arrowstack AI Task 1 application.")
    logger.info("Project root: %s", PROJECT_ROOT)

    pipeline = build_pipeline(CONFIG, logger)
    try:
        if args.batch:
            out = args.output or os.path.join(PROJECT_ROOT, "annotated_batch")
            results = run_batch(pipeline, args.batch, out, logger)
            logger.info("Batch done. %d images processed.", len(results))
        elif args.image:
            out = args.output or os.path.join(PROJECT_ROOT, "annotated.jpg")
            run_image(pipeline, args.image, out, logger)
        else:
            run_webcam(pipeline, CONFIG, logger)
    except Exception as exc:  # pragma: no cover - top-level safety net
        logger.exception("Application crashed: %s", exc)
        return 1
    finally:
        pipeline.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
