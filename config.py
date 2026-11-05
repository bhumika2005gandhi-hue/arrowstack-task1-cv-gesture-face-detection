"""
config.py
=========
Centralised configuration for the gesture & face detection application.
Every value lives here so reviewers and mentors can tweak behaviour
without editing the application logic.

Author: Bhoomika Gandhi (Arrowstack Batch 2026, AI domain)
Ref ID: AS-2026-26671
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


@dataclass
class AppConfig:
    """Runtime configuration for the webcam loop and detectors."""

    # Camera
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 25

    # Detectors
    face_backend: str = "haar"  # "haar" or "dnn"
    face_model_dir: str = os.path.join(PROJECT_ROOT, "models")
    face_min_size: tuple = (48, 48)
    face_scale_factor: float = 1.1
    face_min_neighbors: int = 5

    max_hands: int = 2
    hand_min_detection_confidence: float = 0.6

    # Output
    snapshot_dir: str = os.path.join(PROJECT_ROOT, "snapshots")
    save_snapshots_on_gesture: bool = True
    log_file: str = os.path.join(PROJECT_ROOT, "logs", "session.log")

    # Display
    show_hud: bool = True
    mirror_camera: bool = True

    # Quit keys
    quit_key: str = "q"
    snapshot_key: str = "s"


# Module-level singleton used by main.py and tests.
CONFIG = AppConfig()
