"""
tests/test_utils.py
===================
Unit tests for the rendering / snapshot helpers in utils.py.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from face_detection import Detection
from gesture_recognition import Gesture
from utils import (
    FpsCounter,
    draw_faces,
    draw_gestures,
    draw_hud,
    save_snapshot,
)


@pytest.fixture
def blank_frame():
    return np.full((480, 640, 3), 50, dtype=np.uint8)


def test_draw_faces_returns_new_image(blank_frame):
    dets = [Detection(10, 10, 100, 120, 0.9, "face_frontal", "haar")]
    annotated = draw_faces(blank_frame, dets)
    assert annotated.shape == blank_frame.shape
    # The original frame must not be modified.
    assert np.array_equal(blank_frame, np.full((480, 640, 3), 50, dtype=np.uint8))


def test_draw_faces_empty_list_returns_copy(blank_frame):
    annotated = draw_faces(blank_frame, [])
    assert annotated.shape == blank_frame.shape


def test_draw_gestures_runs():
    frame = np.full((480, 640, 3), 50, dtype=np.uint8)
    gestures = [Gesture(name="OPEN_PALM", confidence=0.9, hand="Right",
                        pattern="11111", bbox=(100, 100, 80, 80))]
    annotated = draw_gestures(frame, gestures)
    assert annotated.shape == frame.shape


def test_draw_hud_renders_lines(blank_frame):
    annotated = draw_hud(blank_frame, ["line one", "line two"])
    # Panel covers rows 8..68 and cols 8..368. Pick a pixel inside the
    # panel but away from the anti-aliased text glyphs (text is at x=16
    # only, so col=300 is safe). Numpy indexing is [row, col] = [y, x].
    assert annotated[60, 300, 0] < 60
    # And the panel must have actually been drawn (i.e. changed from the
    # original background value of 50).
    assert annotated[60, 300, 0] != 50


def test_save_snapshot_creates_file(tmp_path, blank_frame):
    path = save_snapshot(blank_frame, str(tmp_path), prefix="test")
    assert os.path.exists(path)
    assert path.endswith(".png")
    assert "test" in os.path.basename(path)


def test_fps_counter():
    fps = FpsCounter(window=5)
    assert fps.fps() == 0.0
    for _ in range(3):
        fps.tick()
    # After ticks, fps should be a non-negative finite number.
    val = fps.fps()
    assert val >= 0.0
