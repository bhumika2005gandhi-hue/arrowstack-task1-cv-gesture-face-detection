"""
tests/test_face_detection.py
============================
Unit tests for the FaceDetector class.

Tests use synthetic images (no real face datasets) so they are fully
deterministic and run in any CI environment.
"""

from __future__ import annotations

import os
import sys

# Make sure we can import the project root modules when run as `pytest`
# from inside the project folder.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
import cv2

from face_detection import Detection, FaceDetector


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def detector():
    return FaceDetector(backend="haar")


@pytest.fixture
def blank_image():
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def face_like_image():
    """A 480x640 image with a 'skin-coloured' oval on a dark background.
    Haar cascades are trained for real faces, so this may not trigger a
    detection - we use it to verify the detector returns *some* result
    without crashing on a synthetic pattern."""
    img = np.full((480, 640, 3), 30, dtype=np.uint8)
    # Draw a skin-coloured oval near the centre.
    cv2.ellipse(img, (320, 240), (80, 110), 0, 0, 360, (180, 140, 110), -1)
    # Two darker "eyes" and a "mouth" to nudge the cascade.
    cv2.circle(img, (290, 220), 8, (40, 30, 30), -1)
    cv2.circle(img, (350, 220), 8, (40, 30, 30), -1)
    cv2.ellipse(img, (320, 280), (30, 10), 0, 0, 180, (60, 40, 40), -1)
    return img


# ----------------------------------------------------------------------
# Behavioural tests
# ----------------------------------------------------------------------
def test_detect_returns_list_on_empty_image(detector, blank_image):
    result = detector.detect(blank_image)
    assert isinstance(result, list)


def test_detect_handles_none_gracefully(detector):
    assert detector.detect(None) == []


def test_detect_handles_small_image(detector):
    small = np.zeros((20, 20, 3), dtype=np.uint8)
    result = detector.detect(small)
    assert isinstance(result, list)


def test_detect_runs_on_face_like_image(detector, face_like_image):
    # The synthetic oval may or may not trigger a real detection, but the
    # call must not crash and must return a list.
    result = detector.detect(face_like_image)
    assert isinstance(result, list)
    for det in result:
        assert isinstance(det, Detection)
        assert det.w > 0 and det.h > 0
        assert 0.0 <= det.confidence <= 1.0
        assert det.backend in {"haar", "dnn"}
        assert det.label in {"face_frontal", "face_profile"}


def test_detection_dataclass_properties():
    det = Detection(x=10, y=20, w=100, h=80, confidence=0.9,
                    label="face_frontal", backend="haar")
    assert det.area == 8000
    assert det.center == (60, 60)
    d = det.to_dict()
    assert d["x"] == 10 and d["confidence"] == 0.9


def test_nms_removes_overlaps():
    dets = [
        Detection(0, 0, 100, 100, 0.9, "face_frontal", "haar"),
        Detection(5, 5, 100, 100, 0.7, "face_frontal", "haar"),  # ~88% IoU
        Detection(400, 400, 80, 80, 0.8, "face_frontal", "haar"),  # no overlap
    ]
    kept = FaceDetector._nms(dets, iou_threshold=0.3)
    assert len(kept) == 2
    # Higher-confidence box wins.
    assert kept[0].confidence == 0.9


def test_iou_zero_for_disjoint():
    a = Detection(0, 0, 50, 50, 1.0, "x", "haar")
    b = Detection(200, 200, 50, 50, 1.0, "x", "haar")
    assert FaceDetector._iou(a, b) == 0.0


def test_iou_one_for_identical():
    a = Detection(0, 0, 50, 50, 1.0, "x", "haar")
    b = Detection(0, 0, 50, 50, 1.0, "x", "haar")
    assert FaceDetector._iou(a, b) == pytest.approx(1.0)


def test_dnn_backend_falls_back_when_model_missing(tmp_path):
    # Empty model dir -> DNN backend cannot load -> should fall back to Haar.
    det = FaceDetector(backend="dnn", model_dir=str(tmp_path))
    assert det.backend == "haar"


def test_haar_files_exist():
    """Sanity check: the cascade files must be present in the OpenCV install."""
    import os
    assert os.path.exists(FaceDetector.HAAR_FRONTAL), \
        "haarcascade_frontalface_default.xml missing - OpenCV install incomplete"
