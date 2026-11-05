"""
tests/test_gesture_recognition.py
=================================
Unit tests for the gesture recognition module.

We craft synthetic HandLandmarks objects with specific finger patterns
so the tests are deterministic and do not require MediaPipe to actually
run inference (the classify() logic is pure given the landmarks).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from gesture_recognition import (
    GestureRecognizer,
    GESTURE_TEMPLATES,
    HandLandmarks,
    is_mediapipe_available,
)


# ----------------------------------------------------------------------
# Helpers: build synthetic landmark sets that exhibit known patterns
# ----------------------------------------------------------------------
# MediaPipe hand model: 21 landmarks. We only need a representative subset
# to exercise the finger-extension logic. The exact coordinates below
# place fingertips far from the wrist and PIP joints close to the wrist,
# so the finger is "extended". For "folded" fingers we invert that.
#
# Indices we care about:
#   0  = wrist
#   3  = thumb IP,    4 = thumb tip
#   5  = index MCP,   6 = index PIP,   8 = index tip
#  10 = middle PIP,  12 = middle tip
#  14 = ring PIP,    16 = ring tip
#  18 = pinky PIP,   20 = pinky tip
W = (0.5, 0.5, 0.0)  # wrist at centre


def _lm(finger_states):
    """Build a 21-landmark list with the given finger extension states.

    ``finger_states`` is a 5-char string of 0/1 in the order
    thumb index middle ring pinky.
    """
    # Default: everything folded (PIP near tip, both close to wrist).
    base = [W] * 21
    # MCPs (5, 9, 13, 17) sit slightly above wrist
    base[5]  = (0.45, 0.45, 0.0)
    base[9]  = (0.50, 0.45, 0.0)
    base[13] = (0.55, 0.45, 0.0)
    base[17] = (0.60, 0.45, 0.0)

    # For each finger, decide tip vs PIP positions
    finger_map = {
        # Folded thumb tip is placed very close to the index MCP (5),
        # which is at (0.45, 0.45). The thumb-extension heuristic compares
        # wrist-to-tip and wrist-to-PIP distances; placing the folded tip
        # near the index MCP keeps it close to the palm so the heuristic
        # correctly reports "not extended". The folded tip must also be
        # more than 0.06 away from the folded index tip (0.50, 0.38) so
        # the OK-sign heuristic does not fire.
        "thumb":  {"tip": 4,  "pip": 3,  "ext": (0.20, 0.45, 0.0), "fld": (0.40, 0.42, 0.0)},
        "index":  {"tip": 8,  "pip": 6,  "ext": (0.45, 0.15, 0.0), "fld": (0.50, 0.38, 0.0)},
        "middle": {"tip": 12, "pip": 10, "ext": (0.50, 0.12, 0.0), "fld": (0.55, 0.38, 0.0)},
        "ring":   {"tip": 16, "pip": 14, "ext": (0.55, 0.15, 0.0), "fld": (0.60, 0.38, 0.0)},
        "pinky":  {"tip": 20, "pip": 18, "ext": (0.62, 0.20, 0.0), "fld": (0.65, 0.40, 0.0)},
    }

    for i, finger in enumerate(("thumb", "index", "middle", "ring", "pinky")):
        m = finger_map[finger]
        if finger_states[i] == "1":
            base[m["tip"]] = m["ext"]
            # For an extended finger, place the PIP joint closer to the
            # wrist than the tip is, so the extension heuristic fires.
            if finger == "thumb":
                # Extended thumb IP sits between tip (0.20, 0.45) and
                # index MCP (0.45, 0.45) so dist(tip, mcp) > dist(ip, mcp).
                base[m["pip"]] = (0.30, 0.45, 0.0)
            else:
                base[m["pip"]] = (m["ext"][0], 0.30, 0.0)
        else:
            base[m["tip"]] = m["fld"]
            # For a folded finger, place the PIP joint FARTHER from the
            # wrist than the tip is, so the heuristic reports "folded".
            if finger == "thumb":
                # Folded thumb IP sits away from index MCP so dist(tip, mcp)
                # < dist(ip, mcp).
                base[m["pip"]] = (0.32, 0.50, 0.0)
            else:
                base[m["pip"]] = (m["fld"][0], 0.30, 0.0)

    return base


# ----------------------------------------------------------------------
# Tests on the pure classification logic
# ----------------------------------------------------------------------
def test_gesture_templates_cover_known_gestures():
    names = {n for _, n in GESTURE_TEMPLATES}
    expected = {"OPEN_PALM", "FIST", "THUMBS_UP", "PEACE", "POINT", "ROCK", "FOUR"}
    assert expected.issubset(names)


def test_finger_extended_for_open_palm():
    hand = HandLandmarks(
        handedness="Right",
        landmarks=_lm("11111"),
        wrist=W[:2],
    )
    pattern = hand.finger_pattern()
    assert pattern == "11111"


def test_finger_folded_for_fist():
    hand = HandLandmarks(
        handedness="Right",
        landmarks=_lm("00000"),
        wrist=W[:2],
    )
    assert hand.finger_pattern() == "00000"


def test_peace_pattern():
    hand = HandLandmarks(
        handedness="Right",
        landmarks=_lm("01100"),
        wrist=W[:2],
    )
    assert hand.finger_pattern() == "01100"


@pytest.mark.parametrize("pattern,expected_name", [
    ("11111", "OPEN_PALM"),
    ("00000", "FIST"),
    ("01100", "PEACE"),
    ("01001", "ROCK"),
    ("01111", "FOUR"),
])
def test_classify_returns_expected_gesture(pattern, expected_name):
    hand = HandLandmarks(
        handedness="Right",
        landmarks=_lm(pattern),
        wrist=W[:2],
    )
    name, conf = GestureRecognizer._classify(hand.finger_pattern(), hand)
    assert name == expected_name
    assert 0.0 <= conf <= 1.0


def test_classify_unknown_pattern():
    hand = HandLandmarks(
        handedness="Right",
        landmarks=_lm("00110"),  # middle + ring only -> not in templates
        wrist=W[:2],
    )
    name, conf = GestureRecognizer._classify(hand.finger_pattern(), hand)
    assert name == "UNKNOWN"
    assert conf < 0.6


def test_ok_sign_when_thumb_index_close():
    """When thumb tip and index tip are within 0.06 of each other, we
    classify as OK_SIGN regardless of the pattern."""
    lms = _lm("11111")
    # Move thumb tip and index tip close together.
    lms[4] = (0.45, 0.20, 0.0)
    lms[8] = (0.46, 0.21, 0.0)
    hand = HandLandmarks(
        handedness="Right",
        landmarks=lms,
        wrist=W[:2],
    )
    name, conf = GestureRecognizer._classify(hand.finger_pattern(), hand)
    assert name == "OK_SIGN"


def test_invalid_finger_raises():
    hand = HandLandmarks(handedness="Right", landmarks=[W] * 21, wrist=W[:2])
    with pytest.raises(ValueError):
        hand.finger_extended("ring_finger")


def test_is_mediapipe_available_returns_bool():
    assert isinstance(is_mediapipe_available(), bool)


# ----------------------------------------------------------------------
# Integration test (only runs if MediaPipe is installed)
# ----------------------------------------------------------------------
@pytest.mark.skipif(not is_mediapipe_available(),
                    reason="MediaPipe not installed")
def test_recognizer_initialises_and_closes():
    rec = GestureRecognizer()
    rec.close()
