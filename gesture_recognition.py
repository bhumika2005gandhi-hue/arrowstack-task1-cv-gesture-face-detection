"""
gesture_recognition.py
======================
Hand gesture recognition module using Google's MediaPipe Hands solution.

Recognised gestures
-------------------
* ``OPEN_PALM``  - all five fingers extended
* ``FIST``       - all five fingers folded
* ``THUMBS_UP``  - only the thumb extended, pointing up
* ``PEACE``      - index + middle extended (Victory / V sign)
* ``POINT``      - only the index finger extended
* ``OK_SIGN``    - thumb tip touches index tip (heuristic)
* ``UNKNOWN``    - anything else

Design notes
------------
* The module is intentionally decoupled from MediaPipe specifics. If
  MediaPipe is unavailable, :class:`GestureRecognizer` raises a clear
  ``ImportError`` only when first instantiated, so the import of this
  module never crashes an environment that lacks the dependency.
* Finger-extension logic is rule-based: a finger is considered
  "extended" when its tip is farther from the wrist than its PIP joint
  (along the appropriate axis). This is more robust than comparing Y
  coordinates directly, which breaks when the hand is rotated.
* Every gesture includes a confidence score derived from how cleanly
  the finger pattern matches a known template.

Author: Bhoomika Gandhi (Arrowstack Batch 2026, AI domain)
Ref ID: AS-2026-26671
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in CI without mediapipe
    mp = None
    _MP_AVAILABLE = False


@dataclass
class HandLandmarks:
    """Container for one hand's landmarks returned by MediaPipe."""

    handedness: str  # "Left" or "Right"
    landmarks: List[Tuple[float, float, float]]  # 21 (x, y, z) tuples, normalised 0..1
    wrist: Tuple[float, float] = field(default=(0.0, 0.0))

    def finger_extended(self, finger: str) -> bool:
        """Return True if ``finger`` appears extended.

        ``finger`` must be one of: ``thumb``, ``index``, ``middle``,
        ``ring``, ``pinky``. We compare the distance from the wrist to
        the fingertip against the distance from the wrist to the PIP
        joint; the finger is "extended" when the tip is significantly
        farther away.
        """
        tips = {
            "thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20,
        }
        pips = {
            "thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18,
        }
        if finger not in tips:
            raise ValueError(f"Unknown finger: {finger}")
        tip = np.array(self.landmarks[tips[finger]][:2])
        pip = np.array(self.landmarks[pips[finger]][:2])
        wrist = np.array(self.wrist)

        def dist(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.linalg.norm(a - b))

        # For the thumb we use the x-axis difference relative to the
        # index MCP because thumb extension is mostly lateral.
        if finger == "thumb":
            index_mcp = np.array(self.landmarks[5][:2])
            thumb_tip = np.array(self.landmarks[4][:2])
            thumb_ip = np.array(self.landmarks[3][:2])
            # If the thumb tip is farther from the index MCP than the
            # thumb IP joint, the thumb is extended outward.
            return dist(thumb_tip, index_mcp) > dist(thumb_ip, index_mcp) * 1.1

        return dist(tip, wrist) > dist(pip, wrist) * 1.15

    def finger_pattern(self) -> str:
        """Return a 5-character pattern, e.g. ``'10100'`` for index+middle.

        Order: thumb index middle ring pinky. ``1`` = extended.
        """
        return "".join(
            "1" if self.finger_extended(f) else "0"
            for f in ("thumb", "index", "middle", "ring", "pinky")
        )


@dataclass
class Gesture:
    """A recognised gesture for one hand."""

    name: str
    confidence: float
    hand: str
    pattern: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h in pixel coords

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "confidence": float(self.confidence),
            "hand": self.hand,
            "pattern": self.pattern,
            "bbox": list(self.bbox),
        }


# Templates: pattern -> gesture name. The order matters because some
# patterns are subsets of others (e.g. POINT is a subset of PEACE if we
# are not careful), so more specific patterns come first.
GESTURE_TEMPLATES: List[Tuple[str, str]] = [
    ("11111", "OPEN_PALM"),
    ("00000", "FIST"),
    ("01001", "ROCK"),         # index + pinky (bonus, often seen as "rock on")
    ("01100", "PEACE"),        # index + middle (Victory / V)
    ("00100", "POINT"),        # index only
    ("10000", "THUMBS_UP"),    # thumb only
    ("11000", "THUMBS_UP"),    # thumb + index (close enough; mainly lateral)
    ("01111", "FOUR"),         # index..pinky extended (bonus)
]


class GestureRecognizer:
    """Detect hands and classify their static gesture from a single frame.

    Parameters
    ----------
    max_num_hands:
        Maximum number of hands to detect per frame. Default 2.
    min_detection_confidence:
        Minimum MediaPipe confidence to accept a hand detection.
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.6,
    ) -> None:
        if not _MP_AVAILABLE:
            raise ImportError(
                "MediaPipe is required for gesture recognition. "
                "Install it with: pip install mediapipe"
            )
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self.hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray) -> List[Gesture]:
        """Detect hands in a BGR image and classify each gesture."""
        if image is None or image.size == 0:
            return []
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        if not result.multi_hand_landmarks:
            return []

        gestures: List[Gesture] = []
        for idx, hand_lms in enumerate(result.multi_hand_landmarks):
            handedness = (
                result.multi_handedness[idx].label
                if result.multi_handedness
                else "Unknown"
            )
            lm_tuples = [(lm.x, lm.y, lm.z) for lm in hand_lms.landmark]
            wrist = lm_tuples[0][:2]
            hand = HandLandmarks(
                handedness=handedness,
                landmarks=lm_tuples,
                wrist=wrist,
            )
            pattern = hand.finger_pattern()
            name, confidence = self._classify(pattern, hand)
            bbox = self._bbox_from_landmarks(lm_tuples, w, h)
            gestures.append(
                Gesture(
                    name=name,
                    confidence=confidence,
                    hand=handedness,
                    pattern=pattern,
                    bbox=bbox,
                )
            )
        return gestures

    def draw(self, image: np.ndarray, gestures: List[Gesture]) -> np.ndarray:
        """Draw landmarks on a copy of ``image`` and return it."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        annotated = image.copy()
        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                self._mp_drawing.draw_landmarks(
                    annotated,
                    hand_lms,
                    self._mp_hands.HAND_CONNECTIONS,
                )
        return annotated

    def close(self) -> None:
        self.hands.close()

    def __enter__(self) -> "GestureRecognizer":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _classify(pattern: str, hand: HandLandmarks) -> Tuple[str, float]:
        # Try OK sign first because it depends on thumb-index contact,
        # not just the extension pattern.
        thumb_tip = np.array(hand.landmarks[4][:2])
        index_tip = np.array(hand.landmarks[8][:2])
        if float(np.linalg.norm(thumb_tip - index_tip)) < 0.06:
            return "OK_SIGN", 0.85

        for tmpl, name in GESTURE_TEMPLATES:
            if pattern == tmpl:
                return name, 0.9
        return "UNKNOWN", 0.4

    @staticmethod
    def _bbox_from_landmarks(
        landmarks: List[Tuple[float, float, float]],
        width: int,
        height: int,
    ) -> Tuple[int, int, int, int]:
        xs = [int(lm[0] * width) for lm in landmarks]
        ys = [int(lm[1] * height) for lm in landmarks]
        x, y = min(xs), min(ys)
        return (x, y, max(xs) - x, max(ys) - y)


def is_mediapipe_available() -> bool:
    """Return True if MediaPipe can be imported."""
    return _MP_AVAILABLE
