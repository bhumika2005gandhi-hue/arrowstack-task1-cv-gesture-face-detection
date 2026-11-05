# Architecture

## High-level flow

```
   ┌──────────────┐
   │  Camera /    │
   │  Image file  │
   └──────┬───────┘
          │ BGR frame (HxWx3)
          ▼
   ┌──────────────────────────────────────────────┐
   │             VisionPipeline.process()         │
   │                                              │
   │   ┌─────────────────────┐  ┌──────────────┐  │
   │   │  FaceDetector       │  │ GestureRecog │  │
   │   │   .detect(frame)    │  │  .detect()   │  │
   │   └─────────┬───────────┘  └──────┬───────┘  │
   │             │                     │          │
   │             │ Detection[]         │ Gesture[]│
   │             ▼                     ▼          │
   │   utils.draw_faces()    utils.draw_gestures()│
   │             │                     │          │
   │             └──────────┬──────────┘          │
   │                        ▼                     │
   │              draw_hud() overlay              │
   └────────────────────┬─────────────────────────┘
                        ▼
                  annotated frame
                        │
                        ├──▶ cv2.imshow (webcam)
                        ├──▶ cv2.imwrite (image/batch)
                        └──▶ JSON summary (image/batch)
```

## Module responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | Single source of truth for tunable parameters. |
| `face_detection.py` | Wrap OpenCV Haar cascade + optional DNN/SSD behind a stable interface. Includes NMS. |
| `gesture_recognition.py` | Wrap MediaPipe Hands. Maps 21 landmarks to a 5-bit finger-extension pattern and matches it against a template table. |
| `utils.py` | Pure rendering helpers (boxes, labels, HUD, snapshot, FPS). No detector logic. |
| `main.py` | CLI, three execution modes, lifecycle/logging. |

## Key design decisions

### Backend-agnostic detector

`FaceDetector` exposes a single `detect(image) -> List[Detection]` method.
Internally it can use either:

1. **Haar cascade** (default, ships with OpenCV, always works).
2. **DNN/SSD** (Caffe model, optional). To enable, drop
   `res10_300x300_ssd_iter_140000.caffemodel` and `deploy.prototxt` into
   `models/` and pass `backend="dnn"`.

If the DNN model files are missing, the constructor falls back to Haar
transparently so the rest of the app keeps working.

### Non-maximum suppression

Haar cascades can fire on the same face twice (frontal + profile). The
`_nms` helper removes overlaps using Intersection-over-Union with a 0.3
threshold. This was added after observing duplicate boxes on test images
with 45-degree head poses.

### Finger-extension heuristic

Naive Y-axis comparisons (`tip.y < pip.y`) fail when the hand is rotated.
We instead compare **distances from the wrist**:

```
finger_extended = distance(tip, wrist) > distance(pip, wrist) * 1.15
```

The 1.15 multiplier absorbs MediaPipe's normalisation noise. For the
thumb (which extends laterally, not vertically) we use the distance from
the index MCP instead of the wrist.

### Gesture classification table

A pattern string `"10100"` (thumb + middle) maps to a gesture name via
`GESTURE_TEMPLATES`. The order in the list matters because some patterns
are subsets of others - more specific patterns come first. The OK sign is
detected separately because it depends on **thumb-index contact**, not
extension.

### Graceful degradation without MediaPipe

`gesture_recognition.py` is imported lazily. If MediaPipe is not
installed, `is_mediapipe_available()` returns `False` and `main.py`
skips gesture detection. The face pipeline keeps working.

## Failure modes

| Failure | Behaviour |
|---|---|
| No webcam | `cv2.VideoCapture.isOpened()` returns False; the app exits with a logged error instead of an exception. |
| Empty frame | Logged as a warning; loop continues. |
| MediaPipe import fails | Warning logged; gesture detection disabled. |
| DNN model missing | Falls back to Haar cascade; no crash. |
| Image not found | `FileNotFoundError` raised with the offending path. |

## Where this fits in a production system

The detectors are pure functions over frames. To deploy this behind a web
API you would:

1. Wrap `VisionPipeline.process()` in a FastAPI route that accepts an
   uploaded image and returns the JSON summary.
2. Run the webcam loop only on edge devices; the cloud service would
   only handle one-shot image analysis.
3. Add a rate limiter and a max-image-size guard before exposing it
   publicly.
