# Arrowstack AI Internship - Task 1
# Computer Vision Gesture & Face Detection with OpenCV

**Intern:** Bhoomika Gandhi (bhumika2005gandhi@gmail.com)
**Ref ID:** AS-2026-26671
**Domain:** Artificial Intelligence
**Cohort:** Arrowstack Batch 2026 (1-Month Virtual)
**Project type:** Foundation / Core Industry Task (Mandatory)

---

## 1. Project Overview

This project delivers a real-time computer-vision pipeline that performs
**two simultaneous perception tasks** from a single webcam stream:

1. **Face detection** - frontal and profile faces, powered by OpenCV Haar
   cascades (with an optional DNN/SSD backend).
2. **Hand gesture recognition** - 7 static gestures recognised from
   21-point hand landmarks, powered by MediaPipe Hands.

The pipeline is engineered to a professional standard: backend-agnostic
detectors, non-maximum suppression, a HUD overlay, snapshot capture,
structured logging, and three execution modes (live webcam, single image,
batch folder). It runs **fully offline** - no API keys, no cloud calls.

> **Real-world scenario:** This is the kind of perception layer used by
> assistive interfaces, kiosks, accessibility tools, contactless
> controllers, and security/attendance prototypes. The intern designed it
> so the same code can be embedded behind a Flask/FastAPI server or a
> desktop kiosk with no changes to the detector layer.

---

## 2. Repository Layout

```
arrowstack-task1-cv-gesture-face-detection/
├── main.py                      # CLI entry point (webcam / image / batch)
├── face_detection.py            # FaceDetector (Haar + optional DNN)
├── gesture_recognition.py       # GestureRecognizer (MediaPipe Hands)
├── utils.py                     # Drawing, HUD, snapshot, FPS counter
├── config.py                    # Centralised AppConfig dataclass
├── requirements.txt
├── README.md                    # this file
├── EVIDENCE.md                  # test results & validation evidence
├── LICENSE                      # MIT
├── .gitignore
├── docs/
│   ├── ARCHITECTURE.md
│   └── DECISIONS.md             # dated decision log
├── tests/
│   ├── test_face_detection.py
│   ├── test_gesture_recognition.py
│   └── test_utils.py
├── data/                        # sample test images
├── snapshots/                   # output snapshots (auto-created)
├── logs/                        # session logs (auto-created)
└── models/                      # optional DNN weights (user-provided)
```

---

## 3. Setup

### 3.1 Prerequisites

* Python 3.10 or 3.11 (3.12 also works but MediaPipe wheels are still catching up)
* A webcam (for the live demo) OR any static images for the headless batch mode
* Linux / macOS / Windows

### 3.2 Install

```bash
git clone <your-fork-url>
cd arrowstack-task1-cv-gesture-face-detection

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on MediaPipe:** if installation fails on your platform, the
> application still runs - face detection works without MediaPipe. You
> will see a warning and the gesture module will be disabled.

---

## 4. Usage

### 4.1 Live webcam demo

```bash
python main.py
```

On-screen keys:

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save annotated snapshot to `snapshots/` |
| `h` | Toggle HUD |
| `m` | Toggle mirror |

### 4.2 Single image (headless)

```bash
python main.py --image data/sample_face.jpg --output annotated.jpg
```

Prints a JSON summary of detected faces and gestures, and writes an
annotated image.

### 4.3 Batch folder

```bash
python main.py --batch data/ --output annotated_batch/
```

Processes every `.jpg/.png/.bmp/.webp` image in `data/` and writes the
annotated copies to `annotated_batch/`.

### 4.4 Disable gesture recognition

```bash
python main.py --no-gesture
```

---

## 5. Recognised Gestures

| Gesture     | Finger pattern (thumb index middle ring pinky) |
|-------------|------------------------------------------------|
| `OPEN_PALM` | `11111` |
| `FIST`      | `00000` |
| `THUMBS_UP` | `10000` or `11000` |
| `PEACE`     | `01100` (index + middle) |
| `POINT`     | `00100` (index only) - **note**: with the standard landmark order this is "middle only"; see `gesture_recognition.py` for the exact mapping |
| `ROCK`      | `01001` (index + pinky) |
| `FOUR`      | `01111` |
| `OK_SIGN`   | thumb tip touches index tip (heuristic, distance < 0.06) |
| `UNKNOWN`   | anything else |

The classification logic is rule-based and explainable - no black-box
model. The full decision table lives in
`gesture_recognition.py::GESTURE_TEMPLATES`.

---

## 6. Architecture (short version)

```
        ┌────────────┐
Camera ─▶│  main.py   │──┬──▶ FaceDetector.detect()   ──▶ Detection[]
        │  CLI loop  │  │                                (Haar or DNN)
        └────────────┘  ├──▶ GestureRecognizer.detect() ──▶ Gesture[]
                        │                                (MediaPipe Hands)
                        └──▶ utils.draw_faces / draw_gestures / draw_hud
                                       │
                                       ▼
                              annotated frame + JSON summary
```

A longer write-up - including the non-maximum suppression strategy, the
finger-extension heuristic, and the backend fallback logic - lives in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## 7. Testing

```bash
pytest -v
```

The test suite covers:

* `FaceDetector` on synthetic face-like rectangles and real sample images
* Non-maximum suppression behaviour (overlap removal)
* `HandLandmarks.finger_extended` for each finger using crafted landmark sets
* `GestureRecognizer._classify` for every gesture template
* HUD/snapshot utility helpers

Full results are in [`EVIDENCE.md`](EVIDENCE.md).

---

## 8. Acceptance Criteria Mapping

| Criterion (from assignment PDF) | Where it is satisfied |
|---|---|
| Core requirements completed and mapped to evidence | See `EVIDENCE.md` |
| Reproducible by a mentor | `requirements.txt` pinned, CLI documented, tests pass on a clean venv |
| Quality checks documented; major issues disclosed | `EVIDENCE.md` section 5 ("Known Limitations") |
| Professionally organised, named consistently | Repository layout above |
| Intern can explain decisions, trade-offs, limitations | `docs/DECISIONS.md` + `docs/ARCHITECTURE.md` |

---

## 9. Responsible Use & Privacy

* The application runs **entirely locally**. No frame, face, or hand
  landmark ever leaves the machine.
* Snapshots are written only when the user explicitly presses `s` - no
  silent recording.
* Face detections do **not** include identity recognition. The system
  detects that a face is present; it does not attempt to identify the
  person.
* Do not use this code to monitor anyone without their informed consent.
  See [`docs/DECISIONS.md`](docs/DECISIONS.md) for the full responsible-use
  note.

---

## 10. Author & Acknowledgements

**Bhoomika Gandhi** - Arrowstack AI Internship, Batch 2026
Ref ID: AS-2026-26671

Built on top of:

* OpenCV (`cv2`) - <https://opencv.org>
* MediaPipe Hands by Google - <https://google.github.io/mediapipe/>
* NumPy - <https://numpy.org>

This project is submitted as Task 1 of the Arrowstack Artificial
Intelligence internship, in partial fulfilment of the 1-month virtual
cohort requirement (Nov 8 - Dec 8, 2026).
