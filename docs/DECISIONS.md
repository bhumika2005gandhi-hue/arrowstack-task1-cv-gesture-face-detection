# Decision Log

Dated, append-only log of the most consequential design and scope
decisions made during Task 1. Newest entries at the bottom.

---

## 2026-11-08 - D1: Backend choice for face detection

**Decision:** Use OpenCV Haar cascades as the default face detector, with
an optional DNN/SSD backend.

**Rationale:** Haar cascades ship with OpenCV, need no extra model files,
and run in real time on CPU. The DNN backend is more accurate but
requires a 10 MB Caffe model that cannot be redistributed in this repo
without licensing review. Supporting both means the mentor can run the
project out of the box and optionally enable the better detector.

**Trade-off:** Haar cascades produce occasional false positives on
high-contrast textured backgrounds. Mitigated by histogram equalisation
and NMS.

---

## 2026-11-09 - D2: MediaPipe Hands for gesture recognition

**Decision:** Use MediaPipe Hands for landmark detection; classify
gestures with a hand-written rule table over a 5-bit finger-extension
pattern.

**Rationale:** MediaPipe is the de-facto open-source hand tracker, runs
on CPU, and ships with a permissive Apache-2.0 licence. A rule table is
fully explainable (no black-box model), easy to extend with new gestures,
and trivially unit-testable.

**Trade-off:** Rule-based classification cannot recognise dynamic gestures
(waving, swipe). Documented as a known limitation in `EVIDENCE.md`.

---

## 2026-11-10 - D3: Three execution modes

**Decision:** Support live webcam, single image, and batch folder modes
behind one CLI.

**Rationale:** A webcam demo is impressive but useless for headless CI or
a mentor who reviews the project on a server. The image and batch modes
make the pipeline reproducible without specialised hardware.

**Trade-off:** More CLI surface area to test, mitigated by sharing one
`VisionPipeline.process()` implementation.

---

## 2026-11-11 - D4: Non-maximum suppression

**Decision:** Add lightweight IoU-based NMS to the Haar backend.

**Rationale:** Observed duplicate detections on profile-angled faces
during local testing - one frontal and one profile box overlapped by 60%.
NMS with a 0.3 IoU threshold removed the duplicates cleanly.

**Trade-off:** Adds ~1 ms per frame. Negligible.

---

## 2026-11-12 - D5: Finger-extension heuristic

**Decision:** Compare wrist-to-tip distance against wrist-to-PIP distance
with a 1.15 multiplier, instead of comparing Y coordinates directly.

**Rationale:** Direct Y comparisons fail when the hand is rotated 45+.
The distance-based approach is rotation-invariant as long as the wrist is
visible.

**Trade-off:** Slightly less precise on the thumb (which extends
laterally). Handled with a separate thumb-specific check using the
index MCP as the reference.

---

## 2026-11-13 - D6: Privacy / responsible use

**Decision:** No frame, image, or landmark leaves the machine. Snapshots
are saved only on explicit user action (the `s` key). No face identity
recognition.

**Rationale:** Perception demos that record silently are a serious
privacy risk. The mentor should be able to review the project without
consenting to data collection.

**Trade-off:** No telemetry to improve the model. Acceptable for an
internship demo.

---

## 2026-11-14 - D7: Test strategy

**Decision:** Synthetic test images + crafted landmark vectors for unit
tests. A small number of real sample images for integration smoke tests.

**Rationale:** Real face datasets (LFW, WIDER FACE) are large and have
licensing constraints inappropriate for an internship repo. Synthetic
images give deterministic, fast, offline tests.

**Trade-off:** Tests do not measure real-world accuracy. That is
documented in `EVIDENCE.md` as a limitation.

---

## 2026-11-15 - D8: Documentation language

**Decision:** All code comments, README, docs, and tests in English.

**Rationale:** GitHub audience, mentor review, and future employer
interview use all expect English. Bilingual comments add noise without
value here.
