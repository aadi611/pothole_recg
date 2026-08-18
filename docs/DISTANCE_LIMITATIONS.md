# Distance Limitations

**Required reading before trusting any distance this system reports (FR-PRX-11).**

Distance here is **inferred, never measured**. There is no rangefinder, no
radar, no stereo pair. A single camera sees a flat picture; turning that into
a distance requires assumptions, and those assumptions are wrong some of the
time.

This document states plainly where and how. The API and UI carry a
`confidence` and a `method` on every estimate for the same reason.

---

## What the system actually does

The planar estimator (`distance/planar.py`) takes the **bottom edge of the
bounding box** — where the defect meets the road — and projects it onto an
assumed flat road plane:

```
distance = camera_height / tan(depression_angle_to_that_pixel_row)
```

That is the whole method. Its accuracy is entirely determined by how true
three things are: the camera height, the camera pitch, and the flatness of the
road.

---

## Expected accuracy

| Range | Expected error, good calibration | Usable for |
|---|---|---|
| 3–5 m | ±10–15% | Capture, alerting |
| 5–15 m | **±15–25%** | Alerting |
| 15–30 m | ±30–50% | Detection only |
| > 30 m | Worse than ±50% | Not usable for alerting |

The ±15–25% figure at 5–15 m is the one the spec commits to, and it assumes
**measured** mount height and pitch. Estimated values degrade it substantially.

---

## Where it fails

### 1. The road is not flat

The single biggest source of error.

- **Uphill ahead** — the road rises into the camera ray, so the true distance
  is **shorter** than computed. **This is the dangerous direction: the system
  warns late.**
- **Downhill ahead** — the estimate is short, so the warning comes early.
  Annoying, not dangerous.
- **Crown and camber** — Indian roads are often strongly cambered for
  drainage. A pothole at the road edge sits below the plane fitted through the
  centre.

### 2. The camera moves

Calibration assumes fixed pose. In a moving vehicle it is not.

- Braking pitches the nose down; acceleration lifts it.
- Speed bumps, potholes, and load changes all rotate the camera.
- **A 1° pitch error produces roughly a 2–3 m distance error at 15 m.**

Nothing currently compensates for this. An IMU-based pitch correction would
be the fix, and is not implemented.

### 3. The bounding box is not the pothole

The estimate is only as good as the box.

- Water-filled or shadowed potholes are detected with the wrong extent, or
  missed entirely.
- A box clipped by the frame edge has no true bottom edge, so its
  ground-contact point is fictional.
- A box drawn slightly low reads as **closer**; slightly high reads as
  **further**.

### 4. Pixel quantisation near the horizon

Close to the horizon, one pixel row spans several metres of ground. A
one-pixel box error becomes a several-metre distance error. This is why
`confidence` falls with distance — it is computed from exactly this
sensitivity, not chosen by feel.

### 5. The fallback estimator is much weaker

When calibration is unavailable, `bbox_scale` infers distance from apparent
box width against an assumed 0.6 m typical pothole width. Real potholes range
from 0.2 m to over 2 m, so a large distant pothole and a small near one can
produce identical boxes.

**It never reports confidence above 0.35**, which is below the default
`min_distance_confidence` of 0.4. That is structural: the state machine will
warn on it but will never auto-capture, so a guessed distance cannot reach a
civic complaint (FR-PRX-06).

---

## What the system does about it

| Mitigation | Where |
|---|---|
| Every estimate carries `confidence` and `assumptions` | `core/contracts.py::DistanceEstimate` |
| Confidence derived from measured per-pixel sensitivity | `distance/planar.py::_confidence` |
| Above-horizon detections return infinity, not a plausible guess | `distance/planar.py` |
| Smoothing over N frames stops threshold flapping | `distance/smoothing.py` |
| Low confidence warns but never captures | `state/machine.py` |
| UI shows approximate values ("~11 m"), never false precision | FR-UI-02 |

---

## What this means for the driver

The warning is a **prompt to look**, not a precise measurement. It says
roughly "there is probably a pothole roughly this far ahead." On a flat,
well-lit, dry road with a properly calibrated camera it will usually be close.
On a hill, in rain, at night, or after the camera has been knocked, it may be
substantially wrong.

**This system supplements the driver's attention. It does not replace it.**

---

## What would improve it

Not implemented, in rough order of value per effort:

1. **Real intrinsic calibration** (chessboard) instead of a datasheet FOV —
   removes lens distortion error. This is FR-PRX-08.
2. **IMU pitch compensation** — corrects the braking/acceleration case, which
   is the most frequent dynamic error.
3. **Monocular depth cross-check** (FR-PRX-09) — an independent estimate to
   disagree with the planar one, flagging low agreement.
4. **Stereo or depth camera** (FR-PRX-10) — removes the flat-road assumption
   entirely, at hardware cost.
