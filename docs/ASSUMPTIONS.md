# Assumptions

Every assumption the code makes about camera geometry, road shape, and vehicle
motion, with where it lives and what breaks when it is wrong
(`spec/01_AGENT_BRIEF.md`, working agreement).

---

## Camera

| # | Assumption | Where | If wrong |
|---|---|---|---|
| C1 | Mount height is measured ground-to-lens and constant | `config/proximity.yaml: mount_height_m` | Distance scales linearly with the error: a 20 cm error on a 1.25 m mount is ~16% on every estimate |
| C2 | Pitch is fixed and known | `config/proximity.yaml: pitch_deg` | ~1° error ≈ 2–3 m error at 15 m. Violated constantly in practice by braking and bumps |
| C3 | The camera is a pinhole with no significant distortion | `distance/planar.py::vertical_fov_deg` | Wide-angle dashcams have real barrel distortion; edge-of-frame detections read further than they are. Fixed by FR-PRX-08 calibration |
| C4 | Horizontal FOV from the datasheet is accurate | same | Quoted FOV is often diagonal, or optimistic. A wrong FOV skews everything |
| C5 | Zero roll — the horizon is level in frame | `distance/planar.py` | A tilted mount makes left and right sides of the frame project differently. Not modelled at all |
| C6 | Frame shape matches what the detector saw | `estimate(frame_shape=...)` | Passing config resolution instead of actual frame size silently skews every estimate. Guarded by a test |

## Road

| # | Assumption | Where | If wrong |
|---|---|---|---|
| R1 | The road ahead is a flat plane | `distance/planar.py` | The core weakness. **Uphill makes the system warn late** — see `DISTANCE_LIMITATIONS.md` |
| R2 | The plane starts directly below the camera | same | On a strongly cambered road the wheel track is not at the fitted plane height |
| R3 | The defect's ground-contact point is the box bottom edge | `core/contracts.py::BBox.bottom_center` | Occluded or frame-clipped boxes have no true bottom edge |
| R4 | Typical pothole width is 0.6 m | `distance/bbox_scale.py` | Only used by the fallback, which caps confidence at 0.35 precisely because this is a weak assumption |

## Vehicle & motion

| # | Assumption | Where | If wrong |
|---|---|---|---|
| V1 | The vehicle travels roughly straight between frames | `tracking/tracker.py` Kalman | Sharp cornering breaks the constant-velocity model; tracks may fragment |
| V2 | GPS speed is current enough for time-to-impact | `distance/planar.py` | A stale fix produces a wrong time-to-impact. GPS staleness is flagged separately (FR-GEO-04) |
| V3 | Apparent size grows non-linearly on approach | `tracking/tracker.py` | Why box size gets no velocity term in the filter — a linear model diverges in the last few metres |

## Detection

| # | Assumption | Where | If wrong |
|---|---|---|---|
| D1 | Training data resembles the deployment road | `docs/MODEL_CARD.md` | Currently the largest known gap: available datasets are predominantly non-Indian. See `docs/DATASETS.md` |
| D2 | One physical defect yields one track | `tracking/tracker.py` | Measured at 90% under realistic noise (`scripts/bench_tracker.py`). Fragmentation causes duplicate reports |
| D3 | 3 confirmations suppress false positives | `min_hits` | Too low admits shadow flicker; too high delays the warning |

## Time

| # | Assumption | Where | If wrong |
|---|---|---|---|
| T1 | Frame timestamps are capture time, not processing time | `camera/base.py::Frame` | Evidence GPS binding would be offset by the inference latency (FR-EVD-04) |
| T2 | Video presentation timestamps are monotonic | `camera/sources.py` | A malformed container could produce non-monotonic replay |

---

## Calibration values currently shipped

The defaults in `config/proximity.yaml` are **placeholders**, not measurements:

```yaml
mount_height_m: 1.25    # typical windscreen height for a sedan
pitch_deg: -8.0         # a plausible downward angle
horizontal_fov_deg: 78.0  # typical dashcam
```

**These must be replaced with measured values before any real drive.** They
are starting points that make the pipeline runnable, not truth. Assumptions
C1, C2, and C4 are all currently unverified.
