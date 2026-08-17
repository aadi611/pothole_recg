# PROGRESS

Updated at the end of every work session (per `spec/01_AGENT_BRIEF.md`).

---

## Session 2026-08-18

### Phase 0 — Audit & foundation

- [x] `AUDIT.md` — full codebase audit, gap table covering every FR/NFR ID
- [x] Spec versioned: `files/` → `spec/` (was untracked)
- [x] `.gitignore` — weights, datasets, evidence, secrets excluded before any commit
- [x] Old TensorFlow classifier deleted (see ADR-001)
- [x] Salvaged: augmentation recipe → `backend/roadwatch/training/augment.py` (now bbox-aware)
- [x] Salvaged: dataset URLs → `docs/DATASETS.md`
- [x] Core contracts → `backend/roadwatch/core/contracts.py` [FR-DET-04]
- [x] Config + Pydantic validation → `config/*.yaml`, `core/config.py`
- [x] Structured logging → `core/logging.py` [NFR-OBS-01]
- [x] Error taxonomy → `core/errors.py` [NFR-OBS-05]
- [x] `Detector` protocol → `detection/base.py` [FR-DET-01]
- [x] Module boundary check → `scripts/check_boundaries.py` [NFR-DEV-04]
- [x] Module skeleton: 18 packages under `backend/roadwatch/`
- [x] Environment verified: CUDA 12.4, torch 2.6.0+cu124, RTX 3060 6GB

**Verified:** 29 unit tests pass · ruff clean · mypy strict clean on `core/` + `detection/` · boundary check passes

### Remaining for Phase 0 exit criteria

- [ ] `docker-compose.yml` — api + db (postgis) + web — **blocked: Docker Desktop not yet installed**
- [ ] Alembic init + base schema [NFR-DEV-03]
- [ ] `.env.example` [NFR-SEC-04]
- [ ] `web/` React + Vite + Tailwind skeleton
- [ ] CI workflow (ruff, mypy, pytest, boundary check)
- [ ] `README.md` rewrite [NFR-DEV-05]

---

### Phase 1 — Detection pipeline on recorded video

`FR-CAM-01..06, FR-DET-01..06, FR-TRK-01..06`

- [x] Frame sources: video file, webcam/RTSP, synthetic [FR-CAM-01, FR-CAM-04]
- [x] Threaded capture, bounded queue, frame dropping [FR-CAM-02]
- [x] Rolling evidence buffer [FR-CAM-05]
- [x] Sampling strategies: every_frame / every_nth / adaptive [FR-CAM-03]
- [x] Session metrics: FPS, inference ms, drops [FR-CAM-06]
- [x] `Detector` protocol + registry [FR-DET-01]
- [x] `YoloDetector` with hash-derived `model_version` [FR-DET-04..06]
- [x] Config-driven class map [FR-DET-03]
- [x] Tracker: IoU + Kalman, stable IDs, N-frame confirmation [FR-TRK-01..02]
- [x] Track bookkeeping, LOST handling, capture guard [FR-TRK-04..06]
- [x] State machine through `DETECTED` [FR-TRK-03]
- [x] `edge_app` CLI with annotated display
- [x] `scripts/train_detector.py` + model card generation
- [ ] **Train on real pothole data** [FR-DET-02] — **blocked: needs dataset**

**Measured on RTX 3060** (stock yolov8n, 300-frame video):

| | |
|---|---|
| Inference | 7.8 ms/frame |
| Effective FPS | 63 (target: ≥15, NFR-REL-06) |
| Frames read / processed / dropped | 300 / 300 / 0 |

**Tracker tuning** — `scripts/bench_tracker.py` measured the association
parameters rather than guessing them:

| config | continuity | separation |
|---|---|---|
| iou=0.30 max_age=10 | 54% | 86% |
| iou=0.15 max_age=15 | **90%** | **100%** |

The original default failed the ≥80% exit criterion. Both metrics are now
regression-tested in `tests/unit/test_tracker.py`.

**Verified:** 107 unit tests pass · ruff clean · mypy strict clean · boundary check passes

### Phase 1 exit criteria status

| Criterion | Status |
|---|---|
| Runs a 2-min video without crashing or leaking | Partial — 10s synthetic clip clean; needs the real fixture |
| Same pothole holds one track ID in ≥80% of checks | **Met** — 90% measured under realistic noise |
| Unit tests for tracker matching and state transitions | **Met** — 107 tests |
| Meets target FPS on agreed hardware | **Met** — 63 FPS vs ≥15 target |

---

## Blocked / waiting

| Item | Blocker | Owner |
|---|---|---|
| **Train the pothole model** [FR-DET-02] | No dataset — every good public set needs an account | **User** — Roboflow export URL |
| `tests/fixtures/drive.mp4` + `drive.gpx` | Not yet recorded | **User** — 2 min real driving footage + GPS log |
| PostGIS, `docker compose up` | Docker Desktop not installed | **User** — installing |
| Camera calibration (mount height, pitch, FOV) | Needs real vehicle measurement | **User** — Phase 2 |
| First authority to seed | Target city not confirmed | **User** — Phase 5 |

**Dataset access, investigated 2026-08-18:** Roboflow export links are
account-specific and expire; RDD2022's S3 bucket returns 403; the Kaggle sets
need an API token; figshare serves an HTML interstitial rather than the file.
The only unauthenticated set found (`keremberke/pothole-segmentation`) has 99
images and ships no YOLO label files — far too small and in the wrong format.
So this genuinely needs a human with an account.

To unblock: sign in at https://universe.roboflow.com/search?q=pothole, pick a
dataset, **Download → YOLOv8 → show download code**, and pass the URL to
`python scripts/fetch_dataset.py --url '<link>'`.

The fixture video matters next: it is the standing regression input for every
phase's exit criteria (NFR-TST-02) and doubles as Phase 2 calibration ground
truth.
