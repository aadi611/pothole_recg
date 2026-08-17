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

## Next up — Phase 1 (detection pipeline on recorded video)

`FR-CAM-01..06, FR-DET-01..06, FR-TRK-01..06`

Critical path, in order:

1. **Dataset** — fetch Roboflow pothole set (`scripts/fetch_dataset.py`)
2. **Train** — YOLOv8n on RTX 3060; record mAP@50 and mAP@50-95 in `docs/MODEL_CARD.md` [FR-DET-02]
3. **`YoloDetector`** — implement the `Detector` protocol [FR-DET-01]
4. **Camera sources** — video file first (it is how everything is tested), threaded with bounded buffer [FR-CAM-01..02]
5. **Tracker** — IoU + Kalman, stable track IDs, N-frame confirmation [FR-TRK-01..02]
6. **State machine** — through `DETECTED` [FR-TRK-03]
7. **`edge_app` CLI** — `--source video.mp4 --display`

---

## Blocked / waiting

| Item | Blocker | Owner |
|---|---|---|
| PostGIS, `docker compose up` | Docker Desktop not installed | **User** — installing |
| `tests/fixtures/drive.mp4` + `drive.gpx` | Not yet recorded | **User** — 2 min real driving footage + GPS log |
| Camera calibration (mount height, pitch, FOV) | Needs real vehicle measurement | **User** — Phase 2, not blocking yet |
| First authority to seed | Target city not confirmed | **User** — Phase 5, not blocking yet |

The fixture video is the one that matters soonest: it is the standing
regression input for every phase's exit criteria (NFR-TST-02), and it doubles
as calibration ground truth for Phase 2.
