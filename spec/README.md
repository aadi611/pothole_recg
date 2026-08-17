# RoadWatch — AI Pothole Detection & Civic Reporting Platform

Specification set for an AI system that runs on a camera mounted to any moving vehicle and executes the full loop:

> **Detect → Track → Estimate proximity → Warn driver → Auto-capture evidence → Geotag → Classify severity → Route to authority → Report → Aggregate → Map**

**Stack (fixed decisions):** Python backend (FastAPI), React + TypeScript frontend (Vite, npm). See `02_ARCHITECTURE.md` for the full list and the reasoning.

---

## How to use these documents

Read in this order. Load only the file you need for the current task — do not paste the whole set into context at once.

| # | File | Read when |
|---|------|-----------|
| 1 | `01_AGENT_BRIEF.md` | **Always first.** Rules of engagement, what not to do, how to start. |
| 2 | `02_ARCHITECTURE.md` | Before creating any directory or choosing a library. |
| 3 | `03_DATA_CONTRACTS.md` | Before writing any model, schema, endpoint, or DB migration. |
| 4 | `04_REQUIREMENTS.md` | While implementing a feature. Requirements are ID-tagged. |
| 5 | `05_BUILD_PHASES.md` | To pick the next task and to check phase exit criteria. |
| 6 | `06_NONFUNCTIONAL.md` | Privacy, security, offline, observability, testing, deployment. |

---

## Requirement ID scheme

Every requirement in `04_REQUIREMENTS.md` has a stable ID, e.g. `FR-PRX-03`.

```
FR-CAM   Camera & frame pipeline
FR-DET   Detection model layer
FR-TRK   Tracking & detection state machine
FR-PRX   Proximity / distance / time-to-impact
FR-ALT   Driver alerting
FR-EVD   Evidence capture
FR-GEO   GPS, geocoding, geospatial
FR-RPT   Report generation & lifecycle
FR-AUTH  Authority routing & submission
FR-DUP   Duplicate detection & aggregation
FR-API   Backend API
FR-UI    Dashboard, live view, map
FR-SYN   Offline queue & sync
NFR-*    Non-functional (in 06)
```

Reference the ID in every commit message and PR description, e.g.
`feat(proximity): bbox-baseline distance estimator [FR-PRX-01, FR-PRX-02]`.

---

## Progress tracking

Maintain `PROGRESS.md` at the repo root. Update it at the end of every work session:

```markdown
## Phase 2 — Proximity & Alerts
- [x] FR-PRX-01 Road-plane distance estimator — `backend/distance/planar.py`
- [x] FR-PRX-04 Configurable thresholds — `config/detection.yaml`
- [ ] FR-ALT-03 TTS alerts — blocked: needs audio device on target hardware
```

---

## The one-line test for "is this done"

A phone or webcam on a windshield, driven down a real road with no internet, produces geotagged pothole reports with images that appear on a map after the network returns — and the driver was warned before hitting each one.
