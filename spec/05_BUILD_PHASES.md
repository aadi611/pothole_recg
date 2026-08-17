# 05 — Build Phases

Do not begin a phase until the previous phase's exit criteria pass. Each phase ends with a working, demonstrable system — never a half-wired layer.

**MVP flow (Phases 0–4):**
`Camera → Detection → Tracking → Proximity → Driver Alert → Auto Capture → GPS → Report → Backend → Map`

**Then (Phases 5–7):**
`Authority Routing → Government Reporting → Duplicate Detection → Crowdsourced Aggregation → Repair Status → Analytics → Edge/Mobile`

---

## Phase 0 — Audit & foundation

**Deliverables**
- `AUDIT.md` per `01_AGENT_BRIEF.md` — reviewed and approved before anything else.
- Target repo skeleton from `02_ARCHITECTURE.md`; existing valuable code migrated into it.
- `pyproject.toml`, `package.json`, `.env.example`, `docker-compose.yml`, config files with Pydantic validators.
- structlog JSON logging, error types, settings loader.
- Alembic initialised with the base schema.
- CI: lint (ruff), type check (mypy on `core/`, `detection/`, `distance/`), pytest, module-boundary grep check.

**Exit criteria**
- `docker compose up` starts api + db + web; `/health` returns 200; the React app loads and reaches the API.
- Boundary check passes: no model-library import outside `detection/`.
- `AUDIT.md` gap table covers every FR ID in `04_REQUIREMENTS.md`.

---

## Phase 1 — Detection pipeline on recorded video

`FR-CAM-01..06, FR-DET-01..06, FR-TRK-01..06`

**Deliverables**
- Frame sources incl. video file; threaded capture with bounded buffer and frame dropping.
- `Detector` interface + YOLO implementation using existing/retrained weights; `docs/MODEL_CARD.md` with metrics.
- Tracker producing stable IDs; state machine through `DETECTED` (distance states come next phase).
- CLI: `python -m roadwatch.edge_app --source video.mp4 --display` draws boxes, labels, track IDs.
- Session metrics: FPS, inference ms, dropped frames.

**Exit criteria**
- Runs a 2-minute driving video end to end without crashing or leaking memory.
- The same pothole holds one track ID across its visible frames in ≥80% of manual spot-checks.
- Unit tests for tracker matching and state transitions using synthetic detection sequences.
- Meets the target FPS on the agreed hardware, or the shortfall is documented with a mitigation.

---

## Phase 2 — Proximity & driver alerts

`FR-PRX-01..08, FR-PRX-11, FR-ALT-01..06`

**Deliverables**
- Planar-geometry estimator + bbox-scale fallback, both behind `DistanceEstimator`.
- Calibration utility (chessboard + manual two-point ground marking).
- Distance smoothing; time-to-pothole when speed is available.
- Full state machine incl. `APPROACHING` / `VERY_NEAR` driven by configurable thresholds.
- Alert engine with per-track cooldown, global gap, speed suppression; screen + TTS sinks.
- `docs/DISTANCE_LIMITATIONS.md` and `docs/ASSUMPTIONS.md`.

**Exit criteria**
- On a calibrated test video with known ground markings, estimated distance is within ±25% between 5 m and 15 m.
- No alert fires from mere presence in frame — verified with a video where a pothole is visible far ahead for many seconds and produces at most one warn at the threshold.
- One pothole never produces more than one `warn` and one `near`.
- Unit tests: planar projection math against hand-computed values; cooldown logic against a synthetic event stream.

---

## Phase 3 — GPS, evidence capture, local reports

`FR-EVD-01..08, FR-GEO-01..04, FR-RPT-01..02, FR-SYN-01..06`

**Deliverables**
- GPS sources incl. GPX replay synchronised to a video's timeline.
- Automatic best-frame evidence capture on `VERY_NEAR`; full frame + crop; SHA-256.
- Report builder with edge-minted `report_id` and severity + reason.
- Local SQLite store, evidence on disk, sync queue with backoff and storage guard.
- Face and plate redaction applied pre-storage (NFR-PRV-01).

**Exit criteria**
- Video + GPX replay with the network disabled produces N complete local reports with images and correct coordinates.
- Kill the process mid-run: on restart the queue is intact and no report is lost or duplicated.
- Fill the disk to the guard threshold: capture stops cleanly with a driver alert and a loud log, no crash.
- Every report has an image whose SHA-256 matches the stored file.

---

## Phase 4 — Backend, sync, map, dashboard (**MVP complete**)

`FR-API-01..08, FR-GEO-05..08, FR-RPT-03..05, FR-UI-01..16`

**Deliverables**
- FastAPI server: ingestion, query, WebSocket live channel, JWT auth with three roles.
- PostGIS schema via Alembic; image storage behind the storage interface.
- Server-side reverse geocoding with caching, as a retryable background job.
- Uploader syncing the edge queue; idempotent ingest.
- React dashboard: Live Detection, Map, Reports (list + detail), Analytics.
- Generated TS types from OpenAPI.

**Exit criteria**
- Full loop demo: drive (or replay) → alerts fire → evidence captured offline → reconnect → reports sync → markers appear on the map with images and correct city/road.
- Idempotency proven: replay the same batch three times, row counts unchanged.
- Map renders 5,000 seeded potholes smoothly via server-side clustering.
- Integration tests cover ingest → geocode → query → render payload.
- E2E (Playwright): login → map filter → open report detail → change status.

---

## Phase 5 — Authority routing & submission

`FR-AUTH-01..07, FR-AUTH-10, FR-UI-14`

**Deliverables**
- YAML authority config, loader, validator; seed at least three real authorities (one municipal, one state PWD, one national highway) with honest `verified_on` dates and honest channel types.
- Router with road-class → admin-area → bbox precedence, returning the match reason.
- `manual` and `email` submitters; submission audit trail; retry policy.
- Operator manual-submission workflow in the UI.

**Exit criteria**
- Coordinates inside a seeded jurisdiction resolve to the right authority with a stated reason; coordinates outside all jurisdictions return `PENDING_MANUAL` with a reason and **no** guessed authority.
- No code path can set `SENT_TO_AUTHORITY` without a recorded successful channel result. Add an explicit test that asserts this.
- Every attempt, including failures, appears in `submissions`.

---

## Phase 6 — Deduplication & crowdsourced aggregation

`FR-DUP-01..05, FR-RPT-06, FR-UI-17..18`

**Deliverables**
- Spatial dedup on ingest with heading/class compatibility; pothole aggregates with distinct-device confidence.
- Accuracy-weighted centroid recomputation.
- Pothole detail UI showing contributing devices and detection counts.
- Authority performance and worst-roads analytics.

**Exit criteria**
- Three simulated devices passing the same pothole produce **one** pothole entity with three reports and elevated confidence.
- Opposite-carriageway detections 8 m apart produce **two** entities under the default config.
- Dedup adds under 50 ms to ingest at 10k potholes in the DB.

---

## Phase 7 — Edge / mobile

`FR-DET-07..08, FR-PRX-09..10, FR-DUP-06..09, FR-AUTH-08..09`

**Deliverables**
- ONNX/TFLite export and an ONNX detector validated for output equivalence.
- Capacitor wrapper: camera, GPS, background execution, haptics.
- Battery and thermal profiling on a real phone; adaptive sampling under thermal pressure.
- Optional monocular depth cross-check; generic HTTP authority submitter.

**Exit criteria**
- A phone on a windshield runs a 30-minute session offline without thermal shutdown, then syncs cleanly.
- Documented battery drain per hour and sustained FPS.

---

## Standing rules across all phases

- Update `PROGRESS.md` and reference FR IDs in commits.
- Every new module lands with unit tests; every phase lands with at least one integration test of the flow it completes.
- Any deviation from this spec gets written into `docs/DECISIONS.md` with the reason — a short ADR, not a paragraph of prose.
- Keep a recorded test video + GPX pair in `tests/fixtures/` as the standing regression input. Every phase must still pass it.
