# Requirement Status Matrix

Every ID in `spec/04_REQUIREMENTS.md` and `spec/06_NONFUNCTIONAL.md`, with its
current state. **This is the authority on what is built.** Update it in the
same commit that changes the status.

**Legend**
- **DONE** — implemented, tested, verified
- **PARTIAL** — implemented but incomplete or unverified against real data
- **TODO** — not started, in MVP scope (Phases 1–5)
- **LATER** — `[LATER]` in the spec, Phase 6–7
- **BLOCKED** — cannot proceed without an external input

---

## Summary

| Group | DONE | PARTIAL | TODO | LATER | BLOCKED |
|---|---|---|---|---|---|
| FR-CAM (6) | 6 | – | – | – | – |
| FR-DET (8) | 5 | – | – | 2 | 1 |
| FR-TRK (6) | 6 | – | – | – | – |
| FR-PRX (11) | – | – | 9 | 2 | – |
| FR-ALT (7) | – | – | 7 | – | – |
| FR-EVD (9) | – | – | 9 | – | – |
| FR-GEO (8) | – | – | 8 | – | – |
| FR-RPT (6) | – | – | 5 | 1 | – |
| FR-AUTH (11) | – | – | 9 | 2 | – |
| FR-DUP (9) | – | – | 5 | 4 | – |
| FR-API (8) | – | – | 8 | – | – |
| FR-SYN (6) | – | – | 6 | – | – |
| FR-UI (18) | – | – | 16 | 2 | – |
| NFR (43) | 9 | 6 | 24 | 4 | – |
| **Total (151)** | **26** | **6** | **106** | **17** | **1** |

---

## FR-CAM — Camera & frame pipeline ✅ COMPLETE

| ID | Status | Where |
|---|---|---|
| FR-CAM-01 | DONE | `camera/sources.py` — VideoFile, Webcam/RTSP, Synthetic |
| FR-CAM-02 | DONE | `camera/pipeline.py::CaptureThread` — bounded queue, drops oldest |
| FR-CAM-03 | DONE | `camera/pipeline.py::FrameSampler` — every_frame/every_nth/adaptive |
| FR-CAM-04 | DONE | `camera/sources.py::WebcamSource` — backoff reconnect, `is_exhausted` |
| FR-CAM-05 | DONE | `camera/pipeline.py::FrameBuffer` — bounded rolling buffer |
| FR-CAM-06 | DONE | `camera/pipeline.py::SessionMetrics` |

## FR-DET — Detection model layer

| ID | Status | Where / blocker |
|---|---|---|
| FR-DET-01 | DONE | `detection/base.py` + `registry.py`; enforced by `scripts/check_boundaries.py` |
| FR-DET-02 | **BLOCKED** | Needs a training dataset — see PROGRESS.md |
| FR-DET-03 | DONE | `config/detection.yaml` class map |
| FR-DET-04 | DONE | `core/contracts.py::Detection` |
| FR-DET-05 | DONE | `config/detection.yaml` thresholds |
| FR-DET-06 | DONE | `detection/yolo.py::weights_fingerprint` — SHA-256 derived |
| FR-DET-07 | LATER | ONNX export |
| FR-DET-08 | LATER | A/B harness |

## FR-TRK — Tracking & state machine ✅ COMPLETE

| ID | Status | Where |
|---|---|---|
| FR-TRK-01 | DONE | `tracking/tracker.py` — 90% single-ID under realistic noise |
| FR-TRK-02 | DONE | `min_hits` confirmation |
| FR-TRK-03 | DONE | `state/machine.py` — full machine, every transition logged |
| FR-TRK-04 | DONE | `max_age` → LOST; `take_lost()` preserves pending reports |
| FR-TRK-05 | DONE | `TrackedObject.has_been_captured` |
| FR-TRK-06 | DONE | peak_confidence, frames_seen, first/last_seen |

## FR-PRX — Proximity & distance ← **PHASE 2, NEXT**

| ID | Status | Notes |
|---|---|---|
| FR-PRX-01 | DONE | Enforced in `state/machine.py`: no distance ⇒ no advance past DETECTED |
| FR-PRX-02 | TODO | Planar road-plane estimator |
| FR-PRX-03 | TODO | bbox-scale fallback |
| FR-PRX-04 | DONE | `config/proximity.yaml`, Pydantic-validated |
| FR-PRX-05 | TODO | time-to-pothole from GPS speed |
| FR-PRX-06 | DONE | `min_distance_confidence` gate in state machine |
| FR-PRX-07 | TODO | Distance smoothing |
| FR-PRX-08 | TODO | Calibration utility |
| FR-PRX-09 | LATER | Monocular depth |
| FR-PRX-10 | LATER | Stereo |
| FR-PRX-11 | TODO | `docs/DISTANCE_LIMITATIONS.md` |

## FR-ALT — Driver alerting (Phase 2)

All 7 TODO: two levels, cooldown + global gap, pluggable sinks, driving UI,
speed suppression, 300 ms latency budget.

## FR-EVD — Evidence capture (Phase 3)

All 9 TODO: auto-capture on VERY_NEAR, best-frame selection, SHA-256, GPS
binding, local-first write, no continuous video, redaction, retention.

## FR-GEO — GPS & geospatial (Phase 3–4)

All 8 TODO: pluggable sources incl. GPX replay, dropout handling, speed and
heading derivation, server-side reverse geocoding with caching, PostGIS.

## FR-RPT — Reports (Phase 3–4)

FR-RPT-01..05 TODO. FR-RPT-06 LATER (bulk operator actions).

## FR-AUTH — Authority routing (Phase 5)

FR-AUTH-01..07 and 10 TODO. FR-AUTH-08, 09 LATER.
FR-AUTH-05 is the non-negotiable one: `SENT_TO_AUTHORITY` only on a real
accepted delivery. `ReportStatus` already encodes the distinction and
`test_contracts.py` asserts it.

## FR-DUP — Deduplication (Phase 6)

FR-DUP-01..05 TODO. FR-DUP-06..09 LATER.

## FR-API — Backend API (Phase 4)

All 8 TODO.

## FR-SYN — Offline queue (Phase 3)

All 6 TODO.

## FR-UI — Dashboard (Phase 4)

FR-UI-01..16 TODO. FR-UI-17, 18 LATER.

---

## NFR status

| ID | Status | Notes |
|---|---|---|
| NFR-PRV-01..07 | TODO | Redaction, retention, deletion, `docs/PRIVACY.md` |
| NFR-SEC-04 | DONE | `.env.example`; secrets from env only |
| NFR-SEC-01..03, 05..08 | TODO | HTTPS, JWT, RBAC, signed URLs, rate limits, argon2 |
| NFR-REL-01 | PARTIAL | Capture threaded and non-blocking; GPS/disk/net not yet built |
| NFR-REL-02 | PARTIAL | Timeouts on capture reads; no outbound calls yet |
| NFR-REL-03 | DONE | Bounded queues; frames dropped, never reports |
| NFR-REL-04 | PARTIAL | Detector failure degrades; full matrix pending |
| NFR-REL-05 | DONE | `edge_app::_Shutdown` on SIGINT/SIGTERM |
| NFR-REL-06 | PARTIAL | 63 FPS measured, ≥15 target met; API/map targets pending |
| NFR-REL-07 | PARTIAL | Buffers bounded by construction; 2 h soak not yet run |
| NFR-OBS-01 | DONE | `core/logging.py` structlog JSON + contextvars |
| NFR-OBS-02 | DONE | Every transition logs its triggering value |
| NFR-OBS-03 | TODO | Prometheus endpoint |
| NFR-OBS-04 | TODO | Per-session debug artefact |
| NFR-OBS-05 | DONE | `core/errors.py`; no bare except anywhere |
| NFR-TST-01 | DONE | 107 unit tests, no camera/GPS/DB needed |
| NFR-TST-02 | **TODO** | `drive.mp4` + `drive.gpx` fixture not yet recorded |
| NFR-TST-03..06 | TODO | Integration, E2E, negative, FR-AUTH-05 assertion |
| NFR-TST-07 | TODO | ≥70% coverage measurement |
| NFR-DEV-01 | TODO | `docker compose up` with seed data |
| NFR-DEV-02 | DONE | `python -m roadwatch.edge_app --source ...` |
| NFR-DEV-03 | TODO | Alembic |
| NFR-DEV-04 | PARTIAL | ruff + mypy + pytest + boundary check run locally; no CI yet |
| NFR-DEV-05 | TODO | README rewrite |
| NFR-DEV-06 | DONE | Weights by hash, gitignored |
| NFR-DEV-07 | DONE | `docs/DECISIONS.md`, 4 ADRs |

---

## Build order from here

1. **Phase 0 remainder** — `.env.example` ✅, Alembic + DB models, docker-compose, CI, README
2. **Phase 2** — distance estimation + alerts (FR-PRX, FR-ALT) ← unblocked, pure logic
3. **Phase 3** — GPS, evidence, local reports, sync queue
4. **Phase 4** — API, PostGIS, React dashboard
5. **Phase 5** — authority routing
6. **Phase 6–7** — dedup, edge/mobile

Phase 2 is entirely pure geometry and needs no dataset, so it proceeds while
FR-DET-02 stays blocked.
