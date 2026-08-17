# 04 — Functional Requirements

Every requirement is testable and has a stable ID. `[MVP]` = required for Phase 1–4. `[LATER]` = Phase 5+.

---

## FR-CAM — Camera & frame pipeline

| ID | Requirement |
|---|---|
| FR-CAM-01 | `[MVP]` Pluggable frame sources behind one interface: USB/webcam, video file, RTSP, phone stream. Video file source is mandatory — it is how everything gets tested reproducibly. |
| FR-CAM-02 | `[MVP]` Continuous capture in a dedicated thread with a bounded frame buffer. **Drop frames under load; never block capture and never queue unboundedly.** |
| FR-CAM-03 | `[MVP]` Configurable sampling strategy: every frame, every Nth frame, or adaptive (target inference FPS). Detection runs on sampled frames; tracking interpolates between them. |
| FR-CAM-04 | `[MVP]` Handle disconnect, corrupt frame, and zero-byte read without crashing. Auto-reconnect with backoff; log the gap. |
| FR-CAM-05 | `[MVP]` Maintain a short rolling buffer of full-resolution frames so evidence capture can retrieve the best frame retrospectively (see FR-EVD-02). |
| FR-CAM-06 | `[MVP]` Record per-session: frames read, frames processed, dropped frames, avg FPS, avg inference latency. Exposed via API and shown in the UI. |

## FR-DET — Detection model layer

| ID | Requirement |
|---|---|
| FR-DET-01 | `[MVP]` All inference sits behind the `Detector` interface (`03_DATA_CONTRACTS.md §2`). No model-library import outside `detection/`. |
| FR-DET-02 | `[MVP]` Ship a YOLO implementation using the existing project weights if usable; otherwise fine-tune on the existing/public pothole dataset. Record mAP@50 and mAP@50-95 in `docs/MODEL_CARD.md`. |
| FR-DET-03 | `[MVP]` Configurable class map. Ship with `pothole` active; `crack`, `surface_damage`, `manhole`, `debris` defined and switchable without code change. |
| FR-DET-04 | `[MVP]` Output conforms to `Detection`: confidence, bbox, timestamp, frame_id, model_version. Segmentation mask optional and `None` when unsupported. |
| FR-DET-05 | `[MVP]` Configurable confidence threshold, IoU/NMS threshold, input size, device (`cpu`/`cuda`/`mps`). |
| FR-DET-06 | `[MVP]` `model_version` is stamped on every detection and every report — an identifier tied to the weights file hash, not a hand-typed string. |
| FR-DET-07 | `[LATER]` ONNX export path plus a `OnnxDetector` implementation, validated to produce equivalent outputs within tolerance. |
| FR-DET-08 | `[LATER]` A/B harness to compare two detectors on the same recorded video and report precision/recall deltas. |

## FR-TRK — Tracking & detection state machine

| ID | Requirement |
|---|---|
| FR-TRK-01 | `[MVP]` Assign stable `track_id`s across frames (IoU + Kalman, or ByteTrack). Same physical pothole = one track. |
| FR-TRK-02 | `[MVP]` A track is only promoted past `DETECTED` after N consecutive/near-consecutive confirmations (configurable, default 3) — suppresses single-frame false positives. |
| FR-TRK-03 | `[MVP]` Implement the state machine exactly: `UNKNOWN → DETECTED → APPROACHING → VERY_NEAR → CAPTURED → REPORTED`, plus `LOST`. Transitions are driven by distance thresholds and confirmation counts, and every transition is logged with its trigger. |
| FR-TRK-04 | `[MVP]` A track that leaves the frame or is unmatched for M frames goes to `LOST`. A `LOST` track that already reached `CAPTURED` still completes its report. |
| FR-TRK-05 | `[MVP]` Each track produces **at most one** capture and **at most one** report per session. This is the in-session duplicate guard — distinct from cross-vehicle dedup (FR-DUP). |
| FR-TRK-06 | `[MVP]` Track carries peak confidence, frames seen, first/last seen — used for severity and report quality. |

## FR-PRX — Proximity, distance, time-to-impact

| ID | Requirement |
|---|---|
| FR-PRX-01 | `[MVP]` **Alerts must never fire simply because a pothole appears in frame.** Every alert and capture is gated on an estimated distance or time-to-pothole. |
| FR-PRX-02 | `[MVP]` Primary estimator: **road-plane geometry**. Project the bbox bottom-center to the ground plane using camera mount height, pitch, and FOV/intrinsics. Document the flat-road assumption in code and in `docs/ASSUMPTIONS.md`. |
| FR-PRX-03 | `[MVP]` Fallback estimator: bbox scale/perspective heuristic, for when calibration is absent. Lower `confidence` value, and the UI must show it as approximate. |
| FR-PRX-04 | `[MVP]` Thresholds are config, not constants (`config/proximity.yaml`). Reference defaults: detect 20 m, warn 10 m, capture 3–5 m. Treat these as starting points to be tuned on real footage, not as truth. |
| FR-PRX-05 | `[MVP]` When GPS speed is available, compute `time_to_pothole_s = distance_m / speed_ms` and allow a time-based warning threshold. Whichever of distance or time triggers first wins — at 80 km/h, 10 m is under half a second of reaction time. |
| FR-PRX-06 | `[MVP]` Every estimate carries a `confidence` and a list of `assumptions`. Below `min_distance_confidence`, warn the driver but **do not** auto-capture and mark the report `distance_at_capture_m = null`. |
| FR-PRX-07 | `[MVP]` Smooth distance across frames (moving average or Kalman) so the readout does not jitter and thresholds do not flap. |
| FR-PRX-08 | `[MVP]` Ship a calibration utility: chessboard intrinsics capture, plus a manual mode where the user marks two known ground distances in a frame to solve the homography. |
| FR-PRX-09 | `[LATER]` Optional monocular depth estimator (e.g. MiDaS/Depth-Anything class model) behind the same interface, used to cross-check the planar estimate. |
| FR-PRX-10 | `[LATER]` Stereo / depth-camera source support. |
| FR-PRX-11 | `[MVP]` `docs/DISTANCE_LIMITATIONS.md` stating plainly: monocular distance depends on a flat-road assumption, fixed camera pose, and correct intrinsics; it degrades on slopes, potholes filled with water or shadow, bumps that change pitch, and at long range where a few pixels equal several metres. Expected accuracy ±15–25% at 5–15 m with good calibration. The UI must not present these numbers as precise. |

## FR-ALT — Driver warning

| ID | Requirement |
|---|---|
| FR-ALT-01 | `[MVP]` Two levels: `warn` at the warn threshold ("Pothole ahead, 10 metres"), `near` at the capture threshold ("Pothole very near"). |
| FR-ALT-02 | `[MVP]` Per-track cooldown and a global minimum gap between alerts (`config/alerts.yaml`). One pothole = one warn + at most one near. Never repeat-fire for the same track. |
| FR-ALT-03 | `[MVP]` Pluggable sinks: on-screen banner, audible tone, TTS. Haptic on supported devices `[LATER]`. Sinks are selectable in config and degrade silently if unavailable (no audio device must not crash the pipeline). |
| FR-ALT-04 | `[MVP]` Driving UI: large text, high contrast, dark theme, no small controls, no modal dialogs, no interaction required to dismiss an alert. Alerts auto-clear. |
| FR-ALT-05 | `[MVP]` Suppress alerts below a configurable speed (default 5 km/h) — parked or crawling in traffic should be silent. |
| FR-ALT-06 | `[MVP]` Alert latency budget: detection → audible/visible alert under 300 ms on target hardware. Measure it and log it. |

## FR-EVD — Automatic evidence capture

| ID | Requirement |
|---|---|
| FR-EVD-01 | `[MVP]` Capture fires automatically on the `VERY_NEAR` transition. **No button press.** |
| FR-EVD-02 | `[MVP]` Select the best frame from the rolling buffer (highest detection confidence and sharpness within the capture window), not merely the current frame. |
| FR-EVD-03 | `[MVP]` Save the full-resolution frame plus an optional cropped hazard image. JPEG quality configurable. Record SHA-256 of the stored image. |
| FR-EVD-04 | `[MVP]` Bind the exact GPS fix nearest in time to the capture, with its age. If the fix is older than the configured max age, mark `is_stale` and flag the report for review. |
| FR-EVD-05 | `[MVP]` Store metadata per `PotholeReport` in `03_DATA_CONTRACTS.md`. |
| FR-EVD-06 | `[MVP]` Store discrete evidence frames — **never continuous video** in normal operation. Optional short pre/post clip only if explicitly enabled. |
| FR-EVD-07 | `[MVP]` Local-first: write image + report row to local SQLite and disk before any network attempt. Disk-full is handled with a retention policy, not a crash. |
| FR-EVD-08 | `[MVP]` Apply privacy redaction before the image leaves the device (see NFR-PRV-01). |

## FR-GEO — GPS & geospatial

| ID | Requirement |
|---|---|
| FR-GEO-01 | `[MVP]` Pluggable GPS sources: gpsd, serial NMEA, phone/browser geolocation, and a **replay source** that plays a recorded GPX alongside a recorded video (essential for reproducible testing). |
| FR-GEO-02 | `[MVP]` Poll continuously, keep the last known fix with its age, expose accuracy. Never block the detection pipeline on GPS. |
| FR-GEO-03 | `[MVP]` Derive speed and heading from GPS; fall back to computing them from successive fixes when the source does not supply them. |
| FR-GEO-04 | `[MVP]` Handle dropout: tunnels, urban canyons. Reports created during dropout use the last good fix, marked stale, and are flagged in the UI rather than silently trusted. |
| FR-GEO-05 | `[MVP]` Reverse geocoding **server-side** (Nominatim or equivalent), producing city, district, state, country, road name, postcode, OSM way ID. |
| FR-GEO-06 | `[MVP]` Cache geocoding results by rounded coordinate (~10 m grid) and respect the provider's rate limits. Geocoding failure must not block report acceptance — retry as a background job. |
| FR-GEO-07 | `[MVP]` Capture road classification where available (residential / state highway / national highway) — it drives both authority routing and severity. |
| FR-GEO-08 | `[MVP]` Store coordinates as PostGIS geography; all proximity queries use spatial indexes, never brute-force distance loops. |

## FR-RPT — Report generation & lifecycle

| ID | Requirement |
|---|---|
| FR-RPT-01 | `[MVP]` Generate a report automatically on capture, with a `report_id` minted **on the edge** so it is stable before sync. |
| FR-RPT-02 | `[MVP]` Compute severity per `03_DATA_CONTRACTS.md §7`, always with a `severity_reason`. |
| FR-RPT-03 | `[MVP]` Render a human-readable report view (web + printable/PDF export) containing: report ID, timestamp, coordinates, city, road, confidence, severity, direction, distance at capture, evidence image, model version, and the current status. |
| FR-RPT-04 | `[MVP]` Status lifecycle: `draft → queued → submitted_pending → (sent_to_authority | pending_manual) → acknowledged → under_repair → resolved`, with `reopened`, `rejected`, `duplicate`. Every transition writes a `status_events` row with actor and timestamp. |
| FR-RPT-05 | `[MVP]` Status changes are auditable and reversible. `resolved → reopened` is a supported path (potholes come back). |
| FR-RPT-06 | `[LATER]` Operator bulk actions: acknowledge, reject, merge, export a filtered set to CSV/GeoJSON for handover to an authority. |

## FR-AUTH — Authority routing & submission

| ID | Requirement |
|---|---|
| FR-AUTH-01 | `[MVP]` Authorities are defined in YAML config (`03_DATA_CONTRACTS.md §3`). Adding a city or department requires **no code change**. |
| FR-AUTH-02 | `[MVP]` Routing order: road class → admin area → bounding box; highest `priority` among matches wins. The resolver returns the matched authority **and the reason it matched**, both stored and displayed. |
| FR-AUTH-03 | `[MVP]` No match, or an ambiguous match, results in `PENDING_MANUAL` with `authority_id = null` and an explicit reason. **Never guess an authority.** |
| FR-AUTH-04 | `[MVP]` Submitters behind one interface, minimum: `manual` (generate a prefilled submission packet an operator can paste into a portal) and `email`. |
| FR-AUTH-05 | `[MVP]` **Status must reflect reality.** A report only becomes `SENT_TO_AUTHORITY` when a channel actually accepted it and returned a reference or a delivered email. Anything else stays `PENDING_MANUAL`. This rule is non-negotiable — a system that pretends to have filed civic complaints is worse than one that admits it cannot. |
| FR-AUTH-06 | `[MVP]` Persist every attempt in `submissions`: payload, response, success, error, attempt number, timestamp. |
| FR-AUTH-07 | `[MVP]` Retry with exponential backoff and a cap; distinguish retryable (5xx, timeout) from terminal (4xx, validation) failures. |
| FR-AUTH-08 | `[LATER]` Generic HTTP `api` submitter driven entirely by the YAML: endpoint, method, auth from env, payload template, response mapping to `external_reference`. |
| FR-AUTH-09 | `[LATER]` Aggregate submission — file one consolidated complaint for a road segment with N confirmed potholes rather than N separate ones. |
| FR-AUTH-10 | `[MVP]` Authority records carry `verified_on`. Surface a warning in the admin UI when a record is older than 12 months — government contact points change. |

## FR-DUP — Duplicate detection & crowdsourced aggregation

| ID | Requirement |
|---|---|
| FR-DUP-01 | `[MVP]` On ingest, search for an existing pothole within a configurable radius (default 15 m, widened by the report's GPS accuracy) using a PostGIS spatial index. |
| FR-DUP-02 | `[MVP]` Match also requires same hazard class and compatible heading (same carriageway — opposite-direction detections within a few metres are usually the *other* side of the road; make this rule configurable). |
| FR-DUP-03 | `[MVP]` On match, attach the report to the existing `pothole_id` and update: report count, distinct device count, last seen, aggregate confidence, latest image. Do **not** create a second pothole entity. |
| FR-DUP-04 | `[MVP]` Aggregate confidence rises with **distinct devices**, not raw report count — one car passing daily is weaker evidence than ten cars passing once. |
| FR-DUP-05 | `[MVP]` Recompute the pothole centroid as an accuracy-weighted mean of its reports. |
| FR-DUP-06 | `[LATER]` Image-similarity check (perceptual hash / embedding) as a secondary confirmation signal. |
| FR-DUP-07 | `[LATER]` Road-segment snapping (OSM way ID) so dedup follows the road rather than raw radius. |
| FR-DUP-08 | `[LATER]` Operator merge/split tools for when automatic dedup gets it wrong. |
| FR-DUP-09 | `[LATER]` Auto-resolution heuristic: a pothole with no new detections after K passes by ≥2 devices over the same segment is flagged *possibly repaired* — flagged for review, never auto-closed. |

## FR-API — Backend API

| ID | Requirement |
|---|---|
| FR-API-01 | `[MVP]` Implement the endpoints in `03_DATA_CONTRACTS.md §6`. Pydantic validation on every input; RFC 7807 errors. |
| FR-API-02 | `[MVP]` `POST /reports` is idempotent on `report_id` — a resubmitted offline report must never create a duplicate row. |
| FR-API-03 | `[MVP]` `POST /reports/batch` returns per-item success/failure so a partially-failing sync does not lose the successful items. |
| FR-API-04 | `[MVP]` Cursor pagination on all list endpoints. `GET /potholes/geojson` returns server-clustered features by zoom level — never ship 50,000 raw points to the browser. |
| FR-API-05 | `[MVP]` JWT auth with roles: `driver` (submit, view own), `operator` (view all, change status, submit to authority), `admin` (everything, manage authorities). |
| FR-API-06 | `[MVP]` OpenAPI schema published; frontend types generated from it. |
| FR-API-07 | `[MVP]` Rate limiting on ingestion endpoints. |
| FR-API-08 | `[MVP]` `/health` (liveness) and `/ready` (DB + storage reachable) endpoints. |

## FR-SYN — Offline queue & synchronisation

| ID | Requirement |
|---|---|
| FR-SYN-01 | `[MVP]` **A confirmed pothole is never lost to a network failure.** Detect → capture → save locally → queue → sync when connectivity returns. |
| FR-SYN-02 | `[MVP]` Local SQLite queue with attempt count, last error, next attempt time. Survives process restart and device reboot. |
| FR-SYN-03 | `[MVP]` Background uploader with exponential backoff; batches when many reports are pending; never blocks detection. |
| FR-SYN-04 | `[MVP]` Local evidence retained until the server confirms receipt, then subject to a retention policy. Deletion is driven by confirmed sync, never by a timer alone. |
| FR-SYN-05 | `[MVP]` Storage guard: when free disk falls below a threshold, stop capturing new evidence, alert the driver, and log loudly — do not silently drop reports. |
| FR-SYN-06 | `[MVP]` Sync status visible in the UI: queued count, last successful sync, current error. |

## FR-UI — Dashboard

### Live Detection view
| ID | Requirement |
|---|---|
| FR-UI-01 | `[MVP]` Live camera feed with bounding-box overlay, class label, and confidence. |
| FR-UI-02 | `[MVP]` Per-track distance estimate with its confidence shown as approximate (e.g. "~11 m") — never a false-precision decimal. |
| FR-UI-03 | `[MVP]` Status strip: current GPS, speed, heading, resolved road name, GPS accuracy/staleness, detection state, FPS, inference latency, sync queue depth. |
| FR-UI-04 | `[MVP]` Alert banner obeying FR-ALT-04 driving constraints. |
| FR-UI-05 | `[MVP]` Session summary on stop: distance driven, potholes detected, reports created, alerts issued. |

### Map view
| ID | Requirement |
|---|---|
| FR-UI-06 | `[MVP]` MapLibre map of all potholes, clustered at low zoom, individual markers at high zoom. |
| FR-UI-07 | `[MVP]` Marker colour by severity, marker shape or badge by status. Legend always visible. |
| FR-UI-08 | `[MVP]` Marker popup: location, road, severity, aggregate confidence, first/last seen, independent report count, latest image thumbnail, status, link to detail. |
| FR-UI-09 | `[MVP]` Filters: status, severity, date range, hazard class, device, authority. Filters reflected in the URL so views are shareable. |
| FR-UI-10 | `[LATER]` Heatmap layer and per-road-segment defect density. |

### Reports view
| ID | Requirement |
|---|---|
| FR-UI-11 | `[MVP]` Sortable, filterable report table with pagination. |
| FR-UI-12 | `[MVP]` Detail page: full metadata, evidence image (with redaction visible), all linked reports for that pothole, resolved authority and why, submission attempt history, status timeline. |
| FR-UI-13 | `[MVP]` Operator status change with a required note. |
| FR-UI-14 | `[MVP]` Manual submission workflow: show the prefilled packet, a copy button, a link to the portal, and a "mark as submitted, enter reference" action. |

### Analytics view
| ID | Requirement |
|---|---|
| FR-UI-15 | `[MVP]` Totals: potholes detected, reports submitted, open, resolved, high/critical severity, distance travelled, active devices. |
| FR-UI-16 | `[MVP]` Trend over time and severity distribution charts. |
| FR-UI-17 | `[LATER]` Authority performance: average time to acknowledge and to resolve, per authority, against its SLA. |
| FR-UI-18 | `[LATER]` Worst-roads leaderboard by defect density per km. |
