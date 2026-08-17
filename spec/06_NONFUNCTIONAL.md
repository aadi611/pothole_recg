# 06 — Non-Functional Requirements

---

## NFR-PRV — Privacy

A camera continuously filming public roads collects far more than potholes. Treat this as a design constraint, not a disclaimer.

| ID | Requirement |
|---|---|
| NFR-PRV-01 | Automatic face and licence-plate blurring applied to every evidence image **before it is written to disk or transmitted**. Use a lightweight detector (e.g. a small face/plate model) in `evidence/redaction.py`. Record what was redacted in `redaction_applied`. |
| NFR-PRV-02 | If redaction fails or is unavailable, the report is stored **locally only** and flagged `redaction_failed` — it is not uploaded until redaction succeeds or an operator overrides with a logged reason. |
| NFR-PRV-03 | No continuous video retention by default. Store discrete evidence frames. Any optional clip capture is off by default, time-boxed, and locally retained. |
| NFR-PRV-04 | Retention policy in config: evidence images purged after N days unless the pothole is unresolved. Purges are logged. |
| NFR-PRV-05 | GPS traces are report-linked points, not a continuously stored breadcrumb trail of the driver's movements. Aggregate distance-travelled metrics are derived and stored as totals, not raw paths. |
| NFR-PRV-06 | Data deletion path: a device owner can request removal of their device's reports and images; implement it as an actual cascade, not a soft flag. |
| NFR-PRV-07 | Document what is collected, why, and for how long in `docs/PRIVACY.md`. Anyone deploying this in a jurisdiction with dashcam or data-protection rules (India's DPDP Act, GDPR, etc.) needs to check local requirements — say so explicitly rather than implying compliance. |

## NFR-SEC — Security

| ID | Requirement |
|---|---|
| NFR-SEC-01 | HTTPS everywhere in deployment; no plaintext report or image transmission. |
| NFR-SEC-02 | JWT access + refresh tokens; long-lived device tokens for vehicle clients, revocable per device. |
| NFR-SEC-03 | Role-based access: `driver`, `operator`, `admin`. Enforced at the router dependency layer, not in the UI. |
| NFR-SEC-04 | Secrets from environment only. No credentials, API keys, or tokens in YAML, code, or git history. `.env.example` documents required vars with dummy values. |
| NFR-SEC-05 | Evidence images served through authenticated, expiring URLs — never a public unguessable-path directory. |
| NFR-SEC-06 | Validate and re-encode all uploaded images server-side; enforce size and type limits. Never trust a client-supplied path or filename. |
| NFR-SEC-07 | Rate limit ingestion and auth endpoints. |
| NFR-SEC-08 | Passwords hashed with argon2 or bcrypt. Never log tokens, passwords, or full GPS traces. |

## NFR-REL — Reliability & performance

| ID | Requirement |
|---|---|
| NFR-REL-01 | The detection loop must never be blocked by GPS, disk, network, geocoding, or TTS. All of these are async or threaded with timeouts. |
| NFR-REL-02 | Every external call has a timeout and a defined failure behaviour. No unbounded waits. |
| NFR-REL-03 | Bounded queues everywhere. Under sustained overload, drop *frames* (recoverable) — never drop *reports* (not recoverable). |
| NFR-REL-04 | Graceful degradation matrix, implemented and documented: no GPS → detect and alert, store report flagged for manual location; no network → queue locally; no calibration → fallback estimator with lowered confidence; no audio → screen alerts only; no model → hard fail at startup with a clear message. |
| NFR-REL-05 | Clean shutdown on SIGINT/SIGTERM: flush queues, close the camera, finish in-flight captures. |
| NFR-REL-06 | Performance targets (state actual hardware in `docs/PERFORMANCE.md`): ≥15 FPS effective detection on the target device; detection→alert under 300 ms; API p95 under 200 ms for list endpoints; map initial load under 2 s with clustering. |
| NFR-REL-07 | Memory stable over a 2-hour continuous run — no growth in frame buffers, track registry, or image handles. Test it. |

## NFR-OBS — Observability

| ID | Requirement |
|---|---|
| NFR-OBS-01 | Structured JSON logs (structlog) with correlation fields: `session_id`, `track_id`, `report_id`, `device_id`. |
| NFR-OBS-02 | Log every state-machine transition with the value that triggered it — this is the primary debugging tool for tuning thresholds. |
| NFR-OBS-03 | Metrics: FPS, inference latency, detections/min, alerts/min, captures, queue depth, sync successes/failures, submission outcomes by authority. Prometheus-compatible endpoint. |
| NFR-OBS-04 | A per-session debug artefact: annotated output video (optional, off by default) plus a JSONL event log, so a drive can be replayed and analysed afterwards. |
| NFR-OBS-05 | Errors carry context. No bare `except: pass` anywhere in the codebase. |

## NFR-TST — Testing

| ID | Requirement |
|---|---|
| NFR-TST-01 | Unit tests for all pure logic: planar projection, distance smoothing, tracker matching, state transitions, cooldown, severity rules, dedup matching, authority routing precedence. These need no camera, GPS, or DB. |
| NFR-TST-02 | Fixtures: a short recorded driving video with a matching GPX, plus synthetic detection sequences as JSON, committed to `tests/fixtures/`. This is the standing regression input for every phase. |
| NFR-TST-03 | Integration tests: ingest → dedup → geocode → query; offline queue → sync → server state; authority resolution → submission attempt → audit row. |
| NFR-TST-04 | E2E (Playwright): login, live view render, map filter, report detail, status change, manual submission workflow. |
| NFR-TST-05 | Explicit negative tests: no GPS, stale GPS, corrupt frame, disk full, API 500, duplicate report ID, unmatched authority, redaction failure. |
| NFR-TST-06 | A dedicated test asserting that no code path sets `SENT_TO_AUTHORITY` without a recorded successful submission result (FR-AUTH-05). |
| NFR-TST-07 | Coverage target ≥70% on backend logic modules. Coverage is a floor, not the goal — the fixture-video regression run matters more. |

## NFR-DEV — Developer experience & deployment

| ID | Requirement |
|---|---|
| NFR-DEV-01 | `docker compose up` gives a fully working system with seeded demo data: sample potholes, one seeded authority, one demo user per role. |
| NFR-DEV-02 | A single documented command runs the edge pipeline against the fixture video with GPX replay — no camera or vehicle required to develop or review. |
| NFR-DEV-03 | Alembic migrations for every schema change; `create_all()` only in tests. |
| NFR-DEV-04 | Ruff + mypy + pytest + vitest in CI, plus the module-boundary import check. |
| NFR-DEV-05 | `README.md` covers setup, calibration, running edge and server, running tests, and adding a new authority. |
| NFR-DEV-06 | Model weights are versioned artefacts referenced by hash, not committed binaries drifting silently in git. |
| NFR-DEV-07 | `docs/DECISIONS.md` records every deviation from this spec as a short ADR: context, decision, consequence. |

---

## Ethical guardrails (non-negotiable)

1. **Never report a status that did not happen.** Filing civic complaints on someone's behalf is a real-world action. If the system cannot actually file, it says `PENDING_MANUAL` and tells the operator what to do. This is the single most important rule in the project.
2. **Never present estimates as measurements.** Distance and severity are inferred. The UI, the API, and any exported report must carry the confidence and the method.
3. **Never guess an authority.** A misrouted complaint wastes a real person's time and discredits the system.
4. **The driver's attention is the safety-critical resource.** Every UI and alert decision is judged against whether it makes someone look away from the road. When in doubt, show less.
5. **Redact before storing, not before publishing.** Bystanders in the frame did not opt in.
