# 02 — Architecture

## Stack decisions (fixed — do not substitute without raising it)

### Backend / AI — Python 3.11
| Concern | Choice | Why |
|---|---|---|
| API | **FastAPI** + Uvicorn | Async, native WebSockets for the live feed, Pydantic validation for free |
| Validation / DTOs | **Pydantic v2** | Single source of truth for contracts; exports OpenAPI → TS types |
| ORM | **SQLAlchemy 2.0** | Mature, works with both Postgres and SQLite |
| Migrations | **Alembic** | Non-negotiable; no `create_all()` in anything but tests |
| DB (server) | **PostgreSQL 16 + PostGIS** | Geospatial dedup and radius queries are the core of this product |
| DB (edge/offline) | **SQLite** | Local queue and offline store on the vehicle device |
| CV runtime | **OpenCV** + **Ultralytics YOLO** behind an interface | Fastest path to working detection; interface keeps it replaceable |
| Inference portability | **ONNX Runtime** export path | Required for the edge/mobile phase |
| Tracking | **ByteTrack / BoT-SORT** (via Ultralytics) or standalone IoU+Kalman | Cheap, well understood, no re-ID model needed |
| Background jobs | **arq** (Redis) — or an in-process asyncio worker if Redis is unavailable | Sync queue, reverse geocoding, authority submission retries |
| Object storage | Local filesystem, S3-compatible interface | Evidence images; swap to MinIO/S3 later without code change |
| Auth | **JWT** (access + refresh), device tokens for vehicle clients | Fleet devices need long-lived credentials |
| Config | **pydantic-settings** + YAML | Env for secrets, YAML for tunables |
| Logging | **structlog** → JSON | Machine-readable; correlate by `report_id` / `track_id` |
| Testing | **pytest**, `pytest-asyncio`, `httpx` | — |

### Frontend — React 18 + TypeScript
| Concern | Choice | Why |
|---|---|---|
| Build | **Vite** (npm) | Fast HMR, trivial static build |
| Styling | **Tailwind CSS** + **shadcn/ui** | Consistent, accessible primitives; fast to build a real dashboard |
| Maps | **MapLibre GL JS** + OpenStreetMap tiles | No API key, no vendor lock, clustering and heatmaps built in |
| Server state | **TanStack Query** | Caching, polling, retries for report lists |
| Client state | **Zustand** | Live-detection state without Redux ceremony |
| Charts | **Recharts** | Analytics panels |
| Live feed | **WebSocket** (native) + MJPEG fallback | Frames + detection overlay metadata |
| Icons | **lucide-react** | — |
| Tests | **Vitest** + **React Testing Library**, **Playwright** for E2E | — |

### Deployment
Docker Compose: `api`, `worker`, `db` (postgis), `redis`, `web` (nginx-served build). One `docker compose up` must bring up a working system with seeded demo data.

> **Edge/mobile (later phase):** the same React UI wrapped with **Capacitor** for camera + GPS + haptics access, talking to a local Python inference service or an on-device ONNX/TFLite model. Do not build this before Phase 6; just do not architect anything that blocks it.

---

## Pipeline

```
                     ┌─────────────────────────────────────────┐
  Vehicle camera ───▶│ camera/    frame acquisition, buffering  │
                     └──────────────────┬──────────────────────┘
                                        ▼
                     ┌─────────────────────────────────────────┐
                     │ detection/  Detector interface → YOLO    │  ← swappable
                     └──────────────────┬──────────────────────┘
                                        ▼
                     ┌─────────────────────────────────────────┐
                     │ tracking/   stable track IDs, dedup      │
                     └──────────────────┬──────────────────────┘
                                        ▼
                     ┌─────────────────────────────────────────┐
                     │ distance/   distance + time-to-pothole   │
                     └──────────────────┬──────────────────────┘
                                        ▼
                     ┌─────────────────────────────────────────┐
                     │ state/      UNKNOWN→…→REPORTED machine   │
                     └───────┬──────────────────────┬──────────┘
                             ▼                      ▼
                  ┌────────────────────┐  ┌────────────────────────┐
                  │ alerts/ driver warn│  │ evidence/ auto-capture │
                  └────────────────────┘  └───────────┬────────────┘
                                                      ▼
        gps/ ────────────────────────────▶ ┌────────────────────────┐
        (lat, lon, speed, heading)         │ reports/ build report  │
                                           └───────────┬────────────┘
                                                       ▼
                                           ┌────────────────────────┐
                                           │ sync/ offline queue    │
                                           └───────────┬────────────┘
                                                       ▼
                                           ┌────────────────────────┐
                                           │ backend API            │
                                           └───┬────────────┬───────┘
                                               ▼            ▼
                                    ┌──────────────┐  ┌──────────────┐
                                    │ dedup/       │  │ geocoding/   │
                                    └──────┬───────┘  └──────┬───────┘
                                           ▼                 ▼
                                    ┌──────────────────────────────┐
                                    │ authority/ routing + submit  │
                                    └──────────────┬───────────────┘
                                                   ▼
                                    ┌──────────────────────────────┐
                                    │ database + storage → map/UI  │
                                    └──────────────────────────────┘
```

**Two runtimes, one codebase:**
- **Edge runtime** (`edge_app.py`) — camera, detection, tracking, distance, alerts, evidence, GPS, local SQLite queue. Runs on the vehicle. Must work with zero network.
- **Server runtime** (`server_app.py`) — API, dedup, geocoding, authority routing, analytics, dashboard. Runs in the cloud or on a laptop.

They communicate only through the documented API in `03_DATA_CONTRACTS.md`. For a single-machine demo both run on the same box; that must not be an assumption anywhere in the code.

---

## Repository layout

```
roadwatch/
├── README.md  PROGRESS.md  AUDIT.md
├── docker-compose.yml
├── config/
│   ├── detection.yaml          # model, confidence, NMS, class map
│   ├── proximity.yaml          # camera calibration, thresholds
│   ├── alerts.yaml             # cooldowns, modalities, phrasing
│   ├── authorities/            # one YAML per city/region
│   │   ├── _schema.yaml
│   │   ├── in-up-gautam-buddha-nagar.yaml
│   │   └── in-nhai.yaml
│   └── logging.yaml
├── backend/
│   ├── pyproject.toml
│   ├── roadwatch/
│   │   ├── camera/             # sources: webcam, file, RTSP, phone stream
│   │   ├── detection/          # base.Detector, yolo.py, registry.py
│   │   ├── tracking/           # tracker.py, track.py
│   │   ├── distance/           # base.py, planar.py, bbox_scale.py, monodepth.py
│   │   ├── state/              # detection state machine
│   │   ├── alerts/             # engine.py, sinks/{console,tts,ws,haptic}.py
│   │   ├── evidence/           # capture.py, redaction.py, storage.py
│   │   ├── gps/                # sources: gpsd, serial NMEA, phone, replay
│   │   ├── geocoding/          # nominatim.py, cache.py
│   │   ├── reports/            # builder.py, severity.py, ids.py
│   │   ├── dedup/              # spatial.py, visual.py, merge.py
│   │   ├── authority/          # router.py, submitters/{api,email,portal,manual}.py
│   │   ├── sync/               # queue.py, uploader.py
│   │   ├── api/                # routers/, deps.py, ws.py
│   │   ├── db/                 # models.py, session.py, migrations/
│   │   ├── analytics/
│   │   ├── core/               # config, logging, errors, metrics
│   │   ├── edge_app.py
│   │   └── server_app.py
│   └── tests/
│       ├── unit/  integration/  fixtures/
└── web/
    ├── package.json
    └── src/
        ├── pages/              # LiveDetection, Map, Reports, ReportDetail, Analytics
        ├── components/
        ├── features/           # live/, map/, reports/, authority/
        ├── api/                # generated types + query hooks
        ├── stores/
        └── lib/
```

---

## Module boundary rules

| Rule | Enforcement |
|---|---|
| Only `detection/` imports the model library | Lint check in CI: grep for `ultralytics`/`torch` outside `detection/` |
| Only `db/` imports SQLAlchemy | Same |
| `distance/`, `tracking/`, `dedup/` are pure functions of their inputs — no I/O, no DB, no logging side effects beyond structlog | Unit-testable with synthetic data |
| `authority/` submitters share one `Submitter` interface | Adding a city = adding a YAML file, not writing code |
| Frontend never talks to the DB or the model | API only |
| Every cross-module payload is a Pydantic model from `03_DATA_CONTRACTS.md` | No bare dicts across boundaries |

---

## Configuration

Secrets → environment variables (`.env`, never committed). Tunables → YAML, hot-reloadable where practical.

```yaml
# config/proximity.yaml
camera:
  id: "cam-front-01"
  mount_height_m: 1.25          # ground to lens
  pitch_deg: -8.0               # negative = angled down
  horizontal_fov_deg: 78.0
  resolution: [1280, 720]
  calibration_file: null        # optional: intrinsics from chessboard calibration

estimator: "planar"             # planar | bbox_scale | monodepth | stereo
fallback_estimator: "bbox_scale"

thresholds_m:
  detect: 20.0
  warn: 10.0
  capture: 4.0

time_to_pothole_s:
  warn: 1.5                     # used when speed is available; whichever fires first

confidence:
  min_distance_confidence: 0.4  # below this, warn but do not auto-capture
```

```yaml
# config/alerts.yaml
cooldown_s: 6.0                 # per track_id
global_min_gap_s: 1.5           # never two alerts closer than this
sinks: ["screen", "tts"]
tts:
  engine: "pyttsx3"
  warn_template: "Pothole ahead, {distance_m:.0f} meters"
  near_template: "Pothole very near"
suppress_below_speed_kmh: 5.0   # no alerts when crawling or stopped
```

Every config file is validated by a Pydantic model at startup. Invalid config = hard fail with a readable error, never a silent default.
