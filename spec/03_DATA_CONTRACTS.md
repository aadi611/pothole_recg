# 03 — Data Contracts

These are the interfaces every module codes against. Change them by editing this file first, then the code.

---

## 1. Core domain objects (Pydantic v2)

```python
# roadwatch/core/contracts.py
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

class HazardClass(str, Enum):
    POTHOLE = "pothole"
    CRACK = "crack"
    SURFACE_DAMAGE = "surface_damage"
    MANHOLE = "manhole"
    DEBRIS = "debris"
    UNKNOWN = "unknown"

class BBox(BaseModel):
    """Pixel coords, origin top-left."""
    x1: float; y1: float; x2: float; y2: float
    @property
    def width(self) -> float: return self.x2 - self.x1
    @property
    def height(self) -> float: return self.y2 - self.y1
    @property
    def bottom_center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, self.y2)   # ground contact point

class Detection(BaseModel):
    """Single-frame model output. No tracking, no distance."""
    frame_id: int
    timestamp: datetime
    hazard_class: HazardClass
    confidence: float = Field(ge=0, le=1)
    bbox: BBox
    mask_rle: str | None = None          # optional segmentation, RLE-encoded
    model_version: str

class DistanceEstimate(BaseModel):
    distance_m: float
    time_to_pothole_s: float | None      # requires speed
    confidence: float = Field(ge=0, le=1)
    method: str                          # "planar" | "bbox_scale" | "monodepth" | ...
    assumptions: list[str] = []          # human-readable, surfaced in the UI

class TrackState(str, Enum):
    UNKNOWN = "UNKNOWN"
    DETECTED = "DETECTED"
    APPROACHING = "APPROACHING"
    VERY_NEAR = "VERY_NEAR"
    CAPTURED = "CAPTURED"
    REPORTED = "REPORTED"
    LOST = "LOST"

class Track(BaseModel):
    track_id: str
    hazard_class: HazardClass
    state: TrackState
    first_seen: datetime
    last_seen: datetime
    frames_seen: int
    peak_confidence: float
    latest: Detection
    distance: DistanceEstimate | None = None
    estimated_size_m2: float | None = None

class GpsFix(BaseModel):
    latitude: float; longitude: float
    accuracy_m: float | None = None
    speed_kmh: float | None = None
    heading_deg: float | None = None     # 0=N, clockwise
    altitude_m: float | None = None
    timestamp: datetime
    source: str                          # "gpsd" | "nmea" | "phone" | "replay"
    is_stale: bool = False               # last fix older than config max age

class Severity(str, Enum):
    LOW = "low"; MEDIUM = "medium"; HIGH = "high"; CRITICAL = "critical"

class ReportStatus(str, Enum):
    DRAFT = "draft"
    QUEUED = "queued"                    # offline, awaiting sync
    SUBMITTED_PENDING = "submitted_pending"   # sent to server, not yet routed
    PENDING_MANUAL = "pending_manual"    # no automated channel exists — human action required
    SENT_TO_AUTHORITY = "sent_to_authority"
    ACKNOWLEDGED = "acknowledged"
    UNDER_REPAIR = "under_repair"
    RESOLVED = "resolved"
    REOPENED = "reopened"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"              # merged into a parent pothole

class LocationContext(BaseModel):
    city: str | None = None
    district: str | None = None
    state: str | None = None
    country: str | None = None
    road_name: str | None = None
    postcode: str | None = None
    osm_way_id: str | None = None
    geocoder: str | None = None
    geocoded_at: datetime | None = None

class PotholeReport(BaseModel):
    report_id: str                       # PTH-YYYYMMDD-NNNNNN (see §4)
    pothole_id: str | None = None        # assigned server-side after dedup
    device_id: str
    camera_id: str
    detected_at: datetime
    gps: GpsFix
    location: LocationContext = LocationContext()
    hazard_class: HazardClass
    detection_confidence: float
    distance_at_capture_m: float | None
    estimated_size_m2: float | None
    severity: Severity
    severity_reason: str                 # explain the classification
    image_path: str
    crop_path: str | None
    image_sha256: str
    redaction_applied: list[str] = []    # ["faces", "plates"]
    model_version: str
    app_version: str
    status: ReportStatus
    authority_id: str | None = None
    submission_attempts: int = 0
    created_at: datetime
    updated_at: datetime

class Pothole(BaseModel):
    """Server-side aggregate: one physical defect, many reports."""
    pothole_id: str
    latitude: float; longitude: float    # weighted centroid
    location: LocationContext
    hazard_class: HazardClass
    severity: Severity
    confidence: float                    # rises with independent confirmations
    report_count: int
    distinct_device_count: int
    first_reported_at: datetime
    last_seen_at: datetime
    status: ReportStatus
    authority_id: str | None
    latest_image_path: str | None
```

---

## 2. Swappable interfaces

```python
# roadwatch/detection/base.py
class Detector(Protocol):
    model_version: str
    def warmup(self) -> None: ...
    def detect(self, frame: np.ndarray, frame_id: int) -> list[Detection]: ...
    def close(self) -> None: ...
```
Register implementations in `detection/registry.py`; select via `config/detection.yaml`.
**Nothing outside `detection/` may import the model library.**

```python
# roadwatch/distance/base.py
class DistanceEstimator(Protocol):
    method: str
    def estimate(self, det: Detection, frame_shape: tuple[int, int],
                 gps: GpsFix | None) -> DistanceEstimate: ...
```

```python
# roadwatch/gps/base.py
class GpsSource(Protocol):
    def read(self) -> GpsFix | None: ...   # non-blocking, returns last known
    def close(self) -> None: ...
```

```python
# roadwatch/alerts/base.py
class AlertSink(Protocol):
    def emit(self, level: Literal["warn","near"], track: Track, text: str) -> None: ...
```

```python
# roadwatch/authority/base.py
class Submitter(Protocol):
    channel: str        # "api" | "email" | "portal" | "manual"
    def submit(self, report: PotholeReport, authority: Authority) -> SubmissionResult: ...

class SubmissionResult(BaseModel):
    success: bool
    status: ReportStatus
    external_reference: str | None = None
    message: str
    retryable: bool
    attempted_at: datetime
```

---

## 3. Authority configuration schema

One YAML per authority in `config/authorities/`. Adding a city must require **no code**.

```yaml
# config/authorities/in-up-gautam-buddha-nagar.yaml
authority_id: "in-up-gbn-noida-authority"
name: "New Okhla Industrial Development Authority (Noida Authority)"
level: "municipal"                # municipal | district | state | national
country: "IN"
state: "Uttar Pradesh"
jurisdiction:
  match:
    - type: "admin_area"
      city: ["Noida", "Gautam Buddha Nagar"]
    - type: "bbox"
      bbox: [77.28, 28.44, 77.60, 28.66]   # minLon, minLat, maxLon, maxLat
  priority: 100                   # higher wins when multiple authorities match
  excludes:
    - type: "road_class"
      values: ["national_highway"]   # NHAI owns these — see in-nhai.yaml
channel:
  type: "manual"                  # api | email | portal | manual
  # type: api
  #   endpoint, method, auth (env var name), payload_template, response_mapping
  # type: email
  #   to, cc, subject_template, body_template, attach_evidence: true
  portal_url: "https://.../grievance"
  instructions: |
    No public API. Report is queued as PENDING_MANUAL with a prefilled
    submission packet the operator can paste into the portal.
contact:
  email: null
  phone: null
sla_days: 15
verified_on: "2026-08-18"         # authority data goes stale — track this
```

**Routing rule (`authority/router.py`):** match by road class first (national highway → NHAI, state highway → state PWD), then admin area, then bbox. Highest `priority` among matches wins. **No match → `PENDING_MANUAL` with `authority_id = null` and a clear reason.** Never guess.

---

## 4. Identifiers

- `report_id` — `PTH-YYYYMMDD-NNNNNN`, sequence per device per day. Generated **on the edge** so offline reports have stable IDs before sync.
- `pothole_id` — `PH-` + ULID, assigned server-side after dedup.
- `track_id` — UUIDv4, session-scoped, never persisted as a business key.
- `device_id` — UUIDv4 issued at device registration.

---

## 5. Database (PostgreSQL + PostGIS)

```
devices(id, name, vehicle_type, camera_id, app_version, registered_at, last_seen_at, owner_user_id)
users(id, email, password_hash, role, created_at)                 -- role: driver | operator | admin
potholes(id, geom GEOGRAPHY(POINT,4326), hazard_class, severity, confidence,
         report_count, distinct_device_count, status, authority_id,
         location_json, first_reported_at, last_seen_at, latest_image_id)
reports(id, pothole_id FK, device_id FK, detected_at, geom GEOGRAPHY(POINT,4326),
        speed_kmh, heading_deg, gps_accuracy_m, hazard_class, detection_confidence,
        distance_at_capture_m, estimated_size_m2, severity, severity_reason,
        model_version, app_version, status, image_id, crop_image_id,
        image_sha256, redaction_json, created_at, updated_at)
images(id, storage_key, sha256, width, height, bytes, redaction_json, created_at)
authorities(id, name, level, country, state, config_json, sla_days, verified_on)
submissions(id, report_id FK, pothole_id FK, authority_id FK, channel, attempt_no,
            success, status, external_reference, message, payload_json,
            response_json, attempted_at)
status_events(id, pothole_id FK, from_status, to_status, actor, note, created_at)
detection_metrics(id, device_id, session_id, window_start, window_end,
                  frames, detections, avg_fps, avg_inference_ms, distance_km)
sync_queue(id, report_id, payload_json, attempts, last_error, next_attempt_at)  -- SQLite, edge only
```

**Required indexes:** GiST on `potholes.geom` and `reports.geom`; btree on `reports.detected_at`, `reports.status`, `reports.pothole_id`, `submissions.report_id`.

Every schema change ships as an Alembic migration. No exceptions.

---

## 6. API surface

`/api/v1`, JSON, JWT bearer auth. Errors follow RFC 7807 (`type`, `title`, `status`, `detail`, `instance`).

### Ingestion (device → server)
```
POST   /devices/register            → device_id + device token
POST   /reports                     multipart: report JSON + image + crop
                                    Idempotent on report_id. Returns pothole_id after dedup.
POST   /reports/batch               offline sync; array, partial success per item
GET    /reports/{report_id}
```

### Query (dashboard)
```
GET    /potholes?bbox=&status=&severity=&since=&limit=&cursor=
GET    /potholes/geojson?bbox=&zoom=       → FeatureCollection for MapLibre
GET    /potholes/{id}                       → detail + all reports + status timeline
PATCH  /potholes/{id}/status                → {status, note}  (operator/admin only)
GET    /potholes/{id}/reports
GET    /analytics/summary?from=&to=         → totals, open/resolved, severity mix, km driven
GET    /analytics/hotspots?bbox=            → clustered high-density segments
```

### Authority
```
GET    /authorities
GET    /authorities/resolve?lat=&lon=&road_class=   → matched authority + why, or null + reason
POST   /potholes/{id}/submit                        → trigger/retry submission
GET    /submissions?report_id=&status=
```

### Live (edge → dashboard)
```
WS     /ws/live/{device_id}
```
Server→client frames:
```json
{"type":"detection_frame","frame_id":1042,"ts":"...","fps":18.4,
 "tracks":[{"track_id":"...","bbox":[x1,y1,x2,y2],"confidence":0.91,
            "state":"APPROACHING","distance_m":11.3,"distance_confidence":0.62,
            "time_to_pothole_s":1.8}],
 "gps":{"lat":28.57,"lon":77.32,"speed_kmh":42.0,"heading_deg":47.0,"is_stale":false},
 "jpeg_b64":"..."}
{"type":"alert","level":"warn","track_id":"...","text":"Pothole ahead, 10 meters"}
{"type":"capture","report_id":"PTH-20260818-000124"}
```

**Generate TypeScript types from the OpenAPI schema** (`openapi-typescript`) — never hand-write frontend types.

---

## 7. Severity rules (`reports/severity.py`)

Deterministic, explainable, config-driven. Output includes `severity_reason`.

```
estimated_area_m2 (from bbox + planar projection)
  < 0.1  → LOW
  < 0.3  → MEDIUM
  < 0.8  → HIGH
  ≥ 0.8  → CRITICAL

Escalate one level if any:
  - road_class in [national_highway, state_highway]   (speed → danger)
  - distinct_device_count ≥ 5                          (repeatedly confirmed)
  - reported continuously for > 30 days unresolved

Downgrade one level if:
  - distance_confidence < 0.4  (area estimate unreliable)
```

Never emit a severity without the reason string. The UI displays it.
