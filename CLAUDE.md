# CLAUDE.md

Project memory for Claude Code. Read this before every task.

## What this is

**RoadWatch** — a real-time AI pothole detection and civic reporting system. A camera on a moving vehicle detects potholes, warns the driver before impact, auto-captures geotagged evidence, and routes reports to the responsible road authority.

**Stack:** Python 3.11 / FastAPI / PostGIS backend. React + TypeScript + Vite + Tailwind + MapLibre frontend. Ultralytics YOLO behind a swappable interface.

## Specification — read before implementing

The full spec lives in `spec/`. Load only what the current task needs:

| File | Contents |
|---|---|
| `spec/01_AGENT_BRIEF.md` | Rules of engagement, codebase audit protocol |
| `spec/02_ARCHITECTURE.md` | Stack, module map, repo layout, config schema |
| `spec/03_DATA_CONTRACTS.md` | Pydantic models, interfaces, DB tables, API routes |
| `spec/04_REQUIREMENTS.md` | ID-tagged functional requirements (FR-*) |
| `spec/05_BUILD_PHASES.md` | Phase plan and exit criteria |
| `spec/06_NONFUNCTIONAL.md` | Privacy, security, reliability, testing |

Current phase and completed work: see `PROGRESS.md`.

## Non-negotiable rules

1. **Never report a status that did not happen.** A report only becomes `SENT_TO_AUTHORITY` when a channel actually accepted it. Otherwise `PENDING_MANUAL`. No mock submissions that return success.
2. **Never guess an authority.** No jurisdiction match → `PENDING_MANUAL` with a stated reason.
3. **No model-library imports outside `backend/roadwatch/detection/`.** No `ultralytics`, no `torch`. Everything uses the `Detector` protocol.
4. **No SQLAlchemy imports outside `backend/roadwatch/db/`.**
5. **No hard-coded thresholds.** Distances, cooldowns, confidences, model paths → YAML config, validated by Pydantic at startup.
6. **No bare `except: pass`.** Errors carry context.
7. **Never lose a confirmed report to a network failure.** Local write first, then queue, then sync.
8. **Alerts are gated on estimated distance or time-to-impact**, never on mere presence in a frame.
9. **Redact faces and plates before writing evidence to disk**, not before publishing.
10. **Alembic migration for every schema change.** `create_all()` only in tests.

## Commands

```bash
# Backend
cd backend && pip install -e ".[dev]"
uvicorn roadwatch.server_app:app --reload      # API server
python -m roadwatch.edge_app --source tests/fixtures/drive.mp4 --gpx tests/fixtures/drive.gpx --display
alembic upgrade head
alembic revision --autogenerate -m "description"

# Tests
pytest                                          # all
pytest tests/unit -q                            # fast loop
ruff check . && mypy roadwatch/core roadwatch/detection roadwatch/distance

# Frontend
cd web && npm install && npm run dev
npm run build && npm run test
npx openapi-typescript http://localhost:8000/openapi.json -o src/api/schema.ts

# Full stack
docker compose up
```

## Conventions

- Commit format: `type(scope): summary [FR-XXX-NN]` — always cite the requirement ID.
- Update `PROGRESS.md` at the end of every session with what shipped and what is blocked.
- Record any deviation from the spec in `docs/DECISIONS.md` as a short ADR (context / decision / consequence).
- Cross-module payloads are Pydantic models from `spec/03_DATA_CONTRACTS.md`. No bare dicts across boundaries.
- Pure logic modules (`distance/`, `tracking/`, `dedup/`) do no I/O and must be unit-testable with synthetic data.
- Python: ruff formatting, type hints on all public functions.
- Frontend: never hand-write API types — generate them from the OpenAPI schema.

## Testing rule

`tests/fixtures/drive.mp4` + `drive.gpx` is the standing regression input. Every phase must still pass against it. Nothing requires a real vehicle or camera to develop or review.

## When you are unsure

Ask. Do not silently deviate from the spec, and do not implement something you know cannot work in order to satisfy a requirement literally. Raise the conflict and propose an alternative.
