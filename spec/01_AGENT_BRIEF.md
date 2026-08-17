# 01 — Agent Brief (read first)

## Mission

Evolve an existing pothole-detection project into a production-oriented, real-time road monitoring and civic reporting platform. This is a **product prototype**, not a demo notebook.

---

## Hard rules

1. **Audit before you build.** Complete the codebase audit below and write `AUDIT.md` before changing a single file. Get the audit reviewed before Phase 1 implementation begins.
2. **Do not rewrite from scratch.** Reuse trained weights, datasets, working inference code, useful utilities. Restructure weak parts; delete only what is genuinely dead or superseded, and list every deletion in `AUDIT.md`.
3. **Never fake success.** If an authority API does not exist, the report status is `PENDING_MANUAL`, not `SUBMITTED`. If distance is a rough estimate, the API says so via a confidence field. No mock data paths that can silently reach production. No placeholder submission that returns 200 and does nothing.
4. **No hard-coded thresholds.** Distances, confidences, cooldowns, model paths, authority mappings — all config-driven. See `config/` in `02_ARCHITECTURE.md`.
5. **The model layer is swappable.** Nothing outside `detection/` may import `ultralytics`, `torch`, or any model-specific symbol. Everything talks to the `Detector` interface in `03_DATA_CONTRACTS.md`.
6. **Build incrementally by phase.** Do not start Phase N+1 until Phase N's exit criteria pass. Phases are in `05_BUILD_PHASES.md`.
7. **Every module ships with tests.** A phase is not complete without them. See `06_NONFUNCTIONAL.md`.
8. **State limitations in code and in docs.** Monocular distance estimation is approximate; write down the assumptions where the code makes them, and surface the uncertainty in the UI.

---

## Step 1 — Codebase audit (do this now)

Produce `AUDIT.md` answering exactly these questions:

### A. What exists
- Directory tree (depth 3), with a one-line purpose per top-level item.
- Language/runtime versions, dependency files, lockfiles.
- Entry points: how is it currently run?
- Is there a trained model? Which architecture, which weights file, what classes, what input size, what mAP if recorded?
- Is there a dataset, annotation format, or training script?
- Is there any existing backend, DB, or frontend? What framework and state?
- Existing tests, CI, Docker, config files.

### B. What is worth keeping
For each asset: **keep as-is / keep and refactor / replace / delete**, with a one-line reason. Trained weights and datasets are almost always *keep*.

### C. Gap analysis
Map the current state against `04_REQUIREMENTS.md`. Produce a table:

| Requirement ID | Status | Notes |
|---|---|---|
| FR-DET-01 | Partial | YOLOv8n weights exist, no abstraction layer |
| FR-PRX-01 | Missing | No distance estimation at all |

### D. Migration plan
How existing code moves into the target structure in `02_ARCHITECTURE.md`. Name the files. Flag anything that cannot be migrated cleanly and why.

### E. Open questions for the human
Anything you should not decide alone. Examples: target deployment hardware (laptop? Jetson? Android phone?), whether a real GPS source is available for testing, which city's authority data to seed first, whether PostGIS is acceptable or SQLite is required.

**Stop after `AUDIT.md` and wait for review.**

---

## Working agreement

- Small, reviewable commits. One requirement group per commit where possible.
- Update `PROGRESS.md` at the end of every session.
- When a spec requirement conflicts with reality on the ground (hardware limits, missing data, a library that does not do what the spec assumes), **raise it and propose an alternative** — do not silently deviate and do not silently implement something that cannot work.
- Prefer boring, well-supported libraries over clever ones. This has to run on a phone eventually.
- Write down every assumption you make about camera height, focal length, road geometry, and vehicle speed in `docs/ASSUMPTIONS.md`.

---

## Definition of "production-oriented" for this project

Not: enterprise scale, five nines, k8s.
Yes: it survives a two-hour drive with no internet, a camera unplug, a GPS dropout, a corrupt frame, a full disk, and an authority endpoint that returns 500 — and afterwards you can reconstruct exactly what happened from the logs and the local queue.
