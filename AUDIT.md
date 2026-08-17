# AUDIT.md — RoadWatch Codebase Audit

**Date:** 2026-08-18
**Auditor:** Claude (agent)
**Repo:** `c:\Projects\PotHoleRecognition` — branch `main`, single commit `726a2f7 "first commit"`
**Status:** Phase 0, Step 1. **No implementation has begun. Awaiting review per `01_AGENT_BRIEF.md`.**

---

## Headline finding

The existing project is a **TensorFlow/Keras image-classification tutorial scaffold**, not a detection system. It answers "does this 224×224 image contain a pothole?" The spec requires "where is the pothole in this frame, how far ahead is it, and is it the same one I saw last frame?"

Those are different problems. Classification produces a single label per image with **no bounding box** — and without a bounding box there is no `BBox.bottom_center`, therefore no ground-contact point, therefore **no planar distance estimation, no tracking, no proximity gating**. FR-PRX, FR-TRK, and FR-EVD all rest on box geometry that this architecture cannot produce.

Compounding this: **there are no trained weights, no dataset, and no recorded metrics.** `models/`, `results/`, and `logs/` are empty directories. `data/` contains one text file of dataset URLs. The only artifact that has ever been produced by this repo is a stale notebook traceback.

The brief says trained weights and datasets are "almost always keep." Here there are none to keep. The reuse story is therefore much thinner than the brief anticipates — this is closer to a greenfield build than a migration, and the phase plan should be read with that in mind.

---

## A. What exists

### Directory tree (depth 3)

```
PotHoleRecognition/
├── CLAUDE.md                       # project memory (points at spec/ — see Q1)
├── README.md                       # docs for the OLD classification project
├── .claudeignore                   # context-exclusion rules — already RoadWatch-aware
├── config.json                     # Keras training hyperparameters
├── requirements.txt                # 13 pinned deps, TensorFlow stack
├── setup.py                        # interactive env bootstrapper (NOT a packaging setup.py)
├── data_downloader.py              # dataset fetch helper (Kaggle / URLs)
├── advanced_downloader.py          # second dataset fetch helper — overlaps the above
├── data_preprocessing.py           # split + augment + write processed JPEGs
├── train_model.py                  # Keras transfer-learning train/eval/predict
├── pothole_training_notebook.ipynb # notebook restating train_model.py; last run ERRORED
├── files/                          # THE SPEC (untracked) — 6 spec docs + README
├── files.zip                       # zip of the above
├── data/                           # EMPTY except dataset_sources.txt
│   ├── raw/{normal,pothole}/               — empty
│   └── processed/{train,val,test}/{...}/   — empty
├── models/                         # EMPTY — no weights
├── results/                        # EMPTY — no metrics
└── logs/                           # EMPTY — no training history
```

### Runtime & dependencies

| Item | Finding |
|---|---|
| Python | 3.11.9 on PATH. Spec targets 3.11 — **match, no upgrade needed.** |
| Virtualenv | **None on disk.** The notebook traceback references `.venv\Lib\site-packages`, so one existed and was deleted. Nothing is currently installed. |
| Dep manifest | `requirements.txt` only. No `pyproject.toml`, no lockfile. |
| Deps | tensorflow 2.13.0, opencv-python 4.8.1.78, numpy 1.24.3, pandas 2.0.3, matplotlib, seaborn, scikit-learn 1.3.0, Pillow, requests, tqdm, imutils, kaggle, albumentations 1.3.1 |
| **Installability** | ⚠️ **`tensorflow==2.13.0` has no cp311 Windows wheel and is unbuildable here. `requirements.txt` cannot be installed as pinned on this machine.** The stack is not merely unused — it is currently non-functional. |
| Git | 1 commit, 10 files, 1893 insertions. No `.gitignore` (so `data/`, `models/`, `.venv` are all unprotected). No tags/branches. |
| CI / Docker / tests | **None.** No `.github/`, no `Dockerfile`, no `docker-compose.yml`, no `tests/`. |

### Entry points

All four are **interactive `input()` menus** — no CLI flags, no importable/scriptable path:

```
python setup.py              → 1 full setup / 2 demo / 3 dirs / 4 pip install
python data_downloader.py    → 1 sample structure / 2 info / 3 kaggle
python advanced_downloader.py→ 1-5 various download methods
python data_preprocessing.py → 1 synthetic / 2 process raw / 3 organize
python train_model.py        → 1 train / 2 evaluate / 3 predict image
```

This blocks automation, CI, and `NFR-DEV-02` (one documented non-interactive command). Every entry point needs rewriting regardless of which direction we go.

### Is there a trained model?

**No.** This is the most consequential finding.

- `models/` — empty. No `.h5`, `.pt`, `.onnx`, anything.
- Architecture *specified* in code: MobileNetV2 / ResNet50 / EfficientNetB0 backbone, ImageNet weights, frozen base, custom head (GAP → Dropout .3 → Dense 128 → BN → Dropout .5 → Dense 64 → Dropout .3 → Dense 2 softmax).
- Classes: `['normal', 'pothole']` — **classification, not detection.**
- Input size: 224×224.
- mAP: **not applicable** — a classifier has no mAP. No accuracy recorded either; `results/` and `logs/` are empty.

**FR-DET-02 says "use the existing project weights if usable; otherwise fine-tune." There are no weights. We must train from scratch on a detection dataset.**

### Is there a dataset?

**No.** `data/raw/{pothole,normal}/` and all nine `data/processed/` leaves are empty. The only content is `data/dataset_sources.txt` — a plain-text list of suggested sources (Kaggle `andrewmvd/pothole-detection-dataset`, Roboflow Universe, `sekilab/RoadDamageDetector`, Mendeley).

Annotation format: none present. Note the sources listed are **object-detection** datasets (bounding boxes) while the preprocessing code consumes **folder-per-class** classification layout — the existing pipeline would discard the box annotations it downloads.

### Existing backend / DB / frontend?

**None of the three.** No FastAPI, no SQLAlchemy, no Alembic, no Postgres/PostGIS, no SQLite, no React, no `package.json`, no `web/`. The README's "Local Database: SQLite database for storing pothole counts and locations" is **aspirational — no such code exists anywhere in the repo.** Same for its "Real-time Detection" claim.

### Tests, CI, Docker, config

- Tests: **none.**
- CI: **none.**
- Docker: **none.**
- Config: `config.json` (Keras hyperparameters only — nothing about cameras, thresholds, or authorities). `.claudeignore` exists and is already written for the RoadWatch structure.

---

## B. What is worth keeping

| Asset | Verdict | Reason |
|---|---|---|
| `.claudeignore` | **Keep as-is** | Already written for the target structure (`evidence/`, `web/dist/`, `*.pt`). Only needs `spec/` handling. |
| Python 3.11.9 runtime | **Keep as-is** | Matches the spec target exactly. |
| `data/dataset_sources.txt` | **Keep and refactor** | The URL list is genuinely useful research. Move to `docs/DATASETS.md`, extend with licence and annotation-format columns per source. |
| `data_preprocessing.py` — augmentation pipeline | **Keep and refactor** | The Albumentations recipe (flip, shift-scale-rotate, brightness/contrast, HSV, noise, blur) is sound and road-appropriate. Port to a YOLO training config with `bbox_params` so boxes transform with the image. ~30 useful lines of ~273. |
| `data_preprocessing.py` — everything else | **Replace** | Folder-per-class splitting writes **pre-normalised, pre-augmented JPEGs to disk** — augmentation is baked in permanently rather than applied per-epoch, and ImageNet-normalised floats are cast to uint8 for saving, corrupting the pixel data. Wrong for detection and wrong on its own terms. |
| `train_model.py` | **Replace** | Classification-only. Cannot emit boxes. Superseded by Ultralytics YOLO training behind `detection/`. |
| `pothole_training_notebook.ipynb` | **Delete** | Duplicates `train_model.py`, adds nothing, has a typo (`plt0`) on line 7 of its first code cell, and its saved state is a failed run. Superseded entirely. |
| `data_downloader.py` | **Replace** | Merge with the below into one non-interactive `scripts/fetch_dataset.py`. |
| `advanced_downloader.py` | **Replace** | Substantially overlapping duplicate of `data_downloader.py` (both implement `download_kaggle_dataset`). Its `download_roboflow_sample()` uses fabricated URLs (`https://source.roboflow.com/pothole1.jpg` — not a real endpoint) that fail silently through a `try/except` returning False. |
| `setup.py` | **Delete** | Interactive menu that pip-installs and mkdirs. Superseded by `pyproject.toml` + Docker. Its filename actively misleads — it is not a packaging file, and leaving it risks a build tool treating it as one. |
| `config.json` | **Delete** | Keras hyperparameters for a model we are not building. Superseded by `config/*.yaml` with Pydantic validators. |
| `requirements.txt` | **Delete** | Pins an uninstallable TensorFlow. Superseded by `backend/pyproject.toml`. |
| `README.md` | **Replace** | Documents the old project, including two features that do not exist (SQLite DB, real-time detection). Rewrite per NFR-DEV-05. |
| `models/`, `results/`, `logs/` | **Delete** | Empty. Nothing to preserve. |
| `data/raw/`, `data/processed/` | **Delete** | Empty. The new layout puts datasets under `datasets/` (already `.claudeignore`d). |

### Deletion manifest (explicit, per brief rule 2)

Every deletion, with the reason it is dead or superseded:

1. `pothole_training_notebook.ipynb` — duplicate of `train_model.py`, errored state, typo'd.
2. `setup.py` — superseded by `pyproject.toml`/Docker; misleading filename.
3. `config.json` — hyperparameters for an abandoned architecture.
4. `requirements.txt` — superseded by `pyproject.toml`; pins an uninstallable TF.
5. `train_model.py` — classification architecture, cannot produce boxes.
6. `data_downloader.py` + `advanced_downloader.py` — merged into one script; contain dead/fabricated URLs.
7. `data_preprocessing.py` — augmentation recipe extracted first, rest replaced.
8. `models/`, `results/`, `logs/`, `data/raw/`, `data/processed/` — empty directories.
9. `files.zip` — redundant with the extracted `files/`.

Nothing on this list contains trained weights, datasets, or metrics — none exist. **Recommendation: do the deletions in one commit separate from any new code**, so the removal is trivially reviewable and revertible in git.

---

## C. Gap analysis

Every FR ID in `04_REQUIREMENTS.md`, plus all NFRs in `06_NONFUNCTIONAL.md`.

**Legend:** Missing = nothing exists · Partial = something reusable exists · N/A-Later = `[LATER]`, out of MVP scope

### FR-CAM — Camera & frame pipeline

| ID | Status | Notes |
|---|---|---|
| FR-CAM-01 | Missing | No frame source abstraction. OpenCV is a dependency but only used for `imread`/`imwrite` on still files. |
| FR-CAM-02 | Missing | No threading, no buffer, no frame dropping anywhere. |
| FR-CAM-03 | Missing | No sampling strategy. |
| FR-CAM-04 | Missing | No disconnect/corrupt-frame handling. `load_and_process_image` returns `None` on a bad read but callers only skip. |
| FR-CAM-05 | Missing | No rolling buffer. |
| FR-CAM-06 | Missing | No session metrics of any kind. |

### FR-DET — Detection model layer

| ID | Status | Notes |
|---|---|---|
| FR-DET-01 | Missing | No `Detector` protocol. `train_model.py` imports TensorFlow directly at module top level — the exact boundary violation the spec forbids. |
| FR-DET-02 | **Missing** | **No weights exist.** Architecture in code is a classifier and cannot yield mAP. Requires training a YOLO detector on a boxed dataset from zero. **This is the single largest work item in Phase 1.** |
| FR-DET-03 | Missing | Class list is a hardcoded `['normal','pothole']` in three separate places (`train_model.py:31`, `config.json`, notebook `Config`). No config-driven class map. |
| FR-DET-04 | Missing | No `Detection` model. Current output is `{'class','confidence','is_pothole','probabilities'}` — no bbox, no frame_id, no model_version. |
| FR-DET-05 | Partial | `config.json` has img_size/batch/lr, but no confidence threshold, no NMS/IoU, no device selection. Thresholds don't exist because a classifier has none. |
| FR-DET-06 | Missing | No model versioning at all. Weights would be timestamp-named (`best_pothole_model_{ts}.h5`) — not hash-tied. |
| FR-DET-07 | N/A-Later | ONNX export. |
| FR-DET-08 | N/A-Later | A/B harness. |

### FR-TRK — Tracking & state machine

| ID | Status | Notes |
|---|---|---|
| FR-TRK-01 → 06 | **Missing** | No tracking of any kind. No track IDs, no state machine, no confirmation counting, no LOST handling, no per-track capture guard, no peak-confidence bookkeeping. Nothing in the repo processes more than one image at a time. |

### FR-PRX — Proximity & distance

| ID | Status | Notes |
|---|---|---|
| FR-PRX-01 → 08 | **Missing** | No distance estimation, no camera calibration, no thresholds, no time-to-impact, no smoothing, no confidence/assumptions, no calibration utility. **Blocked on FR-DET-04** — planar projection needs `bbox.bottom_center`, which a classifier cannot provide. |
| FR-PRX-09, 10 | N/A-Later | Monodepth, stereo. |
| FR-PRX-11 | Missing | `docs/DISTANCE_LIMITATIONS.md` not written. |

### FR-ALT — Driver alerting

| ID | Status | Notes |
|---|---|---|
| FR-ALT-01 → 06 | **Missing** | No alerting subsystem. No sinks, no cooldowns, no speed suppression, no latency budget. |

### FR-EVD — Evidence capture

| ID | Status | Notes |
|---|---|---|
| FR-EVD-01 → 08 | **Missing** | No capture trigger, no best-frame selection, no SHA-256, no GPS binding, no local-first write, no redaction, no retention policy. `cv2.imwrite` in `data_preprocessing.py` is dataset prep, unrelated to evidence. |

### FR-GEO — GPS & geospatial

| ID | Status | Notes |
|---|---|---|
| FR-GEO-01 → 08 | **Missing** | No GPS anywhere. No sources, no replay, no geocoding, no PostGIS. The README claims location storage; no such code exists. |

### FR-RPT — Reports

| ID | Status | Notes |
|---|---|---|
| FR-RPT-01 → 05 | **Missing** | No report entity, no ID minting, no severity, no status lifecycle, no audit trail. |
| FR-RPT-06 | N/A-Later | Bulk operator actions. |

### FR-AUTH — Authority routing

| ID | Status | Notes |
|---|---|---|
| FR-AUTH-01 → 07, 10 | **Missing** | No authority concept whatsoever. Requires the full YAML schema, router, and submitters from zero. |
| FR-AUTH-08, 09 | N/A-Later | Generic HTTP submitter, aggregate submission. |

### FR-DUP — Deduplication

| ID | Status | Notes |
|---|---|---|
| FR-DUP-01 → 05 | **Missing** | No dedup, no spatial index, no aggregation, no centroid math. |
| FR-DUP-06 → 09 | N/A-Later | Image similarity, way snapping, merge tools, auto-resolution. |

### FR-API — Backend

| ID | Status | Notes |
|---|---|---|
| FR-API-01 → 08 | **Missing** | No backend at all. No FastAPI, no auth, no pagination, no OpenAPI, no health endpoints. |

### FR-SYN — Offline queue

| ID | Status | Notes |
|---|---|---|
| FR-SYN-01 → 06 | **Missing** | No queue, no SQLite, no uploader, no storage guard, no sync status. |

### FR-UI — Dashboard

| ID | Status | Notes |
|---|---|---|
| FR-UI-01 → 16 | **Missing** | No frontend. No `package.json`, no `web/`. Existing "UI" is matplotlib `plt.show()` windows in `train_model.py` — desktop-blocking, unusable in a vehicle and unusable headless. |
| FR-UI-17, 18 | N/A-Later | Authority performance, worst-roads. |

### NFR — Non-functional

| ID | Status | Notes |
|---|---|---|
| NFR-PRV-01 → 07 | Missing | No redaction, no retention, no deletion path, no `docs/PRIVACY.md`. |
| NFR-SEC-01 → 08 | Missing | No auth, no secrets management, no `.env.example`, no rate limiting. Note: no secrets are currently committed — clean history to build on. |
| NFR-REL-01 → 07 | Missing | Single-threaded blocking scripts throughout. No timeouts, no bounded queues, no shutdown handling, no degradation matrix. |
| NFR-OBS-01 → 05 | Missing | `print()` only — 60+ calls across the five scripts. No structlog, no metrics, no correlation IDs. ⚠️ `data_preprocessing.py:88` and `advanced_downloader.py:41` are broad `except Exception` handlers that print and swallow — the pattern NFR-OBS-05 forbids. |
| NFR-TST-01 → 07 | Missing | Zero tests. **No `tests/fixtures/drive.mp4` + `drive.gpx`** — the standing regression input named in CLAUDE.md and NFR-TST-02 **does not exist and must be sourced.** See Q4. |
| NFR-DEV-01 → 07 | Missing | No Docker, no CI, no Alembic, no ADR log. All entry points interactive (blocks NFR-DEV-02). |

### Summary

| | Count |
|---|---|
| Requirements fully met | **0** |
| Partial (something reusable) | **2** (FR-DET-05, the augmentation recipe under FR-DET-02) |
| Missing (MVP scope) | **~95** |
| N/A-Later | ~18 |

---

## D. Migration plan

### Reality check

With no weights and no dataset, "migration" means moving **one augmentation recipe and one URL list**. Everything else in `02_ARCHITECTURE.md` is new construction. I want that stated plainly now rather than discovered in Phase 1 — it materially changes the effort estimate for Phase 1, which the phase plan frames as "use existing weights" work.

### File-by-file

| From | To | Action |
|---|---|---|
| `data_preprocessing.py:32-51` (`setup_augmentations`) | `backend/training/augment.py` | Port the Albumentations recipe; add `A.BboxParams(format='yolo')` so boxes transform with the image. Drop `A.Normalize` (YOLO handles it) and `RandomRotate90` (a 90°-rotated dashcam frame is not a real road view and teaches the model nothing useful). |
| `data/dataset_sources.txt` | `docs/DATASETS.md` | Rewrite as a table: source, licence, annotation format, image count, suitability. |
| `data_downloader.py` + `advanced_downloader.py` | `scripts/fetch_dataset.py` | Merge into one **non-interactive, argparse** script. Keep the Kaggle path and the tqdm progress helper; delete the fabricated Roboflow URLs and the interactive menus. |
| `.claudeignore` | (in place) | Add `spec/` handling; keep the rest. |
| `files/*.md` | `spec/*.md` | **`git mv`** — see Q1. Currently untracked, which means the spec is one `git clean` from gone. |
| `README.md` | `README.md` | Full rewrite per NFR-DEV-05. |
| everything else | — | Delete per §B manifest. |

### Target skeleton (Phase 0 deliverable)

Build `02_ARCHITECTURE.md`'s layout: `config/` (detection, proximity, alerts, authorities/, logging — each with a Pydantic validator), `backend/roadwatch/` (all 18 modules), `backend/tests/{unit,integration,fixtures}`, `web/`, plus `pyproject.toml`, `package.json`, `.env.example`, `.gitignore`, `docker-compose.yml`, Alembic init, and CI running ruff + mypy + pytest + the module-boundary grep.

### Cannot be migrated cleanly

1. **The classifier architecture → detection.** Not a refactor. Different task, different output shape, different dataset annotation format. Full replacement.
2. **The `data/processed/` on-disk augmentation model → per-epoch augmentation.** Writing augmented images to disk is architecturally wrong for training. Replaced, not ported.
3. **Interactive `input()` menus → CLI.** Every entry point is rewritten. No mechanical translation.
4. **`requirements.txt` → `pyproject.toml`.** The TF stack is being dropped entirely; only `opencv-python`, `numpy`, and `albumentations` carry over, and all at newer versions. Not a port so much as a fresh dependency set.

### Sequencing recommendation

1. Commit A — `git mv files/ spec/`, add `.gitignore`. (Protects the spec first.)
2. Commit B — all §B deletions, nothing else. Reviewable in isolation.
3. Commit C — port the two salvageable assets.
4. Commit D..N — Phase 0 skeleton.

---

## E. Open questions for the human

**These block Phase 1. Q1–Q4 need answers before I proceed; Q5–Q8 can be deferred to their phase but earlier is better.**

**Q1 — `files/` vs `spec/` (blocking, and quick).**
`CLAUDE.md` points at `spec/`; the documents are in `files/`, **untracked by git** and duplicated in `files.zip`. So the governing specification is currently unversioned and one `git clean -fd` from deletion. Proposal: `git mv files/ spec/`, commit, delete `files.zip`. Confirm?

**Q2 — Detection dataset (blocking).**
There are no weights and no data, so FR-DET-02 means training a detector from zero. Which do you want?
- **(a)** Roboflow Universe pothole dataset — YOLO format already, fastest path, licences vary per dataset.
- **(b)** Kaggle `andrewmvd/pothole-detection-dataset` — 665 images, Pascal VOC boxes, needs conversion. Requires a Kaggle API token — do you have one?
- **(c)** `sekilab/RoadDamageDetector` (RDD) — much larger, multi-class road damage, maps well onto the FR-DET-03 class list, heavier to train.
- **(d)** You supply your own annotated data.

**My recommendation: (a) to unblock the pipeline in Phase 1, then (c) to improve accuracy once the plumbing is proven.** Getting the interfaces right matters more early than mAP does; swapping weights later is a config change by design.

**Q3 — Training hardware (blocking for Q2).**
Is there a CUDA GPU available? I found no GPU-specific config and no evidence of a successful training run. If CPU-only, YOLOv8n on ~600 images is roughly an overnight job — workable but it sets the Phase 1 pace, and I would want to know before committing to option (c), which is materially heavier.

**Q4 — The regression fixture (blocking for Phase 1 exit criteria).**
`CLAUDE.md` and NFR-TST-02 both treat `tests/fixtures/drive.mp4` + `drive.gpx` as the standing regression input for every phase. **Neither file exists.** Every phase's exit criteria depend on it. Options: you record a phone dashcam clip with a GPS logger (best — real road, real potholes, and it becomes ground truth for calibration); or I source a CC-licensed dashcam clip and synthesise a plausible GPX; or I generate a fully synthetic clip (weakest — proves the plumbing only). **Which?** If you can record 2 minutes of real driving, that single asset is worth more to this project than anything else on this list.

**Q5 — Deployment target.** The brief asks. Laptop + webcam, Jetson, or Android phone? This decides the FPS budget (NFR-REL-06 wants ≥15 FPS), whether ONNX/TFLite gets pulled earlier than Phase 7, and model size. I will assume **laptop + USB/phone camera for Phases 1–4** unless told otherwise.

**Q6 — PostGIS or SQLite?** The spec fixes PostgreSQL 16 + PostGIS, and FR-DUP/FR-GEO-08 genuinely need spatial indexes. That requires Docker on this machine. Confirm Docker is available — if not, raise it now and I will propose a SQLite + SpatiaLite fallback as an ADR rather than discovering the blocker mid-Phase-4.

**Q7 — First authority to seed (Phase 5).** `03_DATA_CONTRACTS.md` uses Noida / Gautam Buddha Nagar as its worked example. Is that the real target city? It determines which jurisdictions I research and honestly mark `verified_on`. I will not invent authority contact data — per FR-AUTH-05 and the ethical guardrails, a fabricated contact point is worse than none.

**Q8 — Camera calibration (Phase 2).** FR-PRX-02 needs mount height, pitch, and FOV. Will you be able to measure these on a real vehicle, or should Phase 2 lean on the FR-PRX-03 bbox-scale fallback with lowered confidence until real numbers exist? This is the difference between the ±25% accuracy target in Phase 2's exit criteria being testable or not.

---

## Assessment

The spec is unusually well-specified — contracts, boundaries, and ethical rules are all pinned down, and the ethical guardrails around never faking a submission are the right ones to hold. The gap is that the starting point offers far less than the brief assumes. Phase 1 was framed as "wire up existing weights"; it is actually "acquire a dataset, train a detector, then wire it up."

That is a schedule concern, not a design one. Nothing in the spec is unachievable and I am not proposing any deviation from it. But I would rather flag the mismatch now than let Phase 1 quietly absorb a training project.

**Per `01_AGENT_BRIEF.md` step 1: stopping here for review. No files have been created, modified, or deleted other than this audit.**
