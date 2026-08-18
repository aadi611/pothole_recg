# Architecture Decision Records

Short ADRs: context, decision, consequence. One per deviation from the spec or
per decision a future reader would otherwise have to reconstruct (NFR-DEV-07).

---

## ADR-001 — Rebuild detection rather than migrate the classifier

**Date:** 2026-08-18
**Status:** Accepted

**Context.** The repository contained a TensorFlow/Keras transfer-learning
image classifier (MobileNetV2 backbone, `normal` vs `pothole`). `AUDIT.md`
found no trained weights, no dataset, and no recorded metrics — `models/`,
`results/`, and `logs/` were empty. Additionally `requirements.txt` pinned
`tensorflow==2.13.0`, which has no cp311 Windows wheel and cannot be installed
on the development machine.

**Decision.** Delete the classifier and build detection on Ultralytics YOLO
behind the `Detector` interface.

**Consequence.** A classifier emits one label per image with no bounding box.
Without a box there is no `BBox.bottom_center`, therefore no ground-contact
point, therefore no planar distance estimation (FR-PRX-02), no tracking
(FR-TRK), and no evidence capture (FR-EVD). The requirement that alerts be
gated on estimated distance rather than mere presence in frame (FR-PRX-01) is
unimplementable on a classification architecture. Nothing was lost in the
deletion because no weights or data existed. The cost is that Phase 1 now
includes training a detector from zero, which the phase plan had framed as
"use existing weights" work — flagged in `AUDIT.md` as a schedule risk, not a
design change.

---

## ADR-002 — Google/Apple Maps integration is a partnership, not an API

**Date:** 2026-08-18
**Status:** Accepted

**Context.** A stated product goal is surfacing pothole data in Google Maps
and Apple Maps.

**Decision.** Do not architect toward a direct Maps write API, because none
exists. Build toward data quality good enough to qualify for a civic data
partnership, and ship our own MapLibre map as the primary surface.

**Consequence.** Neither Google Maps nor Apple Maps accepts third-party hazard
data through a public API. Google's road-incident data originates from Waze
and its own fleet. The realistic channel is the **Waze for Cities** programme,
which is a partnership granted to government and civic entities, not an
integration that can be coded against — and which expects verified,
deduplicated, evidence-backed reports.

That requirement is already what the spec describes: spatial dedup with
distinct-device confidence (FR-DUP), authority routing with a stated match
reason (FR-AUTH-02), honest report status (FR-AUTH-05), and redacted evidence
images (NFR-PRV-01). So this is not a detour — the work that makes the data
partnership-grade is the work already planned. What changes is expectation
setting: no Phase 7 task should assume a Maps API call, and export formats
(GeoJSON, CSV per FR-RPT-06) matter more than they otherwise would, since
handover to an authority is the actual distribution mechanism.

**Revisit if** Google, Apple, or Waze publish a public hazard-ingestion API,
or if a partnership is secured — at which point a new `Submitter`
implementation (FR-AUTH-08) is the natural place for it, requiring no change
to the pipeline.

---

## ADR-004 — Drop the ported Albumentations pipeline; use Ultralytics' built-in augmentation

**Date:** 2026-08-18
**Status:** Accepted
**Supersedes:** the augmentation salvage described in `AUDIT.md` §B and §D

**Context.** `AUDIT.md` identified the old project's Albumentations recipe as
one of only two reusable assets, and it was ported to
`backend/roadwatch/training/augment.py` with `bbox_params` added so boxes
travel with the image.

Two things then emerged:

1. `albumentations >= 1.4.8` depends on `stringzilla`, which has no cp311
   Windows wheel and fails to build without a C toolchain. Only 1.4.0 installs
   cleanly on this machine.
2. Ultralytics applies its own augmentation per-epoch and bbox-aware, enabled
   by default: `hsv_h/s/v`, `translate 0.1`, `scale 0.5`, `fliplr 0.5`,
   `mosaic 1.0`, `erasing 0.4`, plus randaugment. This covers the ported
   recipe almost entirely.

**Decision.** Delete `augment.py`, drop the `albumentations` dependency, and
use Ultralytics' pipeline, tuning it through `scripts/train_detector.py`
arguments where the defaults are wrong for dashcam footage.

**Consequence.** Applying both pipelines would augment twice — a
scale-jittered, HSV-shifted image being scale-jittered and HSV-shifted again
— which degrades training rather than improving it. Keeping the module while
*not* using it would leave dead code plus a dependency pinned four minor
versions back for no benefit.

The cost is losing the road-specific transforms Ultralytics does not offer:
`RandomShadow`, `RandomSunFlare`, `RandomFog`, and explicit motion blur. These
matter for Indian conditions (dust haze, harsh sun, wet tarmac). If validation
shows weakness there, the right fix is a custom Ultralytics dataset transform
rather than a parallel pipeline — recorded here so the reasoning is not lost.

This means the salvage from the original project is now `docs/DATASETS.md`
alone. The audit's conclusion stands and is strengthened: there was
essentially nothing reusable in the old codebase.

---

## ADR-003 — `str, Enum` retained over `StrEnum`

**Date:** 2026-08-18
**Status:** Accepted

**Context.** Ruff's `UP042` flags classes inheriting from both `str` and
`enum.Enum`, suggesting `enum.StrEnum` (3.11+).

**Decision.** Keep `str, Enum` and disable `UP042`.

**Consequence.** `spec/03_DATA_CONTRACTS.md` specifies this pattern, and the
contracts rely on enum members comparing equal to their string values and
serialising to plain JSON strings across the API boundary. Changing it would
be a silent deviation from the written contract for a stylistic gain. The
ignore is documented inline in `backend/pyproject.toml`.

---

## ADR-005 — Training requires AC power on this laptop

**Date:** 2026-08-19
**Status:** Accepted

**Context.** CUDA availability proved unstable during development:
`torch.cuda.is_available()` returned True early in a session and False later,
with no code change. `nvidia-smi` began failing with a permissions error.

Diagnosis via `Get-PnpDevice -Class Display` showed the RTX 3060 reporting
**`CM_PROB_PHANTOM`** — Windows held a stale record of the device, but it was
not present on the PCI bus. `Get-CimInstance Win32_Battery` showed
`BatteryStatus 1` (discharging). The NVIDIA driver files were intact (29
packages in DriverStore, `nvapi64.dll` present), and `nvlddmkm` was Stopped.

The cause is hybrid-graphics power management: on battery, the laptop powers
the discrete GPU entirely off the bus. Plugging in restored it immediately
(`BatteryStatus 2`, `CM_PROB_NONE`, `nvidia-smi` responding).

**Decision.** Treat AC power as a precondition for training and for any
GPU-backed benchmark. Do not attempt to work around it in code.

**Consequence.** This is an environment condition, not a defect, so tests that
need CUDA skip rather than fail — see `_cuda_available()` in
`tests/unit/test_detector.py`. Anyone reproducing a training run or an
inference benchmark must be on mains power, or their numbers will silently
come from the CPU.

Measured difference on this machine (yolov8n, 640px, 563 training images):
CPU ~185 s/epoch, GPU ~5 s/epoch — roughly 35x. A run that takes ten minutes
plugged in takes four hours on battery, which is enough of a gap that a
misconfigured run is obvious from the clock alone.

---
