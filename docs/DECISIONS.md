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
