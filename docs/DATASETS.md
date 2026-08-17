# Datasets

Candidate training data for the pothole detector (FR-DET-02). Salvaged and
expanded from the original project's `data/dataset_sources.txt`.

**Nothing here is committed to git.** Datasets are fetched with
`scripts/fetch_dataset.py` into `datasets/`, which is `.gitignore`d.

---

## Selected for Phase 1

### Roboflow Universe — pothole detection

| | |
|---|---|
| **Format** | YOLO (txt), already split train/valid/test |
| **Annotation** | Axis-aligned bounding boxes, single class `pothole` |
| **Why chosen** | No conversion step, no API token, immediate. Proves the pipeline end to end before we spend GPU time on a larger set. |
| **Licence** | Varies per dataset — **check the specific dataset page before use.** Most are CC BY 4.0. |
| **Caveat** | Predominantly Western road surfaces. Expect a domain gap on Indian roads. |

The plan is explicitly to **replace these weights later** (see below). The
`Detector` interface (FR-DET-01) makes that a config change, not a rewrite.

---

## Planned for Phase 1.5 / 2 — India-specific

### RDD2022 — Road Damage Dataset (`sekilab/RoadDamageDetector`)

| | |
|---|---|
| **Format** | Pascal VOC XML → needs conversion to YOLO |
| **Size** | ~47,000 images total across 6 countries |
| **India subset** | ~9,000 images of genuinely Indian road conditions |
| **Classes** | `D00` longitudinal crack, `D10` transverse crack, `D20` alligator crack, `D40` pothole — maps onto the FR-DET-03 class map |
| **Why it matters** | This is the domain-matched data. Indian roads differ from Western ones in surface texture, repair patterns, edge degradation, and the presence of unsealed shoulders. A model trained only on US/EU data will underperform here. |
| **Licence** | Open for research use — verify terms for any commercial deployment |
| **Cost** | Heavier training run; conversion script required |

**Recommendation:** train on Roboflow first to prove the pipeline, then retrain
on RDD2022's India subset and compare with the FR-DET-08 A/B harness. Record
both in `docs/MODEL_CARD.md`.

---

## Other candidates (not currently planned)

| Source | Format | Notes |
|---|---|---|
| Kaggle `andrewmvd/pothole-detection-dataset` | Pascal VOC | 665 images. Small. Needs a Kaggle API token. |
| Kaggle `sachinpatel21/pothole-image-dataset` | Folder-per-class | **Classification only — no boxes.** Unusable for detection. Listed because the original project referenced it. |
| Kaggle `chitholian/annotated-potholes-dataset` | Pascal VOC | ~665 images, overlaps andrewmvd. |
| Mendeley `5ty2wb6gvg` | Images | Licence and annotation format need verifying. |
| Roboflow Universe (search "pothole india") | YOLO | Several community sets with Indian footage. Quality varies — inspect before trusting. |

---

## Collecting our own data

Long term this is the highest-value option, and it is what makes the product
defensible: **nobody has a good labelled dataset of Indian road defects.**

Our own dashcam footage (starting with the `backend/tests/fixtures/drive.mp4`
regression clip) can be labelled and folded into training. Two rules:

1. **Redact before labelling, not after** (NFR-PRV-01) — faces and plates in
   training images are still personal data.
2. Record the collection date, route, camera, and mount geometry alongside the
   footage, so the images stay useful when calibration questions come up later.

---

## Annotation format note

The original project consumed a **folder-per-class** layout
(`data/processed/train/pothole/*.jpg`). That layout discards bounding boxes.
Several of the sources above are object-detection datasets whose box
annotations the old pipeline would have silently thrown away. Detection
training must preserve the boxes — see `backend/training/augment.py`, where
Albumentations is configured with `bbox_params` for exactly this reason.
