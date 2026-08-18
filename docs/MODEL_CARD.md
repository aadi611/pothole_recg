# Model Card — pothole detector v1

## Identity

| | |
|---|---|
| **Weights** | `models/pothole-yolov8n.pt` |
| **SHA-256** | `2435fb8534976b2a25b0979017bafd8e4a1b2dc7c0da5a732ba817c06dec8537` |
| **model_version** | `yolo-2435fb853497` |
| **Architecture** | YOLOv8n (3.0M params, 8.1 GFLOPs) |
| **Trained** | 2026-08-19 |

`model_version` is derived from the hash, never hand-typed (FR-DET-06), and is
stamped on every `Detection` and every report. `config/detection.yaml` carries
the expected SHA-256, so a swapped weights file fails loudly at startup rather
than silently changing behaviour.

## Training

| | |
|---|---|
| Base | `yolov8n.pt` (COCO pretrained) |
| Dataset | Roboflow `pothole.v1i` — 563 train / 152 val, single class |
| Source | `huggingface.co/datasets/rupesh002/pothole-detection-dataset` |
| Licence | CC BY 4.0 |
| Epochs | 97 of 100 (early stopping, patience 20; best at **epoch 77**) |
| Image size | 640 |
| Batch | 16 |
| Optimizer | AdamW, lr 0.002 (auto-selected) |
| Hardware | RTX 3060 Laptop (6 GB) |
| Duration | **15.3 minutes** |

Augmentation is Ultralytics' built-in pipeline — mosaic, HSV, scale, translate,
horizontal flip, random erasing (see ADR-004 for why we do not stack a second
pipeline on top).

## Metrics (validation, 152 images / 244 instances)

| Metric | Value |
|---|---|
| **mAP@50** | **0.633** |
| **mAP@50-95** | **0.401** |
| Precision | 0.671 |
| Recall | 0.570 |

**mAP@50-95 (0.401) is the number that matters for this application**, more so
than for most detectors. It measures localisation tightness, and the *bottom
edge* of the box is the ground-contact point that planar distance estimation
projects (FR-PRX-02). A box that is approximately right but sits low reads as a
pothole closer than it is, and the driver gets warned at the wrong moment.
mAP@50 alone would flatter a model that localises loosely.

**Recall 0.570 means the model misses roughly 4 in 10 potholes** on this
validation set. That is the honest figure and it is not good enough for a
system anyone should rely on. It is adequate to prove the pipeline end to end;
it is not adequate for deployment.

## Inference performance

Measured on RTX 3060 Laptop, fp16, 640px, through the `Detector` interface:

| | |
|---|---|
| Inference | 4.7 ms (Ultralytics internal) / 25.7 ms (end-to-end incl. I/O) |
| Effective | ~39 FPS end-to-end |
| Target | ≥15 FPS (NFR-REL-06) — **met** |

## Known limitations

- **Domain gap — the big one.** Training images are predominantly non-Indian
  roads. Indian surfaces differ in texture, repair patterns, edge degradation,
  and unsealed shoulders. **Expect materially worse recall on Indian roads than
  the 0.570 above.** Retraining on the RDD2022 India split is the planned fix
  (`docs/DATASETS.md`, `scripts/fetch_dataset.py --source rdd2022`).
- **Small dataset.** 563 training images is few. The model has seen a narrow
  slice of lighting, weather, and surface conditions.
- **Water and shadow.** Flooded or deeply shadowed potholes are frequently
  missed or mis-sized. Since severity is derived from apparent area, a mis-sized
  box also produces a wrong severity.
- **Single class.** Only `pothole` is trained. `crack`, `surface_damage`,
  `manhole`, and `debris` exist in the class map but have no training data
  behind them (FR-DET-03).
- **No night validation.** The dataset is predominantly daylight. Night-time
  behaviour is unmeasured — not "acceptable", unmeasured.
- **No Indian-road validation.** Every number here comes from the same
  distribution as training. Real performance on the target roads is unknown
  until we test against `tests/fixtures/drive.mp4`.

These are limitations of the model, not of the pipeline. Report them; do not
work around them silently.

## Reproducing

```bash
python scripts/fetch_dataset.py --source roboflow-pothole
python scripts/train_detector.py \
    --data datasets/roboflow-pothole/data.yaml \
    --epochs 100 --batch 16 --device 0
```

Requires AC power — see ADR-005. On battery the discrete GPU drops off the PCI
bus and training silently falls back to CPU at roughly 35x the wall time.

## Next model

1. Retrain on the **RDD2022 India split** (13 GB, Pascal VOC → needs conversion)
2. Fold in our own labelled dashcam footage once recorded
3. Compare both with the FR-DET-08 A/B harness on the same fixture video, and
   record the deltas here rather than replacing this section
