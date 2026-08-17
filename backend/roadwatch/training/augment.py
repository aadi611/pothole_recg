"""Augmentation pipeline for detector training.

Ported from the original project's ``data_preprocessing.py`` with three
substantive changes:

1. **Boxes travel with the image.** The original was a classification pipeline
   and dropped annotations. Every transform here is bbox-aware via
   ``A.BboxParams``; a geometric transform that moved pixels without moving the
   boxes would silently corrupt the labels.
2. **``A.Normalize`` removed.** Ultralytics normalises internally. Applying it
   here would double-normalise.
3. **``A.RandomRotate90`` removed.** A 90-degree-rotated dashcam frame is not a
   road scene. It taught the classifier nothing and would actively mislead a
   detector that must learn perspective — the ground plane is always at the
   bottom of the frame, and that regularity is signal we want to preserve, not
   augment away.

The original also wrote augmented JPEGs to disk once, permanently baking in a
single random draw per image. Augmentation belongs at training time, per epoch,
so every epoch sees a different draw. These transforms are applied by the
training loop, never persisted.
"""

from __future__ import annotations

import albumentations as A

# YOLO format: (x_center, y_center, width, height), all normalised to [0, 1].
_BBOX_PARAMS = A.BboxParams(
    format="yolo",
    label_fields=["class_labels"],
    # Drop a box if augmentation pushed most of it out of frame. A sliver of a
    # pothole at the image edge is a bad training signal — the model would learn
    # to fire on partial dark shapes.
    min_visibility=0.3,
    min_area=16.0,
)


def build_train_transforms(img_size: int = 640) -> A.Compose:
    """Training-time augmentation.

    Tuned for forward-facing vehicle camera footage. The photometric transforms
    matter more than the geometric ones here: Indian road footage varies far
    more in glare, dust haze, and shadow than it does in camera pose.

    Args:
        img_size: Square target edge length in pixels.

    Returns:
        A bbox-aware Albumentations pipeline.
    """
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=0,
                fill=114,  # Ultralytics' letterbox grey
            ),
            # Horizontal flip is safe: a pothole is not chiral. Vertical flip is
            # not — it would put the road surface in the sky.
            A.HorizontalFlip(p=0.5),
            # Small pose jitter only. Large rotations do not occur on a
            # windshield mount, and training for them wastes capacity.
            A.Affine(
                translate_percent=(-0.06, 0.06),
                scale=(0.85, 1.15),
                rotate=(-8, 8),
                border_mode=0,
                fill=114,
                p=0.5,
            ),
            # Lighting: the dominant real-world variation. Harsh midday sun,
            # overcast haze, and low evening light all change surface contrast
            # far more than they change shape.
            A.RandomBrightnessContrast(
                brightness_limit=0.25, contrast_limit=0.25, p=0.6
            ),
            A.HueSaturationValue(
                hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=20, p=0.3
            ),
            # Wet tarmac, dust film, and low sun all flatten or blow out
            # contrast in ways plain brightness jitter does not reproduce.
            A.OneOf(
                [
                    A.RandomShadow(p=1.0),
                    A.RandomSunFlare(src_radius=80, p=1.0),
                    A.RandomFog(p=1.0),
                ],
                p=0.2,
            ),
            # Sensor and motion degradation. A pothole at 60 km/h is not sharp.
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.ImageCompression(quality_range=(45, 85), p=1.0),
                ],
                p=0.3,
            ),
            A.GaussNoise(p=0.2),
        ],
        bbox_params=_BBOX_PARAMS,
    )


def build_val_transforms(img_size: int = 640) -> A.Compose:
    """Validation/test preprocessing.

    Resize and pad only. Any augmentation here would make the metric measure
    the augmentation rather than the model.
    """
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=0,
                fill=114,
            ),
        ],
        bbox_params=_BBOX_PARAMS,
    )
