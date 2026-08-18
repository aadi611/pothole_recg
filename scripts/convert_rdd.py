#!/usr/bin/env python
"""Convert RDD2022 (Pascal VOC XML) to YOLO format.

RDD2022 is the domain-matched dataset: it has a dedicated India split with
real Indian road surfaces, which the Roboflow set does not (docs/DATASETS.md).
It ships Pascal VOC XML rather than YOLO txt, so it needs converting first.

Defaults to the **India split only**. The full archive spans six countries;
mixing them back in reintroduces the domain gap we are trying to close.

RDD2022 damage classes:
    D00 longitudinal crack   D10 transverse crack
    D20 alligator crack      D40 pothole

By default only D40 (pothole) is kept, matching the active class map in
config/detection.yaml. Pass --classes to keep more.

Usage:
    python scripts/convert_rdd.py --src datasets/RDD2022 --country India
    python scripts/convert_rdd.py --src datasets/RDD2022 --country India \\
        --classes D40 D20 --val-split 0.2
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# RDD2022 label -> our class index. Order defines the YOLO class ids.
DEFAULT_CLASSES = ["D40"]

CLASS_MEANING = {
    "D00": "longitudinal crack",
    "D10": "transverse crack",
    "D20": "alligator crack",
    "D40": "pothole",
}


def _find_split(src: Path, country: str) -> tuple[Path, Path]:
    """Locate the images/ and annotations/ directories for one country.

    RDD2022 archives vary in layout between releases, so search rather than
    assume a fixed path — a wrong guess here produces zero labels and a model
    that trains to zero mAP with no obvious error.
    """
    candidates = [
        src / country / "train",
        src / "RDD2022" / country / "train",
        src / country,
    ]
    for base in candidates:
        images = base / "images"
        annots = base / "annotations" / "xmls"
        if not annots.is_dir():
            annots = base / "annotations"
        if images.is_dir() and annots.is_dir():
            return images, annots

    # Fall back to a recursive search for any directory holding both.
    for images in src.rglob("images"):
        if country.lower() not in str(images).lower():
            continue
        for annots in (images.parent / "annotations" / "xmls", images.parent / "annotations"):
            if annots.is_dir():
                return images, annots

    raise SystemExit(
        f"error: could not locate images/ and annotations/ for '{country}' under {src}.\n"
        f"Top-level contents: {sorted(p.name for p in src.iterdir())[:12]}"
    )


def _convert_one(
    xml_path: Path, keep: list[str]
) -> tuple[list[str], int]:
    """Convert one VOC XML to YOLO lines. Returns (lines, skipped_count)."""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        print(f"  warning: unparseable XML {xml_path.name}: {exc}", file=sys.stderr)
        return [], 0

    size = root.find("size")
    if size is None:
        return [], 0
    width = float(size.findtext("width", "0"))
    height = float(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        return [], 0

    lines: list[str] = []
    skipped = 0
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in keep:
            skipped += 1
            continue

        box = obj.find("bndbox")
        if box is None:
            continue
        x1 = float(box.findtext("xmin", "0"))
        y1 = float(box.findtext("ymin", "0"))
        x2 = float(box.findtext("xmax", "0"))
        y2 = float(box.findtext("ymax", "0"))

        # Clamp to the image: a few RDD boxes exceed the stated bounds, and an
        # out-of-range YOLO coordinate is silently dropped by the loader.
        x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
        y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            continue

        cx = ((x1 + x2) / 2) / width
        cy = ((y1 + y2) / 2) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        lines.append(f"{keep.index(name)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return lines, skipped


def convert(
    src: Path, out: Path, country: str, keep: list[str], val_split: float, seed: int
) -> None:
    images_dir, annots_dir = _find_split(src, country)
    print(f"images:      {images_dir}")
    print(f"annotations: {annots_dir}")

    xmls = sorted(annots_dir.glob("*.xml"))
    if not xmls:
        raise SystemExit(f"error: no XML annotations found in {annots_dir}")
    print(f"found {len(xmls)} annotation files")

    # Only keep images that actually contain a class we want. An image with no
    # pothole is a background sample; a few help, but thousands teach the model
    # that "nothing" is the usual answer.
    pairs: list[tuple[Path, list[str]]] = []
    empty = 0
    for xml_path in xmls:
        lines, _ = _convert_one(xml_path, keep)
        if not lines:
            empty += 1
            continue
        image = images_dir / f"{xml_path.stem}.jpg"
        if not image.is_file():
            continue
        pairs.append((image, lines))

    print(f"{len(pairs)} images contain {keep}; {empty} had none and were dropped")
    if not pairs:
        raise SystemExit(
            f"error: no images matched classes {keep}. "
            f"Available RDD classes: {sorted(CLASS_MEANING)}"
        )

    random.Random(seed).shuffle(pairs)
    cut = int(len(pairs) * (1 - val_split))
    splits = {"train": pairs[:cut], "valid": pairs[cut:]}

    if out.exists():
        shutil.rmtree(out)
    for split, items in splits.items():
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        for image, lines in items:
            shutil.copy2(image, out / split / "images" / image.name)
            (out / split / "labels" / f"{image.stem}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
        print(f"  {split}: {len(items)} images")

    names = "[" + ", ".join(f'"{CLASS_MEANING.get(c, c)}"' for c in keep) + "]"
    (out / "data.yaml").write_text(
        "\n".join([
            f"# RDD2022 {country} split, converted from Pascal VOC by",
            "# scripts/convert_rdd.py. `path` is absolute because Ultralytics",
            "# resolves it against the working directory, not against this file.",
            f"path: {out.resolve().as_posix()}",
            "train: train/images",
            "val: valid/images",
            "",
            f"nc: {len(keep)}",
            f"names: {names}",
            "",
        ]),
        encoding="utf-8",
    )
    print(f"\nDataset ready: {out}")
    print(f"Train with:\n  python scripts/train_detector.py --data {out.as_posix()}/data.yaml --device 0")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="Extracted RDD2022 directory")
    parser.add_argument("--country", default="India", help="Country split (default: India)")
    parser.add_argument("--out", default="datasets/rdd2022-india", help="Output directory")
    parser.add_argument(
        "--classes", nargs="+", default=DEFAULT_CLASSES,
        help=f"RDD classes to keep. Options: {sorted(CLASS_MEANING)}",
    )
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = Path(args.src)
    if not src.is_dir():
        print(f"error: {src} is not a directory", file=sys.stderr)
        return 1

    unknown = [c for c in args.classes if c not in CLASS_MEANING]
    if unknown:
        print(f"error: unknown classes {unknown}; valid: {sorted(CLASS_MEANING)}", file=sys.stderr)
        return 1

    convert(src, Path(args.out), args.country, args.classes, args.val_split, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
