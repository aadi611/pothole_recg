#!/usr/bin/env python
"""Fetch a pothole detection dataset into ``datasets/``.

Non-interactive, argparse-driven (NFR-DEV-02) — the original project's
downloaders were ``input()`` menus that could not run in CI.

Datasets are never committed; ``datasets/`` is gitignored. See
``docs/DATASETS.md`` for what is available and why.

Usage:
    python scripts/fetch_dataset.py --source roboflow-pothole
    python scripts/fetch_dataset.py --source roboflow-pothole --force
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    description: str
    licence: str


SOURCES: dict[str, Source] = {
    # Roboflow serves a direct zip for public datasets. YOLOv8 format: images
    # plus .txt label files plus a data.yaml, already split train/valid/test.
    "roboflow-pothole": Source(
        name="roboflow-pothole",
        url=(
            "https://universe.roboflow.com/ds/dQTGqCJcBw?key=YourKeyHere"
        ),
        description="Roboflow Universe pothole detection, YOLOv8 format",
        licence="CC BY 4.0 — verify on the dataset page before deployment",
    ),
}


def _download(url: str, dest: Path) -> None:
    """Stream a URL to disk with progress."""
    print(f"Downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            total = int(response.headers.get("content-length", 0))
            done = 0
            with dest.open("wb") as fh:
                while chunk := response.read(65536):
                    fh.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = 100 * done / total
                        print(f"\r  {pct:5.1f}%  {done/1e6:.1f} MB", end="", flush=True)
            print()
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            f"error: download failed with HTTP {exc.code}.\n"
            f"Roboflow export links are account-specific and expire. Get a "
            f"fresh one:\n"
            f"  1. https://universe.roboflow.com/search?q=pothole\n"
            f"  2. Pick a dataset, choose Download > YOLOv8 > 'show download code'\n"
            f"  3. Re-run with --url '<the link>'\n"
            f"Or place a YOLO-format dataset at {DATASETS}/<name>/ manually."
        ) from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"error: network failure: {exc.reason}") from exc


def _verify_yolo_layout(root: Path) -> None:
    """Confirm the extracted tree looks like a YOLO dataset.

    A silently-wrong dataset layout produces a model that trains to zero mAP
    with no obvious error, so fail early and loudly instead.
    """
    data_yaml = root / "data.yaml"
    if not data_yaml.is_file():
        raise SystemExit(
            f"error: no data.yaml at {root}. This does not look like a YOLO "
            f"export. Contents: {[p.name for p in root.iterdir()][:10]}"
        )

    splits = [d for d in ("train", "valid", "test") if (root / d).is_dir()]
    if "train" not in splits:
        raise SystemExit(f"error: no train/ split under {root}")

    for split in splits:
        images = list((root / split / "images").glob("*")) if (root / split / "images").is_dir() else []
        labels = list((root / split / "labels").glob("*.txt")) if (root / split / "labels").is_dir() else []
        print(f"  {split}: {len(images)} images, {len(labels)} labels")
        if images and not labels:
            raise SystemExit(
                f"error: {split}/ has images but no labels. Detection training "
                f"needs boxes; this looks like a classification dataset."
            )


def fetch(source: Source, *, url_override: str | None, force: bool) -> Path:
    DATASETS.mkdir(parents=True, exist_ok=True)
    target = DATASETS / source.name

    if target.exists():
        if not force:
            print(f"{target} already exists. Use --force to re-download.")
            _verify_yolo_layout(target)
            return target
        shutil.rmtree(target)

    archive = DATASETS / f"{source.name}.zip"
    _download(url_override or source.url, archive)

    print(f"Extracting to {target}")
    target.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
    except zipfile.BadZipFile as exc:
        raise SystemExit(
            "error: downloaded file is not a valid zip. The URL may have "
            "returned an HTML error page instead of the dataset."
        ) from exc
    finally:
        archive.unlink(missing_ok=True)

    _verify_yolo_layout(target)
    print(f"\nDataset ready: {target}")
    print(f"Licence: {source.licence}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=sorted(SOURCES),
        default="roboflow-pothole",
        help="Which dataset to fetch",
    )
    parser.add_argument(
        "--url",
        help="Override the download URL (Roboflow export links expire)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if present"
    )
    args = parser.parse_args()

    source = SOURCES[args.source]
    fetch(source, url_override=args.url, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
