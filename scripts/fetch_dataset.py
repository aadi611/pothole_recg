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
    # A Roboflow export mirrored on Hugging Face, which is what makes it
    # fetchable without an account. Roboflow's own export links are
    # account-specific and expire; RDD2022 on S3 returns 403, Kaggle needs an
    # API token, and figshare serves an HTML interstitial instead of the file.
    "roboflow-pothole": Source(
        name="roboflow-pothole",
        url=(
            "https://huggingface.co/datasets/rupesh002/pothole-detection-dataset"
            "/resolve/main/pothole_dataset.zip"
        ),
        description="Roboflow pothole.v1i, YOLO format, 715 images (563 train / 152 val)",
        licence="CC BY 4.0",
    ),
    # The domain-matched set: RDD2022 has a dedicated India split with real
    # Indian road surfaces (docs/DATASETS.md). Two caveats before using it:
    # 13 GB, and Pascal VOC XML rather than YOLO, so it needs conversion.
    "rdd2022": Source(
        name="rdd2022",
        url=(
            "https://huggingface.co/datasets/Duke5t/pothole_detection"
            "/resolve/main/RDD2022.zip"
        ),
        description="RDD2022 road damage, 13GB, Pascal VOC (conversion required)",
        licence="Open for research - verify terms for commercial deployment",
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



def _flatten_single_root(root: Path) -> Path:
    """Unwrap a zip whose contents sit inside one nested folder.

    Roboflow exports extract to e.g. ``pothole.v1i.yolov4pytorch/``, which
    would otherwise leave data.yaml one level below where callers expect it.
    """
    if (root / "data.yaml").is_file():
        return root
    children = [c for c in root.iterdir() if c.is_dir()]
    if len(children) == 1 and (children[0] / "data.yaml").is_file():
        inner = children[0]
        for item in list(inner.iterdir()):
            shutil.move(str(item), str(root / item.name))
        inner.rmdir()
    return root


def _rewrite_data_yaml(root: Path) -> None:
    """Point data.yaml at this machine.

    Upstream exports commonly ship an absolute ``path:`` from whoever created
    them - this one carried ``C:/Users/bhard/Downloads/...``. Ultralytics
    resolves ``path`` against the working directory rather than against the
    yaml file, so a relative path does not fix it either: it has to be
    absolute and local.
    """
    data_yaml = root / "data.yaml"
    if not data_yaml.is_file():
        return

    names = '["pothole"]'
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("names:"):
            names = line.split(":", 1)[1].strip()
            break

    body = "\n".join([
        "# Rewritten by scripts/fetch_dataset.py.",
        "# `path` is absolute because Ultralytics resolves it against the",
        "# working directory rather than against this file.",
        f"path: {root.resolve().as_posix()}",
        "train: train/images",
        "val: valid/images",
        "",
        "nc: 1",
        f"names: {names}",
        "",
    ])
    data_yaml.write_text(body, encoding="utf-8")
    print(f"  rewrote {data_yaml.name} with a local absolute path")


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

    target = _flatten_single_root(target)
    _rewrite_data_yaml(target)
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
