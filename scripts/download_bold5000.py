"""Utility to download the BOLD5000 fMRI-image dataset via OpenNeuro.

The BOLD5000 dataset (OpenNeuro ID: ds001499) provides ~5k natural images with
whole-brain 7T fMRI recordings from four participants. This script downloads a
manageable subset by default (stimuli images and pycortex surface betas) while
allowing users to request the full BIDS archive if desired.

Examples
--------
Download the default subset (stimuli + pycortex derivatives) into ./data:
    python scripts/download_bold5000.py --output ./data

Fetch full raw NIfTI runs as well (large download):
    python scripts/download_bold5000.py --output ./data --include-all

Requires the ``openneuro`` Python client (listed in requirements.txt).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from openneuro import download


DEFAULT_DATASET = "ds001499"
DEFAULT_INCLUDES = [
    "stimuli/*",  # COCO/ImageNet/Scene images used in the study
    "derivatives/pycortex/*/S*/betas.nii.gz",  # surface-projected beta weights
    "participants.tsv",
    "dataset_description.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET,
        help="OpenNeuro dataset identifier (default: %(default)s for BOLD5000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./data/bold5000"),
        help="Target directory for the downloaded data",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Download the full dataset instead of the curated subset",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Custom include patterns passed to openneuro.download",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=4,
        help="Number of concurrent downloads (passed to openneuro.download)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_dir = args.output.expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    if args.include_all:
        include_patterns = None
    elif args.include:
        include_patterns = args.include
    else:
        include_patterns = DEFAULT_INCLUDES

    print(f"Downloading {args.dataset_id} to {target_dir}...")
    if include_patterns:
        print(f"Using include patterns: {include_patterns}")
    else:
        print("Downloading all dataset files (this may be very large)...")

    download(
        dataset=args.dataset_id,
        target_dir=str(target_dir),
        include=include_patterns,
        resume=True,
        concurrent=args.concurrent,
        silent=False,
    )
    print("Download complete. You can now preprocess fMRI volumes and align them\n"
          "with stimuli filenames to build an ncc.FMRIDataset for training.")


if __name__ == "__main__":
    main()
