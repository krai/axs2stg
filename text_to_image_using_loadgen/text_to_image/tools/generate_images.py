#!/usr/bin/env python3
"""
extract_images.py – Dump every generated image from an MLPerf Stable Diffusion XL
accuracy log and save a caption mapping in the style you requested: each line in
*captions.txt* is now written as

    <idx>_<caption text>

so image 0’s caption line will start with `0_`, image 1’s with `1_`, etc.

Optionally, add `--separate-caption-files` if you prefer an individual text file
named `<idx>_caption.txt` next to each PNG.

Usage
-----
python extract_images.py \
    --mlperf-accuracy-file offline_test/mlperf_log_accuracy.json \
    --caption-path coco2014/captions/captions_source.tsv \
    --output-dir all_generated_images \
    --separate-caption-files         # <‑‑ optional flag

Arguments
---------
--mlperf-accuracy-file : required – path to *mlperf_log_accuracy.json* produced by
                         LoadGen.
--caption-path         : optional – TSV file with two columns (`image_idx`,
                         `caption`) that matches COCO-style indexing. When
                         supplied, a *captions.txt* file is written in the
                         output directory with lines of the form:
                             <idx>_<caption text>
                         and/or individual `<idx>_caption.txt` files when the
                         `--separate-caption-files` flag is used.
--output-dir           : optional – directory where PNGs (and caption files) are
                         saved; created if it doesn’t exist. Defaults to
                         *all_generated_images* in the current working dir.
--separate-caption-files : optional – write a dedicated text file per image
                         (named `<idx>_caption.txt`) instead of/in addition to
                         `captions.txt`.

The script assumes each entry in *mlperf_log_accuracy.json* contains:
    {
        "qsl_idx": <int>,    # index into the original caption dataset
        "data":    <str>     # hex‑encoded bytes of a 1024×1024×3 uint8 image
    }

Images are decoded, converted to PNG, and written as
    <output-dir>/<qsl_idx>.png
"""
import argparse
import json
import os
from typing import Optional

from PIL import Image
import numpy as np
import pandas as pd


###############################################################################
# Core helpers
###############################################################################

def slugify(text: str, max_len: int = 60) -> str:
    """Create a filesystem‑safe slug from *text* (for filenames)."""
    import re
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)         # drop non‑word chars
    text = re.sub(r"[\s_-]+", "_", text)           # collapse whitespace & dashes
    return text[:max_len].rstrip("_")


def save_images_with_captions(
    json_path: str,
    caption_path: Optional[str] = None,
    output_dir: str = "all_generated_images",
    height: int = 1024,
    width: int = 1024,
    separate_caption_files: bool = False,
) -> None:
    """Decode every image inside *json_path* and write it to *output_dir*.

    If *caption_path* is supplied, a *captions.txt* file mapping `idx` →
    `caption` is written alongside the PNGs using the requested underscore style
    (e.g. "0_A dog in the park").  When *separate_caption_files* is True, an
    additional `<idx>_caption.txt` is emitted per image.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load caption table early (optional)
    captions_df = None
    caption_file_handle = None
    if caption_path is not None:
        captions_df = pd.read_csv(caption_path, sep="\t", header=0)
        caption_file_handle = open(os.path.join(output_dir, "captions.txt"), "w", encoding="utf-8")

    # Read accuracy log
    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Iterate and save
    for entry in results:
        idx = entry["qsl_idx"]
        raw_bytes = bytes.fromhex(entry["data"])
        img_array = np.frombuffer(raw_bytes, np.uint8).reshape(height, width, 3)
        img = Image.fromarray(img_array)
        img.save(os.path.join(output_dir, f"{idx}.png"))

        # Write caption(s) if available
        if captions_df is not None:
            try:
                caption = str(captions_df.loc[idx, "caption"])
            except (KeyError, IndexError):
                caption = ""
            # captions.txt with underscore style
            caption_file_handle.write(f"{idx}_{caption}\n")

            # Optional individual caption file
            if separate_caption_files:
                with open(os.path.join(output_dir, f"{idx}_caption.txt"), "w", encoding="utf-8") as cf:
                    cf.write(caption + "\n")

    if caption_file_handle is not None:
        caption_file_handle.close()


###############################################################################
# CLI wrappers
###############################################################################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract PNG images (and optional caption map) from an MLPerf Stable Diffusion XL accuracy log.")
    parser.add_argument("--mlperf-accuracy-file", required=True,
                        help="Path to mlperf_log_accuracy.json")
    parser.add_argument("--caption-path", default=None,
                        help="Optional TSV file containing captions for each QSL index (two columns: image_idx, caption)")
    parser.add_argument("--output-dir", default="all_generated_images",
                        help="Directory to write PNGs and captions – created if missing")
    parser.add_argument("--separate-caption-files", action="store_true",
                        help="Write an individual <idx>_caption.txt next to each image in addition to captions.txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_images_with_captions(
        json_path=args.mlperf_accuracy_file,
        caption_path=args.caption_path,
        output_dir=args.output_dir,
        separate_caption_files=args.separate_caption_files,
    )


if __name__ == "__main__":
    main()

