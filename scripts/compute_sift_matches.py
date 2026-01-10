#!/usr/bin/env python3
"""
Compute SIFT match counts for stereo pairs and log results to CSV.

The script scans a pairs directory for *.iio files whose names encode left/right
image IDs (e.g., JAX_0000-OMA_0001.iio). For each pair not already present in
the output CSV, it locates the corresponding TIFFs in the DFC2019 dataset
(Track3-RGB-1 for JAX, Track3-RGB-2 for OMA), loads them in grayscale, and
computes feature matches via utils.rectification_utils.compute_matches. It
appends the match count per pair to a CSV so interrupted runs can resume
without recomputing completed entries.
"""
import argparse
import csv
import os
from glob import glob

import cv2
import iio
from tqdm import tqdm

import sys

sys.path.append("../")
from utils import rectification_utils as ru  # noqa: E402


def parse_pair_filename(path):
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    if "-" not in stem:
        raise ValueError(f"Unexpected pair filename (no '-'): {base}")
    left_id, right_id = stem.split("-", 1)
    return left_id, right_id


def find_tif(dataset_dir, image_id):
    """Map IDs to folders deterministically: JAX -> Track3-RGB-1, OMA -> Track3-RGB-2."""
    if image_id.startswith("JAX_"):
        p = os.path.join(dataset_dir, "Track3-RGB-1", f"{image_id}.tif")
    elif image_id.startswith("OMA_"):
        p = os.path.join(dataset_dir, "Track3-RGB-2", f"{image_id}.tif")
    else:
        raise ValueError(f"Unknown ID prefix for {image_id}. Expected JAX_ or OMA_.")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing TIFF for {image_id}: {p}")
    return p


def load_grayscale(path):
    img = iio.read(path)
    if img.ndim == 3 and img.shape[2] >= 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def read_processed_pairs(out_csv):
    done = set()
    if not os.path.exists(out_csv):
        return done
    with open(out_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            l = row.get("left_img")
            r = row.get("right_img")
            if l and r:
                done.add(f"{l}|{r}")
    return done


def append_row(out_csv, header_written, left_id, right_id, n_matches):
    write_header = not os.path.exists(out_csv) and not header_written
    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["left_img", "right_img", "sift_matches"])
        if write_header:
            writer.writeheader()
        writer.writerow(
            {"left_img": left_id, "right_img": right_id, "sift_matches": int(n_matches)}
        )
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, help="Path to DFC2019 root")
    ap.add_argument(
        "--pairs-subdir",
        default="stereo_pairs/L",
        help="Relative path from dataset-dir to the pair files (*.iio)",
    )
    ap.add_argument(
        "--out-csv",
        default="sift_matches.csv",
        help="Output CSV path (absolute or under dataset-dir)",
    )
    args = ap.parse_args()

    dataset_dir = args.dataset_dir
    pairs_dir = (
        args.pairs_subdir
        if os.path.isabs(args.pairs_subdir)
        else os.path.join(dataset_dir, args.pairs_subdir)
    )
    out_csv = (
        args.out_csv
        if os.path.isabs(args.out_csv)
        else os.path.join(dataset_dir, args.out_csv)
    )

    pair_files = sorted(glob(os.path.join(pairs_dir, "*.iio")))
    if not pair_files:
        raise SystemExit(f"No .iio pair files found in {pairs_dir}")

    already = read_processed_pairs(out_csv)
    to_process = []
    for pf in pair_files:
        try:
            l_id, r_id = parse_pair_filename(pf)
        except Exception as e:
            print(f"[warn] Skipping malformed pair file {pf}: {e}")
            continue
        key = f"{l_id}|{r_id}"
        if key not in already:
            to_process.append((l_id, r_id))

    print(
        f"Total pairs found: {len(pair_files)} | Already in CSV: {len(already)} | To compute: {len(to_process)}"
    )

    header_written = os.path.exists(out_csv)

    try:
        for l_id, r_id in tqdm(to_process, desc="Computing SIFT matches", unit="pair"):
            try:
                l_path = find_tif(dataset_dir, l_id)  # JAX -> 1, OMA -> 2
                r_path = find_tif(dataset_dir, r_id)

                left_img = load_grayscale(l_path)
                right_img = load_grayscale(r_path)

                matches = ru.compute_matches(left_img, right_img)
                n_matches = len(matches)

                append_row(out_csv, header_written, l_id, r_id, n_matches)
                header_written = True
            except KeyboardInterrupt:
                print(
                    "\n[info] Interrupted by user. Progress saved; resume will skip completed pairs."
                )
                break
            except Exception as e:
                print(f"[warn] Failed {l_id} - {r_id}: {e}")
                continue
    except KeyboardInterrupt:
        print(
            "\n[info] Interrupted by user. Progress saved; resume will skip completed pairs."
        )


if __name__ == "__main__":
    main()
