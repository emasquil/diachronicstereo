#!/usr/bin/env python3
"""
Generate the date-differences CSVs.

Outputs:
  - DFC2019 OMA/JAX: parse .IMD metadata and compute circular day distances.
  - SatNeRF DFC/IARPA: parse RPC JSON acquisition_date (with IARPA fallback).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from statistics import mean, median

MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}
IARPA_DATE_RE = re.compile(
    r"(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})"
)
IMD_FIRSTLINE_RE = re.compile(r"^\s*firstLineTime\s*=\s*([^;]+);", re.MULTILINE)


@dataclass(frozen=True)
class Stats:
    max_days: int | None
    min_days: int | None
    mean_days: float | None
    median_days: float | None
    count: int
    over_60_days: int


def normalize_year(d: date) -> date:
    """Map to non-leap 2001; clamp Feb 29 -> 28."""
    if d.month == 2 and d.day == 29:
        return date(2001, 2, 28)
    return date(2001, d.month, d.day)


def circular_day_distance(a: date, b: date) -> int:
    """Circular day distance over a 365-day year (month/day only)."""
    da = normalize_year(a)
    db = normalize_year(b)
    diff = abs((db - da).days)
    return min(diff, 365 - diff)


def compute_pairs(id2date: dict[str, date]) -> list[tuple[str, str, int]]:
    """Return all unique id pairs with their circular day distance."""
    ids = list(id2date.keys())
    dates = [id2date[i] for i in ids]
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            pairs.append((ids[i], ids[j], circular_day_distance(dates[i], dates[j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


def summarize_pairs(pairs: list[tuple[str, str, int]]) -> Stats:
    if not pairs:
        return Stats(None, None, None, None, 0, 0)
    diffs = [p[2] for p in pairs]
    return Stats(
        max(diffs),
        min(diffs),
        mean(diffs),
        median(diffs),
        len(diffs),
        sum(d > 60 for d in diffs),
    )


def write_pairs_csv(pairs: list[tuple[str, str, int]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w") as f:
        f.write("id1,id2,days_difference\n")
        for a, b, d in pairs:
            f.write(f"{a},{b},{d}\n")


def read_imd_dates(imd_dir: Path) -> dict[str, date]:
    """Read IMD firstLineTime values into date objects."""
    id2date: dict[str, date] = {}
    for imd_path in sorted(imd_dir.glob("*.[iI][mM][dD]")):
        img_id = imd_path.stem
        try:
            text = imd_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            print(f"Failed to read {imd_path}: {exc}")
            continue

        m = IMD_FIRSTLINE_RE.search(text)
        if not m:
            continue
        ts = m.group(1).strip()
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            print(f"Skipping {imd_path}: bad timestamp {ts!r}")
            continue
        id2date[img_id] = dt.date()
    return id2date


def build_dfc_map(root: Path) -> dict[str, date]:
    """DFC: read acquisition_date from RPC JSON as YYYYMMDDHHMMSS."""
    id2date: dict[str, date] = {}
    for aoi_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for jp in aoi_dir.glob("*.json"):
            img_id = jp.stem
            s = json.loads(jp.read_text()).get("acquisition_date")
            if not s or len(s) < 8:
                continue
            m = int(s[4:6])
            d = int(s[6:8])
            id2date.setdefault(img_id, normalize_year(date(2001, m, d)))
    return id2date


def build_iarpa_map(root: Path) -> dict[str, date]:
    """IARPA: prefer acquisition_date; else fallback to DDMMMYY token in filename."""
    id2date: dict[str, date] = {}
    for aoi_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for jp in aoi_dir.glob("*.json"):
            img_id = jp.stem
            d: date | None = None
            s = json.loads(jp.read_text()).get("acquisition_date")
            if s and len(s) >= 8:
                d = normalize_year(date(2001, int(s[4:6]), int(s[6:8])))
            else:
                m = IARPA_DATE_RE.search(img_id)
                if m:
                    d = normalize_year(date(2001, MONTHS[m.group(2)], int(m.group(1))))
            if d is not None:
                id2date.setdefault(img_id, d)
    return id2date


def print_stats(label: str, stats: Stats) -> None:
    if stats.count == 0:
        print(f"{label} stats: no pairs")
        return
    print(
        f"{label} stats: max={stats.max_days}  min={stats.min_days}  "
        f"mean={stats.mean_days:.2f}  median={stats.median_days}  "
        f"Npairs={stats.count}  >60d={stats.over_60_days}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate date-differences CSVs from IMD and RPC metadata.",
    )
    parser.add_argument(
        "--oma-imd-root",
        type=Path,
        default=Path("/home/emasquil/datasets/DFC2019/Track3-Metadata/OMA"),
        help="OMA IMD directory (contains *.IMD files).",
    )
    parser.add_argument(
        "--jax-imd-root",
        type=Path,
        default=Path("/home/emasquil/datasets/DFC2019/Track3-Metadata/JAX"),
        help="JAX IMD directory (contains *.IMD files).",
    )
    parser.add_argument(
        "--dfc-rpc-root",
        type=Path,
        default=Path(
            "/home/emasquil/datasets/SatNeRF_DFC2019/root_dir/crops_rpcs_ba_v2"
        ),
        help="DFC RPC root (AOI subdirs with *.json files).",
    )
    parser.add_argument(
        "--iarpa-rpc-root",
        type=Path,
        default=Path("/home/emasquil/datasets/SatNeRF_IARPA/root_dir/rpcs_ba"),
        help="IARPA RPC root (AOI subdirs with *.json files).",
    )
    parser.add_argument(
        "--oma-out",
        type=Path,
        default=Path("/home/emasquil/datasets/DFC2019/OMA_date_differences.csv"),
        help="Output CSV for OMA IMD pairs.",
    )
    parser.add_argument(
        "--jax-out",
        type=Path,
        default=Path("/home/emasquil/datasets/DFC2019/JAX_date_differences.csv"),
        help="Output CSV for JAX IMD pairs.",
    )
    parser.add_argument(
        "--dfc-out",
        type=Path,
        default=Path("/home/emasquil/datasets/SatNeRF_DFC2019/date_differences.csv"),
        help="Output CSV for SatNeRF DFC pairs.",
    )
    parser.add_argument(
        "--iarpa-out",
        type=Path,
        default=Path("/home/emasquil/datasets/SatNeRF_IARPA/date_differences.csv"),
        help="Output CSV for SatNeRF IARPA pairs.",
    )
    args = parser.parse_args()

    oma_dates = read_imd_dates(args.oma_imd_root)
    jax_dates = read_imd_dates(args.jax_imd_root)
    dfc_map = build_dfc_map(args.dfc_rpc_root)
    iarpa_map = build_iarpa_map(args.iarpa_rpc_root)

    oma_pairs = compute_pairs(oma_dates)
    jax_pairs = compute_pairs(jax_dates)
    dfc_pairs = compute_pairs(dfc_map)
    iarpa_pairs = compute_pairs(iarpa_map)

    print_stats("OMA", summarize_pairs(oma_pairs))
    print_stats("JAX", summarize_pairs(jax_pairs))
    print_stats("DFC", summarize_pairs(dfc_pairs))
    print_stats("IARPA", summarize_pairs(iarpa_pairs))

    write_pairs_csv(oma_pairs, args.oma_out)
    write_pairs_csv(jax_pairs, args.jax_out)
    write_pairs_csv(dfc_pairs, args.dfc_out)
    write_pairs_csv(iarpa_pairs, args.iarpa_out)

    print(f"wrote {args.oma_out}")
    print(f"wrote {args.jax_out}")
    print(f"wrote {args.dfc_out}")
    print(f"wrote {args.iarpa_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
