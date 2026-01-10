#!/usr/bin/env python3
"""
Batch georeference DSM rasters in-place using sidecar .txt metadata.

The script walks an input directory, finds *_CLS.tif files, and looks for a
matching *_DSM.txt that contains xmin/xoff, ymin/yoff, grid size, and pixel
resolution. It builds an affine transform from that metadata, assigns the CRS
based on an AOI prefix in the filename (e.g., JAX_* -> EPSG:32617), and writes
both transform and CRS into the TIF. Use --dry-run to preview changes without
modifying files.
"""
import argparse
from pathlib import Path

import rasterio
from rasterio.transform import from_origin
from pyproj import CRS

# Map AOI prefix -> EPSG
AOI_EPSG = {
    "JAX": 32617,  # Jacksonville, UTM 17N
    "OMA": 32615,  # Omaha, UTM 15N
}


def read_txt_metadata(txt_path: Path):
    """Read [xoff, yoff, xsize, resolution] from the .txt file."""
    vals = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                vals.append(float(line))
    if len(vals) < 4:
        raise ValueError(
            f"{txt_path} must contain 4 lines: xoff, yoff, xsize, resolution"
        )
    xoff, yoff, xsize, res = vals[:4]
    return float(xoff), float(yoff), int(round(xsize)), float(res)


def compute_transform(xmin, ymin, xsize, res):
    """
    Build an affine transform for a square grid at resolution 'res'.
    In this convention, txt gives xmin=xoff and ymin=yoff.
    The raster's northing (ymax) is ymin + (xsize * res).
    """
    ymax = ymin + xsize * res
    return from_origin(xmin, ymax, res, res)


def choose_epsg_from_name(name: str) -> int:
    """Pick EPSG based on filename prefix (e.g., JAX_*, OMA_*)."""
    prefix = name.split("_", 1)[0].upper()
    if prefix not in AOI_EPSG:
        raise ValueError(
            f"Cannot determine EPSG for '{name}'. Unknown prefix '{prefix}'."
        )
    return AOI_EPSG[prefix]


def georef_inplace(tif_path: Path, txt_path: Path, epsg: int, dry_run: bool = False):
    xoff, yoff, xsize, res = read_txt_metadata(txt_path)
    transform = compute_transform(xoff, yoff, xsize, res)
    crs = CRS.from_epsg(epsg)

    with rasterio.Env():
        with rasterio.open(tif_path, "r+") as dst:
            if dry_run:
                print(
                    f"[DRY] {tif_path.name} -> EPSG:{epsg}, res={res} m, "
                    f"transform={transform}"
                )
                return
            dst.transform = transform
            dst.crs = crs

    print(f"[OK]  {tif_path.name} -> EPSG:{epsg}, res={res} m")


def main():
    ap = argparse.ArgumentParser(
        description="Batch inject CRS+transform into *_DSM.tif using sidecar *_DSM.txt."
    )
    ap.add_argument(
        "input_dir", type=Path, help="Directory to scan for *_DSM.tif files"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done, don't modify files",
    )
    args = ap.parse_args()

    # tifs = sorted(args.input_dir.glob("*_DSM.tif"))
    tifs = sorted(args.input_dir.glob("*_CLS.tif"))
    if not tifs:
        print(f"No *_DSM.tif found under {args.input_dir}")
        return

    for tif in tifs:
        # txt = tif.with_suffix(".txt")
        txt = tif.with_name(tif.name.replace("_CLS.tif", "_DSM.txt"))
        if not txt.exists():
            print(f"[SKIP] {tif.name} (missing sidecar: {txt.name})")
            continue
        try:
            epsg = choose_epsg_from_name(tif.name)
        except ValueError as e:
            print(f"[SKIP] {tif.name} ({e})")
            continue

        try:
            georef_inplace(tif, txt, epsg, dry_run=args.dry_run)
        except Exception as e:
            print(f"[ERR] {tif.name}: {e}")


if __name__ == "__main__":
    main()
