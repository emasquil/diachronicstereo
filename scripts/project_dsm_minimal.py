#!/usr/bin/env python3
"""
Minimal CLI to project one DSM onto a single image plane using RPCs.

It upsamples the DSM, transforms its grid to WGS84, projects (lon,lat,z) into
image space with the RPC, resolves overlapping points via GPU z-buffer
(torch_scatter.scatter_max), and writes a float32 GeoTIFF with NaNs preserved
and LZW compression.

Required args: --img <image.tif> --rpc <rpc.json> --dsm <dsm.tif> --out <dst.tif>.
"""

import argparse
import os
from typing import Tuple

import numpy as np
import rasterio
import torch
from pyproj import Transformer
from torch_scatter import scatter_max

# repo-local utilities expected in PYTHONPATH (same as your original code)
import sys

sys.path.append("..")
from utils import misc  # rpc_from_json, upsample_dsm

UPSCALE_DSM_FACTOR = 4  # keep exactly as in your pipeline


def project_dsm(img_path: str, rpc_path: str, dsm_path: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path):
        print(f"[skip] Projected DSM already exists: {out_path}")
        return

    # --- Read image dims + base metadata
    with rasterio.open(img_path) as src_img:
        img_h, img_w = src_img.height, src_img.width
        meta = src_img.meta.copy()

    # Single-band float32 output (preserve NaNs), LZW compression
    meta.update(
        {
            "count": 1,
            "dtype": "float32",
            "compress": "lzw",
            "predictor": 3,  # better for float
            "width": img_w,
            "height": img_h,
        }
    )

    # --- Upsample DSM and get XY grid + DSM metadata (incl. CRS)
    dsm_up, (X_dsm, Y_dsm), dsm_meta = misc.upsample_dsm(dsm_path, UPSCALE_DSM_FACTOR)

    # --- Load RPC
    rpc = misc.rpc_from_json(rpc_path)

    # --- Transform DSM (X,Y) from DSM CRS -> (lon,lat) WGS84
    dsm_crs = dsm_meta["crs"]  # should be something like "EPSG:32614", etc.
    transformer = Transformer.from_crs(dsm_crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(X_dsm, Y_dsm)

    # --- RPC projection: (lon, lat, z) -> (x,y) in image
    x_img, y_img = rpc.projection(lon.ravel(), lat.ravel(), dsm_up.ravel())
    x_img = np.round(x_img).astype(np.int64)
    y_img = np.round(y_img).astype(np.int64)
    z_vals = dsm_up.ravel()

    # --- Keep in-bounds & finite Z
    valid = (
        (x_img >= 0)
        & (x_img < img_w)
        & (y_img >= 0)
        & (y_img < img_h)
        & np.isfinite(z_vals)
    )
    if not np.any(valid):
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(np.full((img_h, img_w), np.nan, dtype=np.float32), 1)
        print(f"[ok] Empty projection (no valid points). Saved NaN raster: {out_path}")
        return

    x_img = x_img[valid]
    y_img = y_img[valid]
    z_vals = z_vals[valid].astype(np.float32)

    # --- Prepare output (NaNs)
    proj = np.full((img_h, img_w), np.nan, dtype=np.float32)

    # Flattened pixel indices
    flat_idx = (y_img * img_w + x_img).astype(np.int64)

    # Identify unambiguous pixels (those that occur exactly once)
    uniq_idx, counts = np.unique(flat_idx, return_counts=True)
    unamb_set = set(uniq_idx[counts == 1])
    unamb_mask = np.fromiter(
        (i in unamb_set for i in flat_idx), count=len(flat_idx), dtype=bool
    )

    # Unambiguous: write directly
    if np.any(unamb_mask):
        ux, uy, uz = x_img[unamb_mask], y_img[unamb_mask], z_vals[unamb_mask]
        proj[uy, ux] = uz

    # Ambiguous: z-buffer with scatter_max on GPU
    if np.any(~unamb_mask):
        amb_idx_np = flat_idx[~unamb_mask]
        amb_z_np = z_vals[~unamb_mask]

        amb_idx = torch.from_numpy(amb_idx_np).to("cuda")
        amb_z = torch.from_numpy(amb_z_np).to("cuda")

        # scatter_max returns (values, indices) where values[k] is max for index k
        z_buffer, _ = scatter_max(amb_z, amb_idx)
        z_buffer = z_buffer.detach().cpu().numpy()

        ax, ay = x_img[~unamb_mask], y_img[~unamb_mask]
        a_flat = (ay * img_w + ax).astype(np.int64)
        proj[ay, ax] = z_buffer[a_flat]

    # --- Save GeoTIFF preserving NaNs
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(proj, 1)

    print(f"[ok] Projected DSM saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Project DSM onto image plane using RPCs.")
    ap.add_argument("--img", required=True, help="Path to LEFT image (GeoTIFF).")
    ap.add_argument("--rpc", required=True, help="Path to LEFT RPC JSON.")
    ap.add_argument("--dsm", required=True, help="Path to ground-truth DSM (GeoTIFF).")
    ap.add_argument("--out", required=True, help="Output projected DSM (GeoTIFF).")
    args = ap.parse_args()

    project_dsm(args.img, args.rpc, args.dsm, args.out)


if __name__ == "__main__":
    main()
