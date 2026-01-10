"""
Project DSM rasters into image space for DFC-style crops and AOIs.

Supports single-crop projection, parallel/sequential batch runs, and full DFC
processing. For each crop, it upsamples the DSM, transforms it to WGS84, projects
via RPC, resolves overlapping points with a GPU z-buffer, and writes a float32
GeoTIFF (NaNs preserved) to a projected_dsm_* directory.
"""

import argparse
import os
import sys
from time import time

import numpy as np
import pandas as pd
import rasterio
import torch
from pyproj import Transformer
from torch_scatter import scatter_max
from tqdm import tqdm

sys.path.append("..")
from utils import misc  # noqa: E402

UPSCALE_DSM_FACTOR = 4
UTM_ZONES = {"OMA": 14, "UCSD": 11, "JAX": 17}
DSM_DIR = "rdsm_ransac_adjusted"


# abstraction of projection logic agnostic to any saving or reading naming convention
def _project_dsm(img_path, rpc_path, dsm_path, projected_dsm_path):
    """
    Project the DSM onto the image plane.

    The projected DSM is saved as a GeoTIFF (with NaNs preserved) under projected_dsm_path.

    Args:
        img_path (str): Path to the satellite image crop (GeoTIFF).
        rpc_path (str): Path to the RPC metadata file (JSON).
        dsm_path (str): Path to the DSM file (GeoTIFF).
        projected_dsm_path (str): Output path for the projected DSM (GeoTIFF).
    """
    os.makedirs(os.path.dirname(projected_dsm_path), exist_ok=True)

    # If output already exists, skip processing
    if os.path.exists(projected_dsm_path):
        print(f"DSM projection already exists at {projected_dsm_path}")
        return

    # --- Load crop image to get shape + base metadata
    with rasterio.open(img_path) as src_img:
        img_h, img_w = src_img.height, src_img.width
        meta = src_img.meta.copy()

    # Single-band float32 output (preserve NaNs), LZW compression
    meta.update(
        {
            "count": 1,
            "dtype": "float32",
            "compress": "lzw",
            "predictor": 3,
        }
    )

    # --- Upsample DSM (and get its XY grid in DSM CRS)
    # Uses your existing utility + factor
    dsm_for_projection, xy_grid, dsm_meta = misc.upsample_dsm(
        dsm_path, UPSCALE_DSM_FACTOR
    )
    X_dsm, Y_dsm = xy_grid  # in DSM CRS coordinates

    # --- Load RPC
    rpc = misc.rpc_from_json(rpc_path)

    # --- Transform DSM (X,Y) -> (lon,lat) using DSM CRS
    dsm_crs = dsm_meta["crs"]  # provided by upsample_dsm
    transformer = Transformer.from_crs(dsm_crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(X_dsm, Y_dsm)

    # --- RPC projection to image plane
    x_img, y_img = rpc.projection(
        lon.flatten(), lat.flatten(), dsm_for_projection.flatten()
    )
    x_img = np.round(x_img).astype(np.int64)
    y_img = np.round(y_img).astype(np.int64)
    dsm_flat = dsm_for_projection.flatten()

    # --- Keep only in-bounds & finite altitudes
    valid = (
        (x_img >= 0)
        & (x_img < img_w)
        & (y_img >= 0)
        & (y_img < img_h)
        & np.isfinite(dsm_flat)
    )
    if not np.any(valid):
        with rasterio.open(projected_dsm_path, "w", **meta) as dst:
            dst.write(np.full((img_h, img_w), np.nan, dtype=np.float32), 1)
        return

    x_img = x_img[valid]
    y_img = y_img[valid]
    dsm_flat = dsm_flat[valid]

    # --- Flat indices per pixel
    flat_idx = (y_img * img_w + x_img).astype(np.int64)

    # --- Unambiguous vs ambiguous pixels
    unique_index, counts = np.unique(flat_idx, return_counts=True)
    unique_set = set(unique_index[counts == 1])
    unamb_mask = np.fromiter(
        (i in unique_set for i in flat_idx), count=len(flat_idx), dtype=bool
    )

    # --- Initialize output with NaNs
    projected_dsm = np.full((img_h, img_w), np.nan, dtype=np.float32)

    # Unambiguous: direct write
    if np.any(unamb_mask):
        ux = x_img[unamb_mask]
        uy = y_img[unamb_mask]
        uz = dsm_flat[unamb_mask].astype(np.float32)
        projected_dsm[uy, ux] = uz

    # Ambiguous: z-buffer with scatter_max (GPU)
    if np.any(~unamb_mask):
        amb_idx = torch.from_numpy(flat_idx[~unamb_mask])
        amb_z = torch.from_numpy(dsm_flat[~unamb_mask].astype(np.float32))
        z_buffer, _ = scatter_max(amb_z.to("cuda"), amb_idx.to("cuda"))
        z_buffer = z_buffer.cpu()

        ax = x_img[~unamb_mask]
        ay = y_img[~unamb_mask]
        a_flat = (ay * img_w + ax).astype(np.int64)
        projected_dsm[ay, ax] = z_buffer[a_flat].numpy()

    # --- Save GeoTIFF preserving NaNs
    with rasterio.open(projected_dsm_path, "w", **meta) as dst:
        dst.write(projected_dsm, 1)


def project_dsm_single_crop(args):
    """
    Project the DSM onto the image plane for a single crop.

    The projected DSM is saved as a GeoTIFF (with NaNs preserved) under:

        <dataset_dir>/dsm_projection/<aoi>/<aoi>_<crop_number>_projected_dsm.tif

    Args:
        args (tuple): (aoi, dataset_dir, crop_number)
    """
    aoi, dataset_dir, crop_number = args
    city = aoi.split("_")[0]

    # Create output directory: dsm_projection/<aoi>
    output_dir = os.path.join(dataset_dir, "projected_dsm_ransac_adjusted", aoi)
    os.makedirs(output_dir, exist_ok=True)

    t0 = time()
    # Define file paths for input data.
    crop_path = os.path.join(
        dataset_dir, "crops_radiometric_correction", aoi, f"{aoi}_{crop_number}_pan.tif"
    )
    root_path = os.path.join(
        dataset_dir, "root_dir_ba", aoi, f"{aoi}_{crop_number}_pan.json"
    )
    dsm_name = [
        f
        for f in os.listdir(os.path.join(dataset_dir, DSM_DIR, aoi))
        if f.endswith(".tif")
    ][0]
    dsm_path = os.path.join(dataset_dir, DSM_DIR, aoi, dsm_name)
    if not os.path.exists(dsm_path):
        raise FileNotFoundError(f"DSM not found for {aoi}")

    dsm_output_fname = os.path.join(
        output_dir,
        dsm_path.split("/")[-1]
        .replace("rdsm", "projected_rdsm")
        .replace(f"{aoi}", f"{aoi}_{crop_number}"),
    )
    if os.path.exists(dsm_output_fname):
        print(f"DSM projection already exists for {aoi}_{crop_number}")
        return

    # Open crop image to obtain dimensions and metadata.
    with rasterio.open(crop_path) as src:
        img = src.read().squeeze()
        meta = src.meta.copy()
    # Update metadata for a single band float32 output and add compression options.
    meta.update(
        {
            "count": 1,
            "dtype": "float32",
            "compress": "lzw",  # Use LZW compression
            "predictor": 3,  # Predictor for floating-point data
        }
    )

    # Upsample DSM; returns the DSM, its UTM coordinate grid, and metadata.
    dsm_for_projection, xy_utm_upscaled, _ = misc.upsample_dsm(
        dsm_path, UPSCALE_DSM_FACTOR
    )

    # Load RPC from JSON metadata.
    rpc = misc.rpc_from_json(root_path)

    # Transform DSM UTM coordinates to lat/lon.
    transformer = Transformer.from_crs(
        f"epsg:326{UTM_ZONES[city]}", "epsg:4326", always_xy=True
    )
    lon, lat = transformer.transform(xy_utm_upscaled[0], xy_utm_upscaled[1])

    # Project the DSM coordinates to image space using the RPC.
    x_img, y_img = rpc.projection(
        lon.flatten(), lat.flatten(), dsm_for_projection.flatten()
    )
    x_img = np.round(x_img).astype(int)
    y_img = np.round(y_img).astype(int)
    dsm_flat = dsm_for_projection.flatten()

    # Filter DSM points to those within the image bounds.
    valid = (
        (x_img >= 0) & (x_img < img.shape[1]) & (y_img >= 0) & (y_img < img.shape[0])
    )
    x_img = x_img[valid]
    y_img = y_img[valid]
    dsm_flat = dsm_flat[valid]

    # Compute a flat index for each valid image coordinate.
    index = y_img * img.shape[1] + x_img
    tensor_index = torch.tensor(index)
    tensor_dsm = torch.tensor(dsm_flat)

    # Identify pixels that receive only one DSM value.
    unique_index, unique_counts = np.unique(index, return_counts=True)
    unique_index = unique_index[unique_counts == 1]
    unique_mask = np.isin(index, unique_index)

    # Initialize the projected DSM image with NaNs.
    projected_dsm = np.full_like(img, np.nan, dtype=np.float32)

    # --- Unambiguous pixels: assign DSM values directly.
    unambiguous_x = x_img[unique_mask]
    unambiguous_y = y_img[unique_mask]
    unambiguous_dsm = dsm_flat[unique_mask]
    projected_dsm[unambiguous_y, unambiguous_x] = unambiguous_dsm

    # --- Ambiguous pixels: use scatter_max on the DSM values.
    ambiguous_index = tensor_index[~unique_mask]
    ambiguous_dsm = tensor_dsm[~unique_mask]

    # Compute maximum DSM for each ambiguous pixel using GPU scatter_max.
    # The output z_buffer will have a length equal to (max(ambiguous_index)+1)
    z_buffer, _ = scatter_max(ambiguous_dsm.to("cuda"), ambiguous_index.to("cuda"))
    z_buffer = z_buffer.cpu()  # maximum DSM value per flattened pixel index

    # Retrieve the pixel coordinates for the ambiguous DSM values.
    ambiguous_x = x_img[~unique_mask]
    ambiguous_y = y_img[~unique_mask]
    # Compute the flattened indices for these ambiguous pixels.
    ambiguous_pixel_indices = ambiguous_y * img.shape[1] + ambiguous_x

    # Get the maximum DSM values from the scatter result.
    max_dsm = z_buffer[ambiguous_pixel_indices]

    # Update the DSM projection image for ambiguous pixels.
    projected_dsm[ambiguous_y, ambiguous_x] = max_dsm

    # Save the projected DSM as a GeoTIFF preserving float data and NaNs.
    with rasterio.open(dsm_output_fname, "w", **meta) as dst:
        dst.write(projected_dsm, 1)

    print(f"Projected DSM saved for {aoi}_{crop_number} in {time() - t0:.2f} seconds")


def safe_project_dsm_single_crop(args):
    try:
        project_dsm_single_crop(args)
    except Exception as e:
        print(f"Error projecting DSM for crop {args}: {e}")


def project_dsm_parallel(dataset_dir: str):
    """
    Process all AOI crops in parallel.

    This function reads the AOI list from 'curated_aois_v3.csv' and, for each AOI,
    processes all crops found in the 'root_dir_ba' folder.
    """
    args_list = []
    # aoi_df = pd.read_csv(os.path.join(dataset_dir, "curated_aois_v3.csv"))
    aoi_df = pd.read_csv("curated_aois_v3.csv")
    for _, row in aoi_df.iterrows():
        aoi = row["aoi_name"]
        root_dir = os.path.join(dataset_dir, "root_dir_ba", aoi)
        if not os.path.exists(root_dir):
            continue
        crop_files = os.listdir(root_dir)
        # Assumes crop number is the second-to-last underscore-separated token in the filename.
        crop_numbers = [
            int(f.split("_")[-2]) for f in crop_files if f.endswith(".json")
        ]
        args_list.extend([(aoi, dataset_dir, crop) for crop in crop_numbers])

    from multiprocessing import Pool, cpu_count

    with Pool(processes=cpu_count() // 4) as pool:
        for _ in tqdm(
            pool.imap_unordered(safe_project_dsm_single_crop, args_list),
            total=len(args_list),
        ):
            pass


def project_all_dsm_dfc(dataset_dir: str):
    """
    Projects all DSMs from DFC 2019 dataset
    """
    rpc_dir = os.path.join(dataset_dir, "root_dir")
    oma_dir = os.path.join(dataset_dir, "Track3-RGB-2")
    jax_dir = os.path.join(dataset_dir, "Track3-RGB-1")
    dsm_dir = os.path.join(dataset_dir, "Track3-Truth-no-trees")
    out_dir = os.path.join(dataset_dir, "projected_dsm_no_trees")
    for rpc_name in tqdm(os.listdir(rpc_dir)):
        # rpc_name are like JAX_467_007_RGB.json; OMA_332_027_RGB.json
        # AOI name is first two tokens: JAX_467, OMA_332
        if not rpc_name.endswith(".json"):
            continue
        aoi = "_".join(rpc_name.split("_")[:2])
        city = aoi.split("_")[0]
        crop_number = rpc_name.split("_")[2]
        if city == "OMA":
            img_path = os.path.join(oma_dir, f"{aoi}_{crop_number}_RGB.tif")
        elif city == "JAX":
            img_path = os.path.join(jax_dir, f"{aoi}_{crop_number}_RGB.tif")

        rpc_path = os.path.join(rpc_dir, rpc_name)
        dsm_path = os.path.join(dsm_dir, f"{aoi}_DSM.tif")
        projected_dsm_path = os.path.join(out_dir, f"{aoi}_{crop_number}_DSM.tif")
        _project_dsm(img_path, rpc_path, dsm_path, projected_dsm_path)
        # cls_path = dsm_path.replace("DSM.tif", "CLS.tif")
        # if os.path.exists(cls_path):
        #     projected_cls_path = projected_dsm_path.replace("DSM.tif", "CLS.tif")
        #     _project_dsm(img_path, rpc_path, cls_path, projected_cls_path)


def project_dsm_sequential(dataset_dir: str):
    """
    Process all AOI crops sequentially.

    This function reads the AOI list from 'curated_aois_v3.csv' and, for each AOI,"
    """

    aoi_df = pd.read_csv(os.path.join("curated_aois_v3.csv"))
    for _, row in tqdm(aoi_df.iterrows(), total=len(aoi_df), desc="Processing AOIs"):
        aoi = row["aoi_name"]
        root_dir = os.path.join(dataset_dir, "root_dir_ba", aoi)
        if not os.path.exists(root_dir):
            continue
        crop_files = os.listdir(root_dir)
        # Assumes crop number is the second-to-last underscore-separated token in the filename.
        crop_numbers = [
            int(f.split("_")[-2]) for f in crop_files if f.endswith(".json")
        ]
        for crop in crop_numbers:
            safe_project_dsm_single_crop((aoi, dataset_dir, crop))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Project DSM onto image plane for satellite image crops."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Single crop parser
    single_parser = subparsers.add_parser(
        "project_dsm", help="Project a single DSM crop"
    )
    single_parser.add_argument("aoi", type=str, help="AOI name")
    single_parser.add_argument("dataset_dir", type=str, help="Dataset directory path")
    single_parser.add_argument("crop_number", type=int, help="Crop number to process")

    # Parallel processing parser
    parallel_parser = subparsers.add_parser(
        "project_dsm_parallel", help="Project all DSM crops in parallel"
    )
    parallel_parser.add_argument("dataset_dir", type=str, help="Dataset directory path")

    # Sequential processing parser
    sequential_parser = subparsers.add_parser(
        "project_dsm_sequential", help="Project all DSM crops sequentially"
    )
    sequential_parser.add_argument(
        "dataset_dir", type=str, help="Dataset directory path"
    )

    # Standalone DFC processing parser
    dfc_parser = subparsers.add_parser(
        "project_all_dsm_dfc", help="Project all DSMs from DFC 2019 dataset"
    )
    dfc_parser.add_argument("dataset_dir", type=str, help="Dataset directory path")

    args = parser.parse_args()

    if args.command == "project_dsm":
        safe_project_dsm_single_crop((args.aoi, args.dataset_dir, args.crop_number))
    elif args.command == "project_dsm_parallel":
        project_dsm_parallel(args.dataset_dir)
    elif args.command == "project_dsm_sequential":
        project_dsm_sequential(args.dataset_dir)
    elif args.command == "project_all_dsm_dfc":
        project_all_dsm_dfc(args.dataset_dir)
