"""
Stereo rectification driver for DFC pairs using rectification_utils.

Supports rectifying one explicit pair, all pairs in an AOI, or multiple AOIs
(including optional diachronic sampling). Loads image/RPC pairs, crops a central
patch for SIFT/LightGlue matching, estimates rectification homographies, writes
rectified left/right images plus homography/disparity sidecars, and can use a
projected DSM to derive ground-truth disparities when available.
"""

import argparse
import csv
import json
import os
import re
import sys
from itertools import combinations
from pathlib import Path
from random import sample, seed

import cv2
import iio
import numpy as np
import pandas as pd
from s2p import homography
from tqdm import tqdm
from scipy.ndimage import map_coordinates


sys.path.append("../")
from utils import misc  # noqa: E402
from utils import rectification_utils as ru  # noqa: E402

# MINIMUM_ALLOWED_DISPARITY = 10

# MINIMUM_ALLOWED_DISPARITY = 50
# DIACHRONIC_PAIRS_PER_AOI_OMA = 30
# DIACHRONIC_PAIRS_PER_AOI_JAX = 3
# SINCHRONIC_PAIRS_PER_AOI = 5
# MAXIMUM_SIFT_MATCHES = 40
# MAX_DIACHRONIC_TRIES_JAX = 20
# MAX_DIACHRONIC_TRIES_OMA = 60

MINIMUM_ALLOWED_DISPARITY = 50
DIACHRONIC_PAIRS_PER_AOI_OMA = 0
DIACHRONIC_PAIRS_PER_AOI_JAX = 0
SINCHRONIC_PAIRS_PER_AOI = 15
MAXIMUM_SIFT_MATCHES = 40
MAX_DIACHRONIC_TRIES_JAX = 20
MAX_DIACHRONIC_TRIES_OMA = 60


def compute_disparity_map_vectorized(
    rectified_left, raw_left, dsm_left, H_left, H_right, rpc_left, rpc_right
):
    """
    Vectorized version - processes all pixels at once
    """
    H_left_inv = np.linalg.inv(H_left)
    h, w = rectified_left.shape[:2]

    # Create coordinate grids for all pixels
    u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
    coords_rectified = np.stack([u_grid.ravel(), v_grid.ravel()], axis=1)

    # Transform all coordinates to raw left image space
    coords_raw = homography.points_apply_homography(H_left_inv, coords_rectified)
    u_raw, v_raw = coords_raw[:, 0], coords_raw[:, 1]

    # Create masks for valid coordinates
    valid_mask = (
        (u_raw >= 0)
        & (u_raw < raw_left.shape[1])
        & (v_raw >= 0)
        & (v_raw < raw_left.shape[0])
    )

    # Sample altitudes for all valid coordinates at once
    z_values = np.full(len(u_raw), np.nan)
    if np.any(valid_mask):
        z_valid = map_coordinates(
            dsm_left, [v_raw[valid_mask], u_raw[valid_mask]], order=1, mode="nearest"
        )
        z_values[valid_mask] = z_valid

    # Update valid mask to exclude invalid altitudes
    valid_mask &= np.isfinite(z_values)

    # Process only valid pixels
    if not np.any(valid_mask):
        return np.full((h, w), np.nan), np.full((h, w), np.nan)

    # Get coordinates and altitudes for valid pixels only
    u_raw_valid = u_raw[valid_mask]
    v_raw_valid = v_raw[valid_mask]
    z_valid = z_values[valid_mask]

    # Vectorized RPC operations
    lon_valid, lat_valid = rpc_left.localization(u_raw_valid, v_raw_valid, z_valid)
    u_right_valid, v_right_valid = rpc_right.projection(lon_valid, lat_valid, z_valid)

    # Transform to right rectified coordinates
    coords_right = np.stack([u_right_valid, v_right_valid], axis=1)
    coords_right_rectified = homography.points_apply_homography(H_right, coords_right)

    # Compute disparities
    u_left_valid = coords_rectified[valid_mask, 0]
    v_left_valid = coords_rectified[valid_mask, 1]

    disparity_valid = u_left_valid - coords_right_rectified[:, 0]
    vertical_disparity_valid = v_left_valid - coords_right_rectified[:, 1]

    # Fill results
    disparity_map = np.full((h, w), np.nan, dtype=np.float32)
    vertical_disparity_map = np.full((h, w), np.nan, dtype=np.float32)

    valid_indices = np.where(valid_mask)[0]
    valid_v = valid_indices // w
    valid_u = valid_indices % w

    disparity_map[valid_v, valid_u] = disparity_valid
    vertical_disparity_map[valid_v, valid_u] = vertical_disparity_valid

    return disparity_map, vertical_disparity_map


def process_image_pair(
    left_img_path,
    right_img_path,
    left_rpc_path,
    right_rpc_path,
    output_dir,
    left_dsm_path=None,
):
    """
    Process a single image pair for stereo rectification.

    Args:
        left_img_path (str): Path to the left image.
        right_img_path (str): Path to the right image.
        left_rpc_path (str): Path to the left RPC file.
        right_rpc_path (str): Path to the right RPC file.
        output_dir (str): Directory to save the rectified images and metadata.
        left_dsm_path (str, optional): Path to the projected DSM file. Defaults to None. If available,
                                    it will be used to compute ground truth disparities.

    Returns:
        None
    """

    # Create output directories
    os.makedirs(f"{output_dir}/L", exist_ok=True)
    os.makedirs(f"{output_dir}/R", exist_ok=True)
    os.makedirs(f"{output_dir}/disparity", exist_ok=True)
    os.makedirs(f"{output_dir}/homography", exist_ok=True)

    # Define output file paths
    # Get the name without the extension
    left_img_name = os.path.splitext(os.path.basename(left_img_path))[0]
    right_img_name = os.path.splitext(os.path.basename(right_img_path))[0]
    left_output_path = f"{output_dir}/L/{left_img_name}-{right_img_name}.iio"
    right_output_path = f"{output_dir}/R/{left_img_name}-{right_img_name}.iio"
    disparity_output_path = (
        f"{output_dir}/disparity/{left_img_name}-{right_img_name}.iio"
    )
    homography_output_path = (
        f"{output_dir}/homography/{left_img_name}-{right_img_name}.npz"
    )

    # If output files already exist, skip processing
    if os.path.exists(left_output_path) and os.path.exists(right_output_path):
        print(
            f"Rectified files for pair {left_img_name}-{right_img_name} already exist. Skipping."
        )
        return

    # Load input data
    left_image = iio.read(left_img_path)
    right_image = iio.read(right_img_path)
    # If image is single channel, squeeze the last dimension
    if left_image.ndim == 3 and left_image.shape[2] == 1:
        left_image = left_image.squeeze(axis=2)
    if right_image.ndim == 3 and right_image.shape[2] == 1:
        right_image = right_image.squeeze(axis=2)
    left_rpc = misc.rpc_from_json(left_rpc_path)
    right_rpc = misc.rpc_from_json(right_rpc_path)
    if left_dsm_path is not None:
        left_dsm = iio.read(left_dsm_path).squeeze(axis=-1)

    # First check if a rotation of the right image is needed for better LightGlue matching
    k, _, _ = ru.suggest_quarter_rotation_from_rpc_scales(left_rpc, right_rpc)

    # # Define AOI
    # x, y, w, h = 0, 0, left_image.shape[1], right_image.shape[0]
    # Define AOI: central 512x512 crop
    crop_size = 704
    x = left_image.shape[1] // 2 - crop_size // 2
    y = left_image.shape[0] // 2 - crop_size // 2
    w, h = crop_size, crop_size
    # if image smaller than crop size, use 512x512
    if left_image.shape[1] < crop_size or left_image.shape[0] < crop_size:
        crop_size = 512
        x = left_image.shape[1] // 2 - crop_size // 2
        y = left_image.shape[0] // 2 - crop_size // 2
        w, h = crop_size, crop_size

    # Compute RPC based matches
    rpc_matches = ru.matches_from_rpc(left_rpc, right_rpc, x, y, w, h)

    # Compute real matches
    right_image_rot = np.rot90(right_image, k=k)
    lg_matches = ru.compute_matches_lightglue(left_image, right_image_rot)

    if lg_matches is None or len(lg_matches) == 0:
        raise ValueError("Not enough matches found between the two images.")

    # Rotate back the matches to the original orientation
    right_H, right_W = right_image.shape[:2]
    lg_matches[:, 2:4] = ru.unrotate_points(
        lg_matches[:, 2:4], k=k, H=right_H, W=right_W
    )

    # Rectification using RPC matches
    H1, H2, _ = ru.rectification_homographies(rpc_matches, x, y, w, h)

    # Compute output image size
    roi = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    roi_after_H1 = homography.points_apply_homography(H1, roi)
    out_roi = np.round(ru.bounding_box2D(roi_after_H1))

    # Reduce disparity range with horizontal shear
    mean_altitude = np.mean(ru.altitude_range_coarse(left_rpc))
    lon, lat, alt = ru.ground_control_points(
        left_rpc, x, y, w, h, mean_altitude, mean_altitude, 4
    )
    x1, y1 = left_rpc.projection(lon, lat, alt)
    x2, y2 = right_rpc.projection(lon, lat, alt)
    m = np.vstack([x1, y1, x2, y2]).T
    m = np.vstack(
        list({tuple(row) for row in m})
    )  # remove duplicates due to no alt range
    H2 = ru.register_horizontally_shear(m, H1, H2)

    # Make disparities unipolar; first try negative, if not we try positive and flip images
    H2_original = H2.copy()
    H2 = ru.register_horizontally_translation(
        lg_matches, H1, H2_original, flag="negative", t_margin=MINIMUM_ALLOWED_DISPARITY
    )
    if not ru.disparity_grows_with_altitude(
        H1, H2, left_rpc, right_rpc, (x + w) // 2, (y + h) // 2, mean_altitude
    ):
        # print(
        #     "Disparity does not grow with altitude, rectifying with positive translation and flipping images"
        # )
        H2 = ru.register_horizontally_translation(
            lg_matches,
            H1,
            H2_original,
            flag="positive",
            t_margin=MINIMUM_ALLOWED_DISPARITY,
        )
        left_image_rect, H1 = ru.warp_with_H(
            left_image, H1, out_roi[-2:], flip=True, preserve_range=True
        )
        right_image_rect, H2 = ru.warp_with_H(
            right_image, H2, out_roi[-2:], flip=True, preserve_range=True
        )
        # print("flippity flip")
    else:
        left_image_rect, _ = ru.warp_with_H(
            left_image, H1, out_roi[-2:], flip=False, preserve_range=True
        )
        right_image_rect, _ = ru.warp_with_H(
            right_image, H2, out_roi[-2:], flip=False, preserve_range=True
        )

    # If DSM is available, compute ground truth disparity
    if left_dsm_path is not None:
        disparity, _ = compute_disparity_map_vectorized(
            left_image_rect, left_image, left_dsm, H1, H2, left_rpc, right_rpc
        )
        iio.write(disparity_output_path, disparity)

    # Save everything
    iio.write(left_output_path, left_image_rect.astype(left_image.dtype))
    iio.write(right_output_path, right_image_rect.astype(right_image.dtype))
    np.savez(homography_output_path, Hleft=H1, Hright=H2)


def process_image_pair_known_homographies(
    left_img_path,
    left_rpc_path,
    right_rpc_path,
    homographies_path,
    left_rectified_path,
    left_dsm_path,
    output_dir,
):
    """
    Only re-compute the disparity map given known homographies.
    """
    # Create output directories
    os.makedirs(f"{output_dir}/disparity_no_trees", exist_ok=True)

    # Define output file paths
    # Get the name without the extension
    left_img_name = os.path.splitext(os.path.basename(left_img_path))[0]
    right_img_name = os.path.splitext(os.path.basename(right_rpc_path))[0]
    disparity_output_path = (
        f"{output_dir}/disparity_no_trees/{left_img_name}-{right_img_name}.iio"
    )
    # If output files already exist, skip processing
    if os.path.exists(disparity_output_path):
        print(
            f"Disparity file for pair {left_img_name}-{right_img_name} already exist. Skipping."
        )
        return

    # Load input data
    left_image = iio.read(left_img_path)
    left_image_rect = iio.read(left_rectified_path)
    # If image is single channel, squeeze the last dimension
    if left_image.ndim == 3 and left_image.shape[2] == 1:
        left_image = left_image.squeeze(axis=2)
    left_rpc = misc.rpc_from_json(left_rpc_path)
    right_rpc = misc.rpc_from_json(right_rpc_path)
    left_dsm = iio.read(left_dsm_path).squeeze(axis=-1)
    homographies = np.load(homographies_path)
    H1 = homographies["Hleft"]
    H2 = homographies["Hright"]

    disparity, _ = compute_disparity_map_vectorized(
        left_image_rect, left_image, left_dsm, H1, H2, left_rpc, right_rpc
    )
    iio.write(disparity_output_path, disparity)


def rectify_20_satnerf_iarpa(dataset_dir: str):
    root_dir = os.path.join(dataset_dir, "root_dir", "rpcs_ba")
    crops_dir = os.path.join(dataset_dir, "crops")
    output_dir = os.path.join(dataset_dir, "stereo_pairs_ba")

    for aoi in tqdm(os.listdir(crops_dir)):
        # Pick randomly 20 pairs of images from the AOI
        images_subdir = os.path.join(crops_dir, aoi)
        output_subdir = os.path.join(output_dir, aoi)
        os.makedirs(output_subdir, exist_ok=True)
        all_images = [f for f in os.listdir(images_subdir) if f.endswith(".tif")]
        selected_pairs = sample(list(combinations(all_images, 2)), 20)
        for left_image, right_image in tqdm(selected_pairs):
            left_img_path = os.path.join(images_subdir, left_image)
            right_img_path = os.path.join(images_subdir, right_image)
            left_rpc_path = os.path.join(
                root_dir, aoi, left_image.replace(".tif", ".json")
            )
            right_rpc_path = os.path.join(
                root_dir, aoi, right_image.replace(".tif", ".json")
            )
            process_image_pair(
                left_img_path,
                right_img_path,
                left_rpc_path,
                right_rpc_path,
                output_subdir,
            )


def rectify_20_satnerf_dfc(dataset_dir: str):
    root_dir = os.path.join(dataset_dir, "root_dir", "crops_rpcs_ba_v2")
    crops_dir = os.path.join(dataset_dir, "Track3-RGB-crops")
    output_dir = os.path.join(dataset_dir, "stereo_pairs_ba")

    for aoi in tqdm(os.listdir(crops_dir)):
        # Pick randomly 20 pairs of images from the AOI
        images_subdir = os.path.join(crops_dir, aoi)
        os.makedirs(output_dir, exist_ok=True)
        all_images = [f for f in os.listdir(images_subdir) if f.endswith(".tif")]
        selected_pairs = sample(list(combinations(all_images, 2)), 20)
        for left_image, right_image in tqdm(selected_pairs):
            left_img_path = os.path.join(images_subdir, left_image)
            right_img_path = os.path.join(images_subdir, right_image)
            left_rpc_path = os.path.join(
                root_dir, aoi, left_image.replace(".tif", ".json")
            )
            right_rpc_path = os.path.join(
                root_dir, aoi, right_image.replace(".tif", ".json")
            )
            process_image_pair(
                left_img_path, right_img_path, left_rpc_path, right_rpc_path, output_dir
            )


def rectify_all_dfc(dfc_dir: str):
    def fname_to_id(name: str) -> int:
        # expects things like "OMA_130_022_RGB.tif" -> 22
        m = re.search(r"_([0-9]{1,3})_RGB", name)
        if not m:
            raise ValueError(f"Cannot parse ID from filename: {name}")
        return int(m.group(1).lstrip("0") or "0")

    root_dir = os.path.join(dfc_dir, "root_dir")
    JAX_crops = os.path.join(dfc_dir, "Track3-RGB-1")
    OMA_crops = os.path.join(dfc_dir, "Track3-RGB-2")
    dsm_dir = os.path.join(dfc_dir, "Track3-Truth")
    projected_dsm_dir = os.path.join(dfc_dir, "projected_dsm")
    jax_day_diff = pd.read_csv(os.path.join(dfc_dir, "JAX_date_differences.csv"))
    oma_day_diff = pd.read_csv(os.path.join(dfc_dir, "OMA_date_differences.csv"))
    output_dir = os.path.join(dfc_dir, "stereo_pairs_non_diachronic_only")
    os.makedirs(output_dir, exist_ok=True)

    diachronic_list = []
    sinchronic_list = []

    # Iterate over all AOIs
    aois = [f[:7] for f in os.listdir(dsm_dir) if f.endswith("_DSM.tif")]
    for aoi in tqdm(aois):
        city = aoi[:3]
        if city == "JAX":
            crops_dir = JAX_crops
            day_diff = jax_day_diff
            diachronic_pairs_per_aoi = DIACHRONIC_PAIRS_PER_AOI_JAX
            max_tries = MAX_DIACHRONIC_TRIES_JAX
        elif city == "OMA":
            crops_dir = OMA_crops
            day_diff = oma_day_diff
            diachronic_pairs_per_aoi = DIACHRONIC_PAIRS_PER_AOI_OMA
            max_tries = MAX_DIACHRONIC_TRIES_OMA
        all_possible_images = [f for f in os.listdir(crops_dir) if f.startswith(aoi)]
        id_set = {fname_to_id(n) for n in all_possible_images}
        id2name = {fname_to_id(n): n for n in all_possible_images}
        # Filter from day differences
        valid_images = day_diff[
            day_diff["id1"].isin(id_set) & day_diff["id2"].isin(id_set)
        ]
        # might_be_diachronic are valid pairs with more than 30 days difference
        # Map pairs back to full filenames
        might_be_diachronic = [
            (id2name[row["id1"]], id2name[row["id2"]])
            for _, row in valid_images.iterrows()
            if abs(row["days_difference"]) > 30
        ]

        sinchronich_pairs = [
            (id2name[row["id1"]], id2name[row["id2"]])
            for _, row in valid_images.iterrows()
            if abs(row["days_difference"]) <= 30
        ]
        # First rectify diachronic pairs
        # While we have less than DIACHRONIC_PAIRS_PER_AOI and we have still pairs to try
        # Randomly sample a pair from might_be_diachronic
        # Then compute sift matches, if sift matches are more than the maximum, we skip the pair
        diachronic_count = 0

        while (
            diachronic_count < diachronic_pairs_per_aoi
            and might_be_diachronic
            and max_tries > 0
        ):
            max_tries -= 1
            pair = sample(might_be_diachronic, 1)[0]
            might_be_diachronic.remove(pair)
            # left image is randomly id1 or id2
            if sample([0, 1], 1)[0] == 0:
                left_image = pair[0]
                right_image = pair[1]
            else:
                left_image = pair[1]
                right_image = pair[0]
            left_img_path = f"{crops_dir}/{left_image}"
            right_img_path = f"{crops_dir}/{right_image}"
            left_img = iio.read(left_img_path)
            right_img = iio.read(right_img_path)
            # If images are colored, convert to grayscale
            if left_img.ndim == 3 and left_img.shape[2] == 3:
                left_img = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
            if right_img.ndim == 3 and right_img.shape[2] == 3:
                right_img = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)
            matches = ru.compute_matches(left_img, right_img)
            print(
                f"Pair {left_image}-{right_image} has {matches.shape[0]} SIFT matches."
            )
            if matches.shape[0] >= MAXIMUM_SIFT_MATCHES:
                continue
            else:
                left_rpc_path = os.path.join(
                    root_dir, left_image.replace(".tif", ".json")
                )
                right_rpc_path = os.path.join(
                    root_dir, right_image.replace(".tif", ".json")
                )
                dsm_path = os.path.join(
                    projected_dsm_dir, left_image.replace("_RGB.tif", "_DSM.tif")
                )
                try:
                    process_image_pair(
                        left_img_path,
                        right_img_path,
                        left_rpc_path,
                        right_rpc_path,
                        output_dir,
                        left_dsm_path=dsm_path,
                    )
                except ValueError as e:
                    print(f"Skipping pair {left_image}-{right_image} due to error: {e}")
                    continue
                diachronic_count += 1
                diachronic_list.append([left_image, right_image, matches.shape[0]])
        # Then rectify sinchronic pairs
        sinchronic_count = 0
        while sinchronic_count < SINCHRONIC_PAIRS_PER_AOI and sinchronich_pairs:
            pair = sample(sinchronich_pairs, 1)[0]
            sinchronich_pairs.remove(pair)
            # left image is randomly id1 or id2
            if sample([0, 1], 1)[0] == 0:
                left_image = pair[0]
                right_image = pair[1]
            else:
                left_image = pair[1]
                right_image = pair[0]
            left_img_path = f"{crops_dir}/{left_image}"
            right_img_path = f"{crops_dir}/{right_image}"
            left_rpc_path = os.path.join(root_dir, left_image.replace(".tif", ".json"))
            right_rpc_path = os.path.join(
                root_dir, right_image.replace(".tif", ".json")
            )
            dsm_path = os.path.join(
                projected_dsm_dir, left_image.replace("_RGB.tif", "_DSM.tif")
            )
            try:
                process_image_pair(
                    left_img_path,
                    right_img_path,
                    left_rpc_path,
                    right_rpc_path,
                    output_dir,
                    left_dsm_path=dsm_path,
                )
            except ValueError as e:
                print(f"Skipping pair {left_image}-{right_image} due to error: {e}")
                continue
            sinchronic_count += 1
            sinchronic_list.append([left_image, right_image])

    # Save diachronic_list and sinchronic_list to csv
    with open(f"{output_dir}/diachronic_pairs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["left_image", "right_image", "num_sift_matches"])
        writer.writerows(diachronic_list)
    with open(f"{output_dir}/sinchronic_pairs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["left_image", "right_image"])
        writer.writerows(sinchronic_list)


def rectify_test_dfc(dfc_dir: str):
    def fname_to_id(name: str) -> int:
        # expects things like "OMA_130_022_RGB.tif" -> 22
        m = re.search(r"_([0-9]{1,3})_RGB", name)
        if not m:
            raise ValueError(f"Cannot parse ID from filename: {name}")
        return int(m.group(1).lstrip("0") or "0")

    root_dir = os.path.join(dfc_dir, "root_dir")
    OMA_crops = os.path.join(dfc_dir, "Track3-RGB-2")
    projected_dsm_dir = os.path.join(dfc_dir, "projected_dsm")
    oma_day_diff = pd.read_csv(os.path.join(dfc_dir, "OMA_date_differences.csv"))
    output_dir = os.path.join(dfc_dir, "test_stereo_pairs")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + "/diachronic", exist_ok=True)
    os.makedirs(output_dir + "/sinchronic", exist_ok=True)

    diachronic_list = []
    sinchronic_list = []

    # Iterate over test AOIs
    aois = ["OMA_247", "OMA_331", "OMA_084", "OMA_134", "OMA_230"]
    for aoi in tqdm(aois):
        crops_dir = OMA_crops
        all_possible_images = [f for f in os.listdir(crops_dir) if f.startswith(aoi)]
        id_set = {fname_to_id(n) for n in all_possible_images}
        id2name = {fname_to_id(n): n for n in all_possible_images}
        # Filter from day differences
        valid_images = oma_day_diff[
            oma_day_diff["id1"].isin(id_set) & oma_day_diff["id2"].isin(id_set)
        ]
        # might_be_diachronic are valid pairs with more than 30 days difference
        # Map pairs back to full filenames
        might_be_diachronic = [
            (id2name[row["id1"]], id2name[row["id2"]])
            for _, row in valid_images.iterrows()
            if abs(row["days_difference"]) > 30
        ]

        might_be_sinchronic = [
            (id2name[row["id1"]], id2name[row["id2"]])
            for _, row in valid_images.iterrows()
            if abs(row["days_difference"]) <= 30
        ]

        # First rectify sinchronic pairs
        sinchronic_count = 0
        while sinchronic_count < 20 and might_be_sinchronic:
            pair = sample(might_be_sinchronic, 1)[0]
            might_be_sinchronic.remove(pair)
            # left image is randomly id1 or id2
            if sample([0, 1], 1)[0] == 0:
                left_image = pair[0]
                right_image = pair[1]
            else:
                left_image = pair[1]
                right_image = pair[0]
            left_img_path = f"{crops_dir}/{left_image}"
            right_img_path = f"{crops_dir}/{right_image}"
            left_img = iio.read(left_img_path)
            right_img = iio.read(right_img_path)
            # If images are colored, convert to grayscale
            if left_img.ndim == 3 and left_img.shape[2] == 3:
                left_img = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
            if right_img.ndim == 3 and right_img.shape[2] == 3:
                right_img = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)
            matches = ru.compute_matches(left_img, right_img)
            print(
                f"Pair {left_image}-{right_image} has {matches.shape[0]} SIFT matches."
            )
            if matches.shape[0] < MAXIMUM_SIFT_MATCHES:
                continue
            else:
                left_rpc_path = os.path.join(
                    root_dir, left_image.replace(".tif", ".json")
                )
                right_rpc_path = os.path.join(
                    root_dir, right_image.replace(".tif", ".json")
                )
                dsm_path = os.path.join(
                    projected_dsm_dir, left_image.replace("_RGB.tif", "_DSM.tif")
                )
                try:
                    process_image_pair(
                        left_img_path,
                        right_img_path,
                        left_rpc_path,
                        right_rpc_path,
                        output_dir + "/sinchronic",
                        left_dsm_path=dsm_path,
                    )
                except ValueError as e:
                    print(f"Skipping pair {left_image}-{right_image} due to error: {e}")
                    continue
                sinchronic_count += 1
                sinchronic_list.append([left_image, right_image, matches.shape[0]])

        # Then rectify diachronic pairs
        # While we have less than 20 and we have still pairs to try
        # Randomly sample a pair from might_be_diachronic
        # Then compute sift matches, if sift matches are more than the maximum, we skip the pair
        diachronic_count = 0

        while diachronic_count < 20 and might_be_diachronic:
            pair = sample(might_be_diachronic, 1)[0]
            might_be_diachronic.remove(pair)
            # left image is randomly id1 or id2
            if sample([0, 1], 1)[0] == 0:
                left_image = pair[0]
                right_image = pair[1]
            else:
                left_image = pair[1]
                right_image = pair[0]
            left_img_path = f"{crops_dir}/{left_image}"
            right_img_path = f"{crops_dir}/{right_image}"
            left_img = iio.read(left_img_path)
            right_img = iio.read(right_img_path)
            # If images are colored, convert to grayscale
            if left_img.ndim == 3 and left_img.shape[2] == 3:
                left_img = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
            if right_img.ndim == 3 and right_img.shape[2] == 3:
                right_img = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)
            matches = ru.compute_matches(left_img, right_img)
            print(
                f"Pair {left_image}-{right_image} has {matches.shape[0]} SIFT matches."
            )
            if matches.shape[0] >= MAXIMUM_SIFT_MATCHES:
                continue
            else:
                left_rpc_path = os.path.join(
                    root_dir, left_image.replace(".tif", ".json")
                )
                right_rpc_path = os.path.join(
                    root_dir, right_image.replace(".tif", ".json")
                )
                dsm_path = os.path.join(
                    projected_dsm_dir, left_image.replace("_RGB.tif", "_DSM.tif")
                )
                try:
                    process_image_pair(
                        left_img_path,
                        right_img_path,
                        left_rpc_path,
                        right_rpc_path,
                        output_dir + "/diachronic",
                        left_dsm_path=dsm_path,
                    )
                except ValueError as e:
                    print(f"Skipping pair {left_image}-{right_image} due to error: {e}")
                    continue
                diachronic_count += 1
                diachronic_list.append([left_image, right_image, matches.shape[0]])

    # Save diachronic_list and sinchronic_list to csv
    with open(f"{output_dir}/diachronic_pairs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["left_image", "right_image", "num_sift_matches"])
        writer.writerows(diachronic_list)
    with open(f"{output_dir}/sinchronic_pairs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["left_image", "right_image"])
        writer.writerows(sinchronic_list)


def rectify_all_dfc_known_homographies(dfc_dir: str):
    root_dir = os.path.join(dfc_dir, "root_dir")
    JAX_crops = os.path.join(dfc_dir, "Track3-RGB-1")
    OMA_crops = os.path.join(dfc_dir, "Track3-RGB-2")
    dsm_dir = os.path.join(dfc_dir, "projected_dsm_no_trees")
    homographies_dir = os.path.join(dfc_dir, "stereo_pairs", "homography")
    output_dir = os.path.join(dfc_dir, "stereo_pairs")
    os.makedirs(output_dir, exist_ok=True)
    for pair_id in tqdm(os.listdir(homographies_dir)):
        if not pair_id.endswith(".npz"):
            continue
        homographies_path = os.path.join(homographies_dir, pair_id)
        left_rectified_path = homographies_path.replace("homography", "L").replace(
            ".npz", ".iio"
        )
        left_image, right_image = pair_id.replace(".npz", "").split("-")
        left_image += ".tif"
        right_image += ".tif"
        aoi = left_image[:7]
        city = aoi[:3]
        if city == "JAX":
            crops_dir = JAX_crops
        elif city == "OMA":
            crops_dir = OMA_crops
        left_img_path = f"{crops_dir}/{left_image}"
        left_rpc_path = os.path.join(root_dir, left_image.replace(".tif", ".json"))
        right_rpc_path = os.path.join(root_dir, right_image.replace(".tif", ".json"))
        dsm_path = os.path.join(dsm_dir, left_image.replace("_RGB.tif", "_DSM.tif"))
        process_image_pair_known_homographies(
            left_img_path,
            left_rpc_path,
            right_rpc_path,
            homographies_path=homographies_path,
            left_rectified_path=left_rectified_path,
            left_dsm_path=dsm_path,
            output_dir=output_dir,
        )


def main():
    parser = argparse.ArgumentParser(
        prog="stereo_rectification",
        description="Stereo rectification using ultimate rectification utils",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_single_pair = sub.add_parser(
        "single_pair", help="Rectify a single pair of images"
    )
    p_single_pair.add_argument("--left_image", type=str, required=True)
    p_single_pair.add_argument("--right_image", type=str, required=True)
    p_single_pair.add_argument("--left_rpc", type=str, required=True)
    p_single_pair.add_argument("--right_rpc", type=str, required=True)
    p_single_pair.add_argument("--output_dir", type=str, required=True)
    p_single_pair.add_argument("--left_dsm", type=str, default=None)

    p_rectify_all_dfc = sub.add_parser(
        "rectify_all_dfc",
        help="Rectify all diachronic and sinchronic pairs from DFC dataset",
    )
    p_rectify_all_dfc.add_argument("--dfc_dir", type=str, required=True)

    p_rectify_satnerf_iarpa = sub.add_parser(
        "rectify_satnerf_iarpa",
        help="Rectify 20 random pairs from each AOI in the SatNerf IARPA dataset",
    )
    p_rectify_satnerf_iarpa.add_argument("--dataset_dir", type=str, required=True)

    p_rectify_satnerf_dfc = sub.add_parser(
        "rectify_satnerf_dfc",
        help="Rectify 20 random pairs from each AOI in the SatNerf DFC dataset",
    )
    p_rectify_satnerf_dfc.add_argument("--dataset_dir", type=str, required=True)

    p_rectify_all_dfc_known_homographies = sub.add_parser(
        "rectify_all_dfc_known_homographies",
        help="Rectify all diachronic and sinchronic pairs from DFC dataset (using already computed homographies)",
    )
    p_rectify_all_dfc_known_homographies.add_argument(
        "--dataset_dir", type=str, required=True
    )

    p_rectify_test_dfc = sub.add_parser(
        "rectify_test_dfc",
        help="Rectify all diachronic and sinchronic pairs from DFC test dataset",
    )
    p_rectify_test_dfc.add_argument("--dfc_dir", type=str, required=True)

    args = parser.parse_args()

    if args.cmd == "single_pair":
        process_image_pair(
            args.left_image,
            args.right_image,
            args.left_rpc,
            args.right_rpc,
            args.output_dir,
            args.left_dsm,
        )
    elif args.cmd == "rectify_all_dfc":
        seed(42)
        rectify_all_dfc(args.dfc_dir)

    elif args.cmd == "rectify_satnerf_iarpa":
        seed(42)
        rectify_20_satnerf_iarpa(args.dataset_dir)

    elif args.cmd == "rectify_satnerf_dfc":
        seed(42)
        rectify_20_satnerf_dfc(args.dataset_dir)

    elif args.cmd == "rectify_all_dfc_known_homographies":
        seed(42)
        rectify_all_dfc_known_homographies(args.dataset_dir)

    elif args.cmd == "rectify_test_dfc":
        seed(42)
        rectify_test_dfc(args.dfc_dir)

    else:
        raise NotImplementedError(f"Command {args.cmd} not implemented yet.")


if __name__ == "__main__":
    main()
