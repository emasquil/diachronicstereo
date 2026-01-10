import json
import os

import iio
import numpy as np
import rasterio
import rpcm
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import rowcol
from scipy.stats import binned_statistic_2d

from . import dsmr


def rpc_from_json(json_path, return_dict=False):
    with open(json_path) as f:
        d = json.load(f)
    if return_dict:
        return d["rpc"]
    return rpcm.RPCModel(d["rpc"], dict_format="rpcm")


def upsample_dsm(dsm_path, upscale_factor=4):
    """Upsample DSM using pixel-as-area approach."""
    src = rasterio.open(dsm_path)
    meta = src.meta.copy()

    # Calculate new dimensions
    original_height, original_width = src.shape
    new_height = int(original_height * upscale_factor)
    new_width = int(original_width * upscale_factor)

    # Calculate the pixel size in the new resolution
    pixel_size_x = src.transform[0] / upscale_factor
    pixel_size_y = abs(src.transform[4]) / upscale_factor

    # Explicitly recalculate the transform to ensure alignment with the original grid
    transform_upscaled = rasterio.transform.from_origin(
        src.transform.c,  # x coordinate of the upper-left corner
        src.transform.f,  # y coordinate of the upper-left corner
        pixel_size_x,
        pixel_size_y,  # new pixel size
    )

    # Update metadata
    meta.update(
        {"height": new_height, "width": new_width, "transform": transform_upscaled}
    )

    # Resample DSM to increase resolution
    dsm_upscaled = src.read(
        out_shape=(1, new_height, new_width), resampling=Resampling.bilinear
    )
    dsm_upscaled = dsm_upscaled.squeeze()

    # Create UTM coordinates for the upscaled grid, adjusted for pixel centers
    x = np.linspace(
        transform_upscaled[2] + pixel_size_x / 2,
        transform_upscaled[2] + pixel_size_x * new_width - pixel_size_x / 2,
        new_width,
    )
    y = np.linspace(
        transform_upscaled[5] - pixel_size_y / 2,
        transform_upscaled[5] - pixel_size_y * new_height + pixel_size_y / 2,
        new_height,
    )

    xy_utm_upscaled = np.array(np.meshgrid(x, y))

    src.close()

    return dsm_upscaled, xy_utm_upscaled, meta


def crop_dsm(ref_dsm_path, in_dsm_path, out_dsm_path):
    """
    Crop and reproject a DSM to match a reference DSM's extent and CRS.

    Args:
        ref_dsm_path (str): Path to reference DSM that defines the target extent and CRS
        in_dsm_path (str): Path to input DSM to be cropped/reprojected
        out_dsm_path (str): Path where the resulting DSM will be saved
    """
    import os

    import rasterio
    from osgeo import gdal

    # Create temporary file for reprojected DSM if needed
    temp_reprojected = None

    try:
        # Open reference and input DSMs to check CRS
        ref_ds = gdal.Open(ref_dsm_path)
        in_ds = gdal.Open(in_dsm_path)

        ref_crs = ref_ds.GetProjection()
        in_crs = in_ds.GetProjection()

        input_path = in_dsm_path

        # If CRS different, reproject input to match reference
        if ref_crs != in_crs:
            temp_reprojected = os.path.join(
                os.path.dirname(out_dsm_path), "temp_reprojected.tif"
            )
            gdal.Warp(
                temp_reprojected, in_ds, dstSRS=ref_crs, resampleAlg=gdal.GRA_Bilinear
            )
            input_path = temp_reprojected

        # Get bounds from reference
        with rasterio.open(ref_dsm_path) as src:
            xoff, yoff = src.bounds.left, src.bounds.bottom
            xsize, ysize = src.width, src.height
            resolution = src.res[0]

        # Define projwin for gdal translate
        ulx = xoff
        uly = yoff + ysize * resolution
        lrx = xoff + xsize * resolution
        lry = yoff

        # Crop to reference extent
        ds = gdal.Translate(
            out_dsm_path,
            input_path,
            options=f"-projwin {ulx} {uly} {lrx} {lry} -tr {resolution} {resolution}",
        )
        ds = None

        assert os.path.exists(out_dsm_path), "Output file was not created"

    finally:
        # Clean up temporary file if it was created
        if temp_reprojected and os.path.exists(temp_reprojected):
            os.remove(temp_reprojected)
            os.remove(temp_reprojected + ".aux.xml")


def altitude_image_from_disparity_vectorized(
    disparity,
    left_rpc,
    right_rpc,
    hleft,
    hright,
    h0=0.0,
    h_step=1.0,
    tol=1e-7,
    max_iters=100,
):
    """
    This function computes an altitude image (each pixel is altitude in meters) from the disparity.
    Vectorized s2p Alg.2 over all valid rectified pixels.
    hleft/hright: homographies raw->rect (full-frame, shifts baked in).
    """
    H, W = disparity.shape
    alt = np.full((H, W), np.nan, dtype=float)

    ys, xs = np.where(np.isfinite(disparity))
    if len(xs) == 0:
        return alt

    xs = xs.astype(float)
    ys = ys.astype(float)
    xR = xs - disparity[ys.astype(int), xs.astype(int)].astype(float)
    yR = ys.copy()

    # ---- rectified -> raw (batch) ----
    HLi = np.linalg.inv(hleft)
    HRi = np.linalg.inv(hright)

    def unrectify(Hinv, x, y):
        hom = np.vstack([x, y, np.ones_like(x)])
        uvw = Hinv @ hom
        return uvw[0] / uvw[2], uvw[1] / uvw[2]

    xL_raw, yL_raw = unrectify(HLi, xs, ys)
    xR_raw, yR_raw = unrectify(HRi, xR, yR)

    # ---- iterative solve (vectorized over pixels) ----
    h = np.full(xs.shape, float(h0))
    done = np.zeros_like(h, dtype=bool)

    for _ in range(max_iters):
        lon0, lat0 = left_rpc.localization(xL_raw, yL_raw, h)  # arrays (N,)
        r0x, r0y = right_rpc.projection(lon0, lat0, h)  # raw-right coords

        h1 = h + h_step
        lon1, lat1 = left_rpc.localization(xL_raw, yL_raw, h1)
        r1x, r1y = right_rpc.projection(lon1, lat1, h1)

        tx = r1x - r0x
        ty = r1y - r0y
        vx = xR_raw - r0x
        vy = yR_raw - r0y

        denom = tx * tx + ty * ty
        good = denom > 1e-15

        h_inc = np.zeros_like(h)
        h_inc[good] = (tx[good] * vx[good] + ty[good] * vy[good]) / denom[good]

        step_small = np.abs(h_inc) < tol
        done |= step_small | (~good)

        upd = (~done) & good
        if np.any(upd):
            h[upd] = h[upd] + h_inc[upd] * h_step
        if np.all(done):
            break

    alt[ys.astype(int), xs.astype(int)] = h
    return alt


def rectified_altitude_to_dsm(
    alt_rect,  # HxW altitudes (meters) on LEFT *rectified* grid
    left_rpc,  # RPC with vectorized localization(u_raw, v_raw, h) -> (lon,lat)
    hleft,  # 3x3 homography: RAW -> RECTIFIED (shifts baked in)
    dsm_shape,  # (height, width) of target DSM grid
    dsm_meta,  # dict with at least {"transform", "crs", "nodata"}
    crop_origin=(0, 0),  # (cx, cy) top-left of rectified crop in full rectified frame
):
    """
    Project rectified altitude image to a DSM grid using the left image RPC.
    """
    tgt_h, tgt_w = dsm_shape
    transform = dsm_meta["transform"]
    crs = dsm_meta["crs"]
    # nodata = dsm_meta.get("nodata", -9999.0)
    nodata = np.nan

    H, W = alt_rect.shape
    # 1) rectified pixel grid (add crop origin if alt_rect is a crop)
    cx, cy = crop_origin
    xs, ys = np.meshgrid(np.arange(W, dtype=float) + cx, np.arange(H, dtype=float) + cy)

    # 2) rectified -> RAW via H^{-1} (vectorized)
    Hinv = np.linalg.inv(hleft)
    hom = np.stack([xs, ys, np.ones_like(xs)], axis=-1).reshape(-1, 3).T  # (3,N)
    uvw = Hinv @ hom
    u_raw = (uvw[0] / uvw[2]).reshape(H, W)
    v_raw = (uvw[1] / uvw[2]).reshape(H, W)

    # 3) Keep valid altitude pixels
    h = alt_rect.astype(float)
    m = np.isfinite(h)
    if not np.any(m):
        raise ValueError("No valid altitude pixels to rasterize.")
    u = u_raw[m]
    v = v_raw[m]
    h = h[m]

    # 4) Localize to lon/lat at per-pixel height (meters)
    lon, lat = left_rpc.localization(u, v, h)

    # 5) lon/lat -> target CRS (e.g., UTM from GT DSM meta)
    to_tgt = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    E, N = to_tgt.transform(lon, lat)

    # 6) Map to DSM cell indices and aggregate (max)
    rows, cols = rowcol(transform, E, N, op=np.floor)
    rows = rows.astype(int)
    cols = cols.astype(int)
    inside = (rows >= 0) & (rows < tgt_h) & (cols >= 0) & (cols < tgt_w)
    rows, cols, h = rows[inside], cols[inside], h[inside]
    if len(h) == 0:
        raise ValueError("All samples fell outside target DSM extent.")

    row_edges = np.arange(-0.5, tgt_h + 0.5, 1.0)
    col_edges = np.arange(-0.5, tgt_w + 0.5, 1.0)
    grid, _, _, _ = binned_statistic_2d(
        rows, cols, h, statistic="max", bins=[row_edges, col_edges]
    )

    pred_dsm = grid.astype("float32")
    pred_dsm[~np.isfinite(pred_dsm)] = nodata

    pred_meta = dsm_meta.copy()
    pred_meta.update(dtype="float32", count=1, nodata=nodata)

    return pred_dsm, pred_meta


def align_dsm_with_gt(
    gt_dsm_path, pred_dsm_path, filter_water=True, filter_foliage=True
):
    water_path = gt_dsm_path.replace("DSM.tif", "WATER.png")
    cls_path = gt_dsm_path.replace("DSM.tif", "CLS.tif")

    if filter_foliage or filter_water:
        # If CLS file exists, mask out foliage and water in GT DSM
        if os.path.exists(cls_path):
            gt_dsm_labels = iio.read(cls_path).squeeze(-1)
            gt_dsm_labels = np.nan_to_num(gt_dsm_labels, nan=9999).astype(np.uint8)

            with rasterio.open(gt_dsm_path) as src:
                gt_dsm = src.read(1)
                gt_meta = src.meta
            # If gt_dsm shape is different from gt_dsm_labels. resize gt_dsm_labels to match gt_dsm
            if gt_dsm.shape != gt_dsm_labels.shape:
                from skimage.transform import resize

                h, w = gt_dsm.shape
                gt_dsm_labels = resize(
                    gt_dsm_labels,
                    (h, w),
                    order=0,
                    preserve_range=True,
                    anti_aliasing=False,
                ).astype(gt_dsm_labels.dtype)
            if filter_foliage:
                gt_dsm[gt_dsm_labels == 5] = np.nan  # foliage
            if filter_water:
                gt_dsm[gt_dsm_labels == 9] = np.nan  # water
            # Save masked GT DSM to temp file
            temp_gt_dsm_path = gt_dsm_path.replace(
                ".tif", f"_masked_water_{filter_water}_foliage_{filter_foliage}.tif"
            )
            with rasterio.open(temp_gt_dsm_path, "w", **gt_meta) as dst:
                dst.write(gt_dsm, 1)
            gt_dsm_path = temp_gt_dsm_path
            # If WATER file exists, mask out water in GT DSM. Only for IARPA 003
            if os.path.exists(water_path) and filter_water:
                gt_dsm_water = iio.read(water_path).squeeze(-1)
                gt_dsm[gt_dsm_water == 0] = np.nan
                # Save masked GT DSM to temp file
                with rasterio.open(gt_dsm_path, "w", **gt_meta) as dst:
                    dst.write(gt_dsm, 1)

    # aligned_path = pred_dsm_path.replace(".tif", f"_aligned_masked_water_{filter_water}_foliage_{filter_foliage}.tif")
    # if filter_foliage or filter_water:
    #     aligned_path = aligned_path.replace(".tif", f"_masked_water_{filter_water}_foliage_{filter_foliage}.tif")

    # shift = dsmr.compute_shift(gt_dsm_path, pred_dsm_path, scaling=False)
    # dsmr.apply_shift(pred_dsm_path, aligned_path, *shift)
    # with rasterio.open(aligned_path) as src:
    #     aligned_dsm = src.read(1)
    # with rasterio.open(gt_dsm_path) as src:
    #     gt_dsm = src.read(1)
    # mae = np.nanmean(np.abs(gt_dsm - aligned_dsm))
    # # print(f"MAE between GT and aligned DSM: {mae}")
    # return mae

    # --- NEW: Crop and reproject pred_dsm to match GT DSM grid ---
    pred_cropped_path = pred_dsm_path.replace(".tif", "_cropped_to_gt.tif")
    crop_dsm(gt_dsm_path, pred_dsm_path, pred_cropped_path)

    # Now compute shift on comparable grids
    shift = dsmr.compute_shift(gt_dsm_path, pred_cropped_path, scaling=False)

    aligned_path = pred_dsm_path.replace(
        ".tif", f"_aligned_masked_water_{filter_water}_foliage_{filter_foliage}.tif"
    )
    dsmr.apply_shift(pred_cropped_path, aligned_path, *shift)

    with rasterio.open(aligned_path) as src:
        aligned_dsm = src.read(1)
    with rasterio.open(gt_dsm_path) as src:
        gt_dsm = src.read(1)

    # --- NEW: shape guard (ignores extra row/col safely) ---
    if gt_dsm.shape != aligned_dsm.shape:
        H = min(gt_dsm.shape[0], aligned_dsm.shape[0])
        W = min(gt_dsm.shape[1], aligned_dsm.shape[1])
        gt_dsm = gt_dsm[:H, :W]
        aligned_dsm = aligned_dsm[:H, :W]
    # -----------------------------------------------

    mae = np.nanmean(np.abs(gt_dsm - aligned_dsm))
    return mae
