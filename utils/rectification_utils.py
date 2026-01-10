"""
Collection of utilities for stereo rectification with negative unipolar disparities.
"""

import os

import cv2
import iio
import numpy as np
import rasterio
import torch
from lightglue import DISK, LightGlue, SuperPoint, match_pair
from PIL import Image
from s2p import homography
from skimage import transform


def altitude_range_coarse(rpc, scale_factor=1):
    """
    Computes a coarse altitude range using the RPC informations only.

    Args:
        rpc: instance of the rpcm.RPCModel class
        scale_factor: factor by which the scale offset is multiplied

    Returns:
        the altitude validity range of the RPC.
    """
    m = rpc.alt_offset - scale_factor * rpc.alt_scale
    M = rpc.alt_offset + scale_factor * rpc.alt_scale
    return m, M


def generate_point_mesh(col_range, row_range, alt_range):
    """
    Generates image coordinates (col, row, alt) of 3D points located on the grid
    defined by col_range and row_range, at uniformly sampled altitudes defined
    by alt_range.
    Args:
        col_range: triplet (col_min, col_max, n_col), where n_col is the
            desired number of samples
        row_range: triplet (row_min, row_max, n_row)
        alt_range: triplet (alt_min, alt_max, n_alt)

    Returns:
        3 lists, containing the col, row and alt coordinates.
    """
    # input points in col, row, alt space
    cols, rows, alts = [
        np.linspace(v[0], v[1], v[2]) for v in [col_range, row_range, alt_range]
    ]

    # make it a kind of meshgrid (but with three components)
    # if cols, rows and alts are lists of length 5, then after this operation
    # they will be lists of length 5x5x5
    cols, rows, alts = (
        (cols + 0 * rows[:, np.newaxis] + 0 * alts[:, np.newaxis, np.newaxis]).reshape(
            -1
        ),
        (0 * cols + rows[:, np.newaxis] + 0 * alts[:, np.newaxis, np.newaxis]).reshape(
            -1
        ),
        (0 * cols + 0 * rows[:, np.newaxis] + alts[:, np.newaxis, np.newaxis]).reshape(
            -1
        ),
    )

    return cols, rows, alts


def ground_control_points(rpc, x, y, w, h, m, M, n):
    """
    Computes a set of ground control points (GCP), corresponding to RPC data.

    Args:
        rpc: instance of the rpcm.RPCModel class
        x, y, w, h: four integers defining a rectangular region of interest
            (ROI) in the image. (x, y) is the top-left corner, and (w, h) are
            the dimensions of the rectangle.
        m, M: minimal and maximal altitudes of the ground control points
        n: cube root of the desired number of ground control points.

    Returns:
        a list of world points, given by their geodetic (lon, lat, alt)
        coordinates.
    """
    # points will be sampled in [x, x+w] and [y, y+h]. To avoid always sampling
    # the same four corners with each value of n, we make these intervals a
    # little bit smaller, with a dependence on n.
    col_range = [x + (1.0 / (2 * n)) * w, x + ((2 * n - 1.0) / (2 * n)) * w, n]
    row_range = [y + (1.0 / (2 * n)) * h, y + ((2 * n - 1.0) / (2 * n)) * h, n]
    alt_range = [m, M, n]
    col, row, alt = generate_point_mesh(col_range, row_range, alt_range)
    lon, lat = rpc.localization(col, row, alt)
    return lon, lat, alt


def matches_from_rpc(rpc1, rpc2, x, y, w, h, n=10):
    """
    Uses RPC functions to generate matches between two images.

    Args:
        rpc1, rpc2: two instances of the rpcm.RPCModel class
        x, y, w, h: four integers defining a rectangular region of interest
            (ROI) in the first view. (x, y) is the top-left corner, and (w, h)
            are the dimensions of the rectangle. In the first view, the matches
            will be located in that ROI.
        n: cube root of the desired number of matches.

    Returns:
        an array of matches, one per line, expressed as x1, y1, x2, y2.
    """
    m, M = altitude_range_coarse(rpc1)
    lon, lat, alt = ground_control_points(rpc1, x, y, w, h, m, M, n)
    x1, y1 = rpc1.projection(lon, lat, alt)
    x2, y2 = rpc2.projection(lon, lat, alt)

    return np.vstack([x1, y1, x2, y2]).T


def _geometric_filter_F(keypoints_ref, keypoints_sec, match_indices, ransac_thr=0.5):
    """
    Filter LightGlue matches with a Fundamental matrix via RANSAC.
    keypoints_*: (N, 2) arrays of [x, y]
    match_indices: (M, 2) int array of [i_ref, i_sec]
    ransac_thr: reprojection error threshold in pixels (OpenCV units)
    """
    if match_indices is None or len(match_indices) < 8:
        return match_indices  # not enough points for a robust F

    pts_ref = keypoints_ref[match_indices[:, 0], :2].astype(np.float32)
    pts_sec = keypoints_sec[match_indices[:, 1], :2].astype(np.float32)

    # FM_RANSAC uses the 8-point algorithm with RANSAC. Threshold is in pixels.
    F, mask = cv2.findFundamentalMat(pts_ref, pts_sec, cv2.FM_RANSAC, ransac_thr)
    if mask is None:
        return None
    inliers = mask.ravel().astype(bool)
    return match_indices[inliers]


def compute_matches_lightglue(img_ref, img_sec, ransac_thr=0.5):
    """
    Returns Nx4 correspondences [x0, y0, x1, y1], filtered with F-matrix RANSAC.
    img_ref, img_sec: HxW or HxWx3 numpy arrays in [0,255] (uint8) or similar
    """
    extractor = DISK().eval()
    matcher = LightGlue(features="disk").eval()

    # ensure 3-channel as you had
    if img_ref.ndim == 2:
        img_ref = np.stack([img_ref] * 3, axis=-1)
    if img_sec.ndim == 2:
        img_sec = np.stack([img_sec] * 3, axis=-1)

    img_ref_t = torch.from_numpy(img_ref.astype(np.float32) / 255.0).permute(2, 0, 1)
    img_sec_t = torch.from_numpy(img_sec.astype(np.float32) / 255.0).permute(2, 0, 1)

    feats_ref, feats_sec, matches = match_pair(extractor, matcher, img_ref_t, img_sec_t)

    # LightGlue outputs
    match_indices = matches["matches"].cpu().numpy()  # (M, 2)
    keypoints_ref = feats_ref["keypoints"].cpu().numpy()  # (N0, 2) [x, y]
    keypoints_sec = feats_sec["keypoints"].cpu().numpy()  # (N1, 2) [x, y]

    # Geometric filtering
    match_indices = _geometric_filter_F(
        keypoints_ref, keypoints_sec, match_indices, ransac_thr
    )
    if match_indices is None or len(match_indices) == 0:
        return np.empty((0, 4), dtype=np.float32)

    # Build Nx4 correspondences [x0, y0, x1, y1]
    points_ref = keypoints_ref[match_indices[:, 0]]
    points_sec = keypoints_sec[match_indices[:, 1]]
    correspondences = np.hstack([points_ref, points_sec]).astype(np.float32)
    return correspondences


def affine_fundamental_matrix(matches):
    """
    Estimates the affine fundamental matrix given a set of point correspondences
    between two images.
    Arguments:
        matches: 2D array of size Nx4 containing a list of pairs of matching
            points. Each line is of the form x1, y1, x2, y2, where (x1, y1) is
            the point in the first view while (x2, y2) is the matching point in
            the second view.
    Returns:
        the estimated affine fundamental matrix, given by the Gold Standard
        algorithm, as described in Hartley & Zisserman book (see chap. 14).
    """
    # revert the order of points to fit H&Z convention (see algo 14.1)
    X = matches[:, [2, 3, 0, 1]]

    # compute the centroid
    N = len(X)
    XX = np.sum(X, axis=0) / N

    # compute the Nx4 matrix A
    A = X - np.tile(XX, (N, 1))

    # the solution is obtained as the singular vector corresponding to the
    # smallest singular value of matrix A. See Hartley and Zissermann for
    # details.
    # It is the last line of matrix V (because np.linalg.svd returns V^T)
    U, S, V = np.linalg.svd(A)
    N = V[-1, :]

    # extract values and build F
    F = np.zeros((3, 3))
    F[0, 2] = N[0]
    F[1, 2] = N[1]
    F[2, 0] = N[2]
    F[2, 1] = N[3]
    F[2, 2] = -np.dot(N, XX)

    return F


def rectifying_similarities_from_affine_fundamental_matrix(F):
    """
    Computes two similarities from an affine fundamental matrix.
    Args:
        F: 3x3 numpy array representing the input fundamental matrix
    Returns:
        S, S': two similarities such that, when used to resample the two images
            related by the fundamental matrix, the resampled images are
            stereo-rectified.
    """
    # check that the input matrix is an affine fundamental matrix
    assert np.shape(F) == (3, 3)
    assert np.linalg.matrix_rank(F) == 2
    np.testing.assert_allclose(F[:2, :2], np.zeros((2, 2)))

    # notations
    a = F[0, 2]
    b = F[1, 2]
    c = F[2, 0]
    d = F[2, 1]
    e = F[2, 2]

    # rotations
    r = np.sqrt(c * c + d * d)
    s = np.sqrt(a * a + b * b)
    R1 = 1 / r * np.array([[d, -c], [c, d]])
    R2 = 1 / s * np.array([[-b, a], [-a, -b]])

    # zoom and translation
    z = np.sqrt(r / s)
    t = 0.5 * e / np.sqrt(r * s)

    # output similarities
    S1 = np.zeros((3, 3))
    S1[0:2, 0:2] = z * R1
    S1[1, 2] = t
    S1[2, 2] = 1

    S2 = np.zeros((3, 3))
    S2[0:2, 0:2] = 1 / z * R2
    S2[1, 2] = -t
    S2[2, 2] = 1

    return S1, S2


def bounding_box2D(pts):
    """
    bounding box for the points pts
    """
    dim = len(pts[0])  # should be 2
    bb_min = [min([t[i] for t in pts]) for i in range(dim)]
    bb_max = [max([t[i] for t in pts]) for i in range(dim)]
    return bb_min[0], bb_min[1], bb_max[0] - bb_min[0], bb_max[1] - bb_min[1]


def matrix_translation(x, y):
    t = np.eye(3)
    t[0, 2] = x
    t[1, 2] = y
    return t


def rectification_homographies(matches, x, y, w, h):
    """
    Computes rectifying homographies from point matches for a given ROI.

    The affine fundamental matrix F is estimated with the gold-standard
    algorithm, then two rectifying similarities (rotation, zoom, translation)
    are computed directly from F.

    Args:
        matches: numpy array of shape (n, 4) containing a list of 2D point
            correspondences between the two images.
        x, y, w, h: four integers defining the rectangular ROI in the first
            image. (x, y) is the top-left corner, and (w, h) are the dimensions
            of the rectangle.
    Returns:
        S1, S2, F: three numpy arrays of shape (3, 3) representing the
        two rectifying similarities to be applied to the two images and the
        corresponding affine fundamental matrix.
    """
    # estimate the affine fundamental matrix with the Gold standard algorithm
    F = affine_fundamental_matrix(matches)

    # compute rectifying similarities
    S1, S2 = rectifying_similarities_from_affine_fundamental_matrix(F)

    # pull back top-left corner of the ROI to the origin (plus margin)
    pts = homography.points_apply_homography(
        S1, [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    )
    x0, y0 = bounding_box2D(pts)[:2]
    T = matrix_translation(-x0, -y0)
    return np.dot(T, S1), np.dot(T, S2), F


def register_horizontally_shear(matches, H1, H2):
    """
    Adjust rectifying homographies with tilt, shear and translation to reduce the disparity range.

    Args:
        matches: list of pairs of 2D points, stored as a Nx4 numpy array
        H1, H2: two homographies, stored as numpy 3x3 matrices

    Returns:
        H2: corrected homography H2

    The matches are provided in the original images coordinate system. By
    transforming these coordinates with the provided homographies, we obtain
    matches whose disparity is only along the x-axis.
    """
    # transform the matches according to the homographies
    p1 = homography.points_apply_homography(H1, matches[:, :2])
    x1 = p1[:, 0]
    y1 = p1[:, 1]
    p2 = homography.points_apply_homography(H2, matches[:, 2:])
    x2 = p2[:, 0]
    y2 = p2[:, 1]

    # we search the (a, b, c) vector that minimises \sum (x1 - (a*x2+b*y2+c))^2
    # it is a least squares minimisation problem
    A = np.column_stack((x2, y2, y2 * 0 + 1))
    a, b, c = np.linalg.lstsq(A, x1, rcond=None)[0]

    # correct H2 with the estimated tilt, shear and translation
    return np.dot(np.array([[a, b, c], [0, 1, 0], [0, 0, 1]]), H2)


def register_horizontally_translation(matches, H1, H2, flag="negative", t_margin=0):
    """
    Adjust rectifying homographies with a translation to modify the disparity range.

    Args:
        matches: list of pairs of 2D points, stored as a Nx4 numpy array
        H1, H2: two homographies, stored as numpy 3x3 matrices
        flag: option needed to control how to modify the disparity range:
            'center': move the barycenter of disparities of matches to zero
            'positive': make all the disparities positive
            'negative': make all the disparities negative. Required for
                Hirshmuller stereo (java)
        t_margin: additional margin to add to the disparity range

    Returns:
        H2: corrected homography H2

    The matches are provided in the original images coordinate system. By
    transforming these coordinates with the provided homographies, we obtain
    matches whose disparity is only along the x-axis. The second homography H2
    is corrected with a horizontal translation to obtain the desired property
    on the disparity range.
    """
    # transform the matches according to the homographies
    p1 = homography.points_apply_homography(H1, matches[:, :2])
    x1 = p1[:, 0]
    y1 = p1[:, 1]
    p2 = homography.points_apply_homography(H2, matches[:, 2:])
    x2 = p2[:, 0]
    y2 = p2[:, 1]

    # compute the disparity offset according to selected option
    t = 0
    if flag == "center":
        t = np.mean(x2 - x1)
    if flag == "positive":
        t = np.min(x2 - x1)
        # If disparity set to positive, the t_margin is the opposite as in the negative case
        t_margin *= -1
    if flag == "negative":
        t = np.max(x2 - x1)
        print(f"max disparity before translation: {t}")

    print(f"t margin: {t_margin}")
    # Add this to shift a bit more the disparity range so the stereo matcher works better (at least for stereoanywhere)
    t += t_margin  # add a margin to the disparity range
    # correct H2 with a translation
    return np.dot(matrix_translation(-t, 0), H2)


def warp_with_H(img, H, output_shape, flip=False, **warp_kwargs):
    """
    Warp an image with a homography H into a fixed output shape.

    Args:
        img : ndarray
        H   : 3x3 homography (source -> target)
        output_shape : (h, w)
        flip : bool, whether to apply a horizontal flip in the target coords
        **warp_kwargs : passed to skimage.transform.warp
            (cval, mode, order, preserve_range, etc.)
    Returns:
        warped image, final homography used
    """
    H = np.asarray(H, dtype=np.float64)

    if flip:
        h, w = output_shape
        H_flip = np.array([[-1, 0, w - 1], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        H = H_flip @ H

    warped = transform.warp(
        img,
        transform.ProjectiveTransform(H).inverse,
        output_shape=output_shape,
        **warp_kwargs,
    )
    return warped, H


def _get_center_and_scales(rpc):
    # Works with typical rpcm.RPCModel attribute names; adjust if yours differ.
    lon0 = getattr(rpc, "lon_offset", getattr(rpc, "lon0", 0.0))
    lat0 = getattr(rpc, "lat_offset", getattr(rpc, "lat0", 0.0))
    h0 = getattr(rpc, "height_offset", getattr(rpc, "h0", 0.0))
    slon = getattr(rpc, "lon_scale", getattr(rpc, "lon_scale", 1.0))
    slat = getattr(rpc, "lat_scale", getattr(rpc, "lat_scale", 1.0))
    sh = getattr(rpc, "height_scale", getattr(rpc, "h_scale", 1.0))
    return float(lon0), float(lat0), float(h0), float(slon), float(slat), float(sh)


def _angle(vec2):
    return np.arctan2(vec2[1], vec2[0])


def _snap_quarter_turns(angle_rad):
    ang_deg = np.degrees(angle_rad) % 360.0
    k = int(np.round(ang_deg / 90.0)) % 4
    return k


def _local_vectors_from_scales(rpc, lon, lat, h, dlon, dlat):
    u0, v0 = rpc.projection(lon, lat, h)
    uE, vE = rpc.projection(lon + dlon, lat, h)
    uN, vN = rpc.projection(lon, lat + dlat, h)
    vE_vec = np.array([uE - u0, vE - v0], dtype=float)
    vN_vec = np.array([uN - u0, vN - v0], dtype=float)
    return vE_vec, vN_vec


def suggest_quarter_rotation_from_rpc_scales(rpc_ref, rpc_sec, frac=0.02):
    """
    Suggest how many 90° CCW rotations to apply to the SECOND image to roughly
    align it with the FIRST, using RPC offsets/scales.

    Args:
        rpc_ref, rpc_sec: RPC objects with *.project(lon, lat, h) and
                          attributes lon_offset/lat_offset/height_offset and
                          lon_scale/lat_scale/height_scale (rpcm-compatible).
        frac (float): fraction of lon_scale/lat_scale to use as small step.

    Returns:
        k (int): quarter turns CCW to apply to SECOND image (0,1,2,3)
        rot_deg (int): k*90 degrees
        debug (dict): angles for inspection
    """
    lon1, lat1, h1, slon1, slat1, _ = _get_center_and_scales(rpc_ref)
    lon2, lat2, h2, slon2, slat2, _ = _get_center_and_scales(rpc_sec)

    # Use the average center; use a conservative small step from the two scales
    lon = 0.5 * (lon1 + lon2)
    lat = 0.5 * (lat1 + lat2)
    h = 0.5 * (h1 + h2)

    dlon = frac * 0.5 * (slon1 + slon2)
    dlat = frac * 0.5 * (slat1 + slat2)

    # Guard against pathological scales; fall back to tiny perturbations if needed
    if not np.isfinite(dlon) or dlon == 0.0:
        dlon = 1e-6
    if not np.isfinite(dlat) or dlat == 0.0:
        dlat = 1e-6

    _, vN1 = _local_vectors_from_scales(rpc_ref, lon, lat, h, dlon, dlat)
    _, vN2 = _local_vectors_from_scales(rpc_sec, lon, lat, h, dlon, dlat)

    theta1 = _angle(vN1)
    theta2 = _angle(vN2)
    dtheta = (theta2 - theta1) % (2 * np.pi)

    k = _snap_quarter_turns(dtheta)
    return (
        k,
        int(k * 90),
        {
            "theta_ref_deg": float(np.degrees(theta1) % 360.0),
            "theta_sec_deg": float(np.degrees(theta2) % 360.0),
            "delta_deg": float(np.degrees(dtheta) % 360.0),
            "dlon_used_deg": float(dlon),
            "dlat_used_deg": float(dlat),
        },
    )


def unrotate_points(points_rot, k, H, W):
    """
    Map points from the ROTATED image back to the ORIGINAL image frame.
    points_rot: (N, 2) array [[x_rot, y_rot], ...]
    k: number of 90° CCW turns used to rotate the image: {0,1,2,3}
    H, W: original image height, width
    """
    x = points_rot[:, 0]
    y = points_rot[:, 1]

    if k % 4 == 0:  # 0° (no rotation)
        x0, y0 = x, y
    elif k % 4 == 1:  # 90° CCW  (np.rot90(..., k=1))
        x0 = W - 1 - y
        y0 = x
    elif k % 4 == 2:  # 180°     (np.rot90(..., k=2))
        x0 = W - 1 - x
        y0 = H - 1 - y
    else:  # 270° CCW (90° CW)  (np.rot90(..., k=3))
        x0 = y
        y0 = H - 1 - x

    return np.stack([x0, y0], axis=1)


def disparity_grows_with_altitude(H1, H2, rpc1, rpc2, x_center, y_center, alt_ground):
    """
    Decide if disparity grows with altitude based on the given homographies and RPCs.

    Returns:
        True if disparity grows with altitude, False otherwise.
    """
    # ROI center
    lon, lat = rpc1.localization(x_center, y_center, alt_ground)
    alt_high = alt_ground + 50

    # project ground and high in image 1 and image 2
    x1_ground, y1_ground = rpc1.projection(lon, lat, alt_ground)
    x2_ground, y2_ground = rpc2.projection(lon, lat, alt_ground)
    x1_high, y1_high = rpc1.projection(lon, lat, alt_high)
    x2_high, y2_high = rpc2.projection(lon, lat, alt_high)

    # apply homographies to both points and compare x coords
    x1_ground_rectified = homography.points_apply_homography(
        H1, [[x1_ground, y1_ground]]
    )[0][0]
    x2_ground_rectified = homography.points_apply_homography(
        H2, [[x2_ground, y2_ground]]
    )[0][0]
    x1_high_rectified = homography.points_apply_homography(H1, [[x1_high, y1_high]])[0][
        0
    ]
    x2_high_rectified = homography.points_apply_homography(H2, [[x2_high, y2_high]])[0][
        0
    ]

    disp_ground = x1_ground_rectified - x2_ground_rectified
    disp_high = x1_high_rectified - x2_high_rectified

    if disp_ground < disp_high:
        print(
            f"Disparity grows with altitude (disp_ground < disp_high): {disp_ground} < {disp_high}"
        )
    else:
        print(
            f"Disparity does not grow with altitude (disp_ground >= disp_high): {disp_ground} >= {disp_high}"
        )
    return disp_ground < disp_high


def geometric_filtering(features_i, features_j, matches_ij, ransac_thr=0.3):
    """
    Given a series of pairwise matches, use OpenCV to fit a fundamental matrix using RANSAC to filter outliers
    The 7-point algorithm is used to derive the fundamental matrix
    https://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#findfundamentalmat
    Args:
        features_i: N[i]x132 array representing the N[i] keypoints from image i
        features_j: N[j]x132 array representing the N[j] keypoints from image j
        matches_ij: Mx2 array representing M matches between features_i and features_j
        ransac_thr (optional): float, RANSAC outlier rejection threshold
    Returns:
        matches_ij: filtered version of matches_ij (will contain same amount of rows or less)
    """
    kp_coords_i = features_i[matches_ij[:, 0], :2]
    kp_coords_j = features_j[matches_ij[:, 1], :2]
    if ransac_thr is None:
        F, mask = cv2.findFundamentalMat(kp_coords_i, kp_coords_j, cv2.FM_RANSAC)
    else:
        F, mask = cv2.findFundamentalMat(
            kp_coords_i, kp_coords_j, cv2.FM_RANSAC, ransac_thr
        )

    # mask = inliers_mask_from_fundamental_matrix(F, kp_coords_i, kp_coords_j, ransac_thr)
    matches_ij = matches_ij[mask.ravel().astype(bool), :] if mask is not None else None
    return matches_ij


def opencv_match_SIFT(
    features_i, features_j, dst_thr=0.8, ransac_thr=0.3, matcher="flann"
):
    """
    Match SIFT keypoints using OpenCV matchers
    Args:
        features_i: N[i]x132 array representing the N[i] keypoints from image i
        features_j: N[j]x132 array representing the N[j] keypoints from image j
        dst_thr (optional): float, threshold for SIFT distance ratio test
        ransac_thr (optional): float, threshold for RANSAC geometric filtering using the fundamental matrix
        matcher (optional): string, identifies the OpenCV matcher to use: either "flann" or "bruteforce"
    Returns:
        matches_ij: Mx2 array representing M matches. Each match is represented by two values (i, j)
                    which means that the i-th kp/row in s2p_features_i matches the j-th kp/row in s2p_features_j
        n_matches_after_ratio_test: integer, the number of matches left after the SIFT distance ratio test
        n_matches_after_geofilt: integer, the number of matches left after RANSAC filtering
    """

    descriptors_i = features_i[:, 4:].astype(np.float32)
    descriptors_j = features_j[:, 4:].astype(np.float32)
    if matcher == "bruteforce":
        # Bruteforce matcher
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors_i, descriptors_j, k=2)
    elif matcher == "flann":
        # FLANN matcher
        # from https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors_i, descriptors_j, k=2)
    else:
        raise ValueError(
            'ERROR: OpenCV matcher is not recognized ! Valid values are "flann" or "bruteforce"'
        )

    # Apply ratio test as in Lowe's paper
    matches_ij = np.array(
        [
            [m.queryIdx, m.trainIdx]
            for m, n in matches
            if m.distance < dst_thr * n.distance
        ]
    )
    n_matches_after_ratio_test = matches_ij.shape[0]

    # Geometric filtering using the Fundamental matrix
    if n_matches_after_ratio_test > 0:
        matches_ij = geometric_filtering(features_i, features_j, matches_ij, ransac_thr)
    else:
        # no matches were left after ratio test
        matches_ij = None
    n_matches_after_geofilt = 0 if matches_ij is None else matches_ij.shape[0]

    return matches_ij, n_matches_after_ratio_test, n_matches_after_geofilt


def opencv_detect_SIFT(input_image, input_mask=None):
    """
    Detect SIFT keypoints in a single input grayscale image using OpenCV
    Requirement: pip3 install opencv-contrib-python==3.4.0.12
    Documentation of opencv keypoint class: https://docs.opencv.org/3.4/d2/d29/classcv_1_1KeyPoint.html
    Args:
        input_image: numpy 2d array
        input_mask: numpy binary mask, to restrict the search of keypoints to a certain area,
                    parts of the mask with 0s are not explored
    Returns:
        features: Nx132 array, where N is the number of SIFT keypoints detected in image i
                  each row/keypoint is represented by 132 values:
                  (col, row, scale, orientation) in columns 0-3 and (sift_descriptor) in the following 128 columns
        n_kp: integer, number of keypoints detected
    """
    input_image = input_image.astype(np.uint8)
    input_mask = None if input_mask is None else input_mask.astype(np.uint8)

    sift = cv2.SIFT_create()  # cv2.xfeatures2d.SIFT_create() for older opencv versions
    kp, des = sift.detectAndCompute(input_image, input_mask)
    features = np.array([[*k.pt, k.size, k.angle, *d] for k, d in zip(kp, des)])

    return features


def compute_matches(img_ref, img_sec):
    features_ref = opencv_detect_SIFT(img_ref)
    features_sec = opencv_detect_SIFT(img_sec)
    matches_idx, _, _ = opencv_match_SIFT(features_ref, features_sec, 0.6)
    if matches_idx is None:
        return np.array([])
    matches = np.hstack(
        [features_ref[matches_idx[:, 0], :2], features_sec[matches_idx[:, 1], :2]]
    )
    return matches


def points_apply_homography(H, pts):
    """
    Applies an homography to a list of 2D points.
    Args:
        H: numpy array containing the 3x3 homography matrix
        pts: numpy array containing the list of 2D points, one per line
    Returns:
        a numpy array containing the list of transformed points, one per line
    """
    # if the list of points is not a numpy array, convert it
    if type(pts) == list:
        pts = np.array(pts)

    # convert the input points to homogeneous coordinates
    if len(pts[0]) < 2:
        raise ValueError(
            "The input must be a numpy array" "of 2D points, one point per line"
        )
    pts = np.hstack((pts[:, 0:2], pts[:, 0:1] * 0 + 1))

    # apply the transformation
    Hpts = (np.dot(H, pts.T)).T

    # normalize the homogeneous result and trim the extra dimension
    Hpts = Hpts * (1.0 / np.tile(Hpts[:, 2], (3, 1))).T
    return Hpts[:, 0:2]


def image_apply_homography(input_image, H, h, w):
    H_ = transform.ProjectiveTransform(matrix=H)
    output_image = transform.warp(
        input_image,
        H_.inverse,
        output_shape=(h, w),
        cval=np.nan,
        mode="constant",
        preserve_range=True,
        order=5,
    )
    return output_image


def pts_apply_homography(pts2d, H):
    tform = transform.ProjectiveTransform(matrix=H)
    return tform(pts2d)


def read_input_image(img_path):
    if img_path.endswith(".tif"):
        with rasterio.open(img_path, "r") as f:
            img = np.transpose(f.read(), (1, 2, 0))
    else:
        img = np.array(Image.open(img_path))
    return img.astype(np.float32)


def save_png(img_path, img):
    im = Image.fromarray(np.clip(np.uint8(img), 0, 255))
    im.save(img_path)


def save_tif(img_path, img):
    if len(img.shape) == 2:
        img = img[..., np.newaxis]
    # expected image shape is H*W*C
    profile = {}
    profile["dtype"] = img.dtype
    profile["height"] = img.shape[0]
    profile["width"] = img.shape[1]
    profile["count"] = img.shape[2]
    with rasterio.open(img_path, "w", **profile) as f:
        f.write(np.transpose(img, (2, 0, 1)))


def refine_rectiying_homographies(matches, H1, H2):
    # this function refines the rectifying homographies H1 and H2 to ensure unidirectional movement to the left

    pts1 = pts_apply_homography(matches[:, :2], H1)
    pts2 = pts_apply_homography(matches[:, 2:], H2)

    def func(p, x, y):
        y_ = p[0] * x[:, 0] + p[1] * x[:, 1] + p[2]
        return abs(y[:, 0] - y_)

    from scipy import optimize

    p0 = np.array([0.0, 0.0, 0.0])
    popt = optimize.leastsq(func, p0, args=(pts1, pts2))[0]
    a, b, c = popt
    H1 = np.array([[a, b, c], [0, 1, 0], [0, 0, 1]]) @ H1

    pts1 = pts_apply_homography(matches[:, :2], H1)
    pts2 = pts_apply_homography(matches[:, 2:], H2)
    disp_ = pts1 - pts2
    x0 = disp_[
        :, 0
    ].min()  # disp_[:, 0] must be positive in all matches to have all displacements in left direction
    margin = 20  # margin in pixels
    if x0 < 0:
        H1 = matrix_translation(-x0 + margin, 0.0) @ H1

    return H1, H2


def get_file_id(filename):
    """
    return what is left after removing directory and extension from a path
    """
    return os.path.splitext(os.path.basename(filename))[0]


def rectify_pair_custom(ref_path, sec_path, x, y, w, h, out_dir):
    # read input images
    img_ref = iio.read(ref_path)
    img_sec = iio.read(sec_path)
    # If images are colored, convert to grayscale
    if len(img_ref.shape) == 3 and img_ref.shape[2] == 3:
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_RGB2GRAY)
    if len(img_sec.shape) == 3 and img_sec.shape[2] == 3:
        img_sec = cv2.cvtColor(img_sec, cv2.COLOR_RGB2GRAY)
    # find sift matches
    matches = compute_matches(img_ref, img_sec)
    # compute rectifying homographies
    H1, H2, F = rectification_homographies(matches, x, y, w, h)
    # refine homographies and rectify
    H1, H2 = refine_rectiying_homographies(matches, H1, H2)
    rectified_ref = image_apply_homography(img_ref, H1, h, w)
    rectified_sec = image_apply_homography(img_sec, H2, h, w)
    # save intermediate files
    input_pair_id = get_file_id(ref_path)
    os.makedirs(os.path.join(out_dir, "rectified_left"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "rectified_right"), exist_ok=True)
    save_tif(
        os.path.join(out_dir, f"rectified_left/{input_pair_id}.tif"), rectified_ref
    )
    save_tif(
        os.path.join(out_dir, f"rectified_right/{input_pair_id}.tif"), rectified_sec
    )
    os.makedirs(os.path.join(out_dir, "H_left"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "H_right"), exist_ok=True)
    np.savetxt(os.path.join(out_dir, f"H_left/{input_pair_id}.txt"), H1, fmt="%12.6f")
    np.savetxt(os.path.join(out_dir, f"H_right/{input_pair_id}.txt"), H2, fmt="%12.6f")
