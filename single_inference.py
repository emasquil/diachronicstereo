"""
Run single-pair stereo inference and optional DSM triangulation.

Given a left/right image pair (PNG/JPG or GeoTIFF), this script runs MonSter,
StereoAnywhere, or FoundationStereo, handling optional horizontal flip, center
crop for very large images, and padding to multiples of 32. Results are written
to <output_dir>/<left_basename>[_flipped]/.

Outputs: left_image.png, right_image.png, disparity.iio, and optionally
ground_truth.iio + residual.iio. If --left_rpc, --right_rpc, --homographies,
and --dsm are provided, it triangulates a DSM, writes pred_dsm.tif, and saves
MAE reports against the GT DSM.
"""

import argparse
import os

import iio
import numpy as np
import rasterio
import torch
import torch.nn.functional as F

from utils import misc
import thirdparty


def read_disparity(path: str) -> np.ndarray:
    """Read disparity and undo KITTI ×256 encoding if present."""
    arr = iio.read(path)
    if arr.ndim == 3:
        arr = arr.squeeze(-1)
    return arr.astype(np.float32) / (256.0 if arr.max() > 256.0 else 1.0)


def read_image(path: str) -> torch.Tensor:
    """
    Load PNG/JPG **or** GeoTIFF and return [1, 3, H, W] float32 in [0 .. 1].

    • 3-band → RGB
    • 1-band (panchromatic) → replicate channel → grey RGB
    """
    img = iio.read(path)  # [H, W, C]
    # If there are NaNs, raise a Warning and replace them with 0.0
    if np.isnan(img).any():
        print(f"Warning: NaNs found in {path}. Replacing with 0.0")
        img = np.nan_to_num(img, nan=0.0)
    # If single channel but 3D, assume it's grayscale and remove the channel dimension
    if img.ndim == 3 and img.shape[2] == 1:
        img = img.squeeze(-1)  # remove the last dimension if it's 1
    # Dimension handling
    if img.ndim == 2:  # single channel (grayscale)
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    elif img.ndim == 3 and img.shape[2] > 3:  # more than 3 channels
        img = img[:, :, :3]
    # Pixel range normalization
    if img.max() == 255:
        img = img.astype(np.float32) / 255.0
    # elif img.max() > 1.0:
    else:
        img = (img - img.min()) / (img.max() - img.min())

    tensor = torch.from_numpy(img).permute(2, 0, 1).float()  # [3,H,W]
    return tensor.unsqueeze(0)  # [1,3,H,W]


def pad_to_multiple(x: torch.Tensor, multiple: int = 32):
    """Symmetric replicate-pad so H & W are divisible by *multiple*."""
    h, w = x.shape[-2:]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    pad = [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2]  # l r t b
    return F.pad(x, pad, mode="replicate"), pad


def unpad(x: torch.Tensor, pad):
    _, _, h, w = x.shape
    return x[..., pad[2] : h - pad[3], pad[0] : w - pad[1]]


def trim_border_disparity(
    disp: np.ndarray,
    H_left: np.ndarray | None = None,
    H_right: np.ndarray | None = None,
    k: int = 32,
):
    """
    Crop k pixels from each border of the LEFT rectified canvas (where disparity lives)
    and shift BOTH homographies so coordinates stay aligned:
        H' = T(-k, -k) @ H
    Returns: (disp_cropped, H_left', H_right')
    """
    h, w = disp.shape[:2]
    disp_cropped = disp[k : h - k, k : w - k]

    if H_left is None and H_right is None:
        return disp_cropped, None, None

    T = np.array(
        [[1.0, 0.0, -float(k)], [0.0, 1.0, -float(k)], [0.0, 0.0, 1.0]], dtype=float
    )

    H_left_new = (T @ H_left) if H_left is not None else None
    H_right_new = (T @ H_right) if H_right is not None else None
    return disp_cropped, H_left_new, H_right_new


def main():
    parser = argparse.ArgumentParser(
        description="Run inference over a single stereo pair using different stereo models."
    )
    parser.add_argument(
        "--stereo_ckpt", type=str, required=True, help="Path to the stereo checkpoint"
    )
    parser.add_argument(
        "--depth_anything_v2_path",
        type=str,
        required=False,
        help="Path to Depth-Anything V2",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run the model on (e.g., 'cuda:0', 'cpu')",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="monster",
        choices=["monster", "stereoanywhere", "foundationstereo"],
        help="Model to use for inference",
    )
    parser.add_argument(
        "--left_image", type=str, required=True, help="Path to the left image"
    )
    parser.add_argument(
        "--right_image", type=str, required=True, help="Path to the right image"
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        required=False,
        help="Path to the ground truth disparity map (optional)",
    )
    parser.add_argument(
        "--left_rpc",
        type=str,
        required=False,
        help="Path to the left image RPC file for triangulating DSM",
    )
    parser.add_argument(
        "--right_rpc",
        type=str,
        required=False,
        help="Path to the right image RPC file for triangulating DSM",
    )
    parser.add_argument(
        "--dsm",
        type=str,
        required=False,
        help="Path to the ground truth DSM file for evaluating triangulated DSM",
    )
    parser.add_argument(
        "--homographies",
        type=str,
        required=False,
        help="Path to the homographies npz file for triangulating DSM",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory where results will be saved",
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        help="If set, flip the input images horizontally and swap left-right",
    )

    args = parser.parse_args()

    if args.model == "monster":
        stereo_model = thirdparty.build_monster(
            monster_ckpt=args.stereo_ckpt,
            depth_anything_v2_path=args.depth_anything_v2_path,
            device=args.device,
        )
        stereo_model.eval()
    elif args.model == "stereoanywhere":
        stereo_model, depth_model = thirdparty.build_stereoanywhere(
            stereo_ckpt=args.stereo_ckpt,
            depth_anything_v2_path=args.depth_anything_v2_path,
            device=args.device,
        )
        stereo_model.eval()
        depth_model.eval()
    elif args.model == "foundationstereo":
        stereo_model = thirdparty.build_foundation_stereo(
            foundation_ckpt=args.stereo_ckpt,
            device=args.device,
        )
        stereo_model.eval()

    # Get as a sub output dir the name of the rectified left image without extension
    subdir = os.path.splitext(os.path.basename(args.left_image))[0]
    if args.flip:
        subdir += "_flipped"
    os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)

    # Read images
    left_image = read_image(args.left_image).to(args.device)
    right_image = read_image(args.right_image).to(args.device)

    # If image is too large, just take a center crop
    center_crop = False
    orig_H, orig_W = left_image.shape[-2:]

    if orig_H > 1024 or orig_W > 1024:
        center_crop = True

        # Target size for each axis = min(original, 1024)
        new_H = min(orig_H, 1024)
        new_W = min(orig_W, 1024)

        # Compute crop offsets (integer centre crop)
        h0 = (orig_H - new_H) // 2
        w0 = (orig_W - new_W) // 2
        h1 = h0 + new_H
        w1 = w0 + new_W

        # Apply the same crop to both images
        left_image = left_image[..., h0:h1, w0:w1]
        right_image = right_image[..., h0:h1, w0:w1]

    if args.flip:
        left_image = torch.flip(left_image, dims=[3])
        right_image = torch.flip(right_image, dims=[3])
        left_image, right_image = right_image, left_image  # swap

    if args.model == "monster":
        left_image *= 255  # MonSter model expects pixel values in [0, 255]
        right_image *= 255  # MonSter model expects pixel values in [0, 255]
        # MonSter model inference and preprocessing
        left_image, pad = pad_to_multiple(left_image)  # [1, 3, H, W]
        right_image, _ = pad_to_multiple(right_image)  # [1, 3, H, W]
        with torch.no_grad():
            disparity = stereo_model(left_image, right_image, iters=32, test_mode=True)
            disparity = unpad(disparity, pad).squeeze([0, 1])
            disparity = disparity.cpu().numpy()
        left_image = unpad(left_image, pad)[0].cpu().numpy()
        right_image = unpad(right_image, pad)[0].cpu().numpy()
        left_image = (left_image).astype(np.uint8)
        right_image = (right_image).astype(np.uint8)
        left_image = np.transpose(left_image, (1, 2, 0))  # [H, W, C]
        right_image = np.transpose(right_image, (1, 2, 0))

    elif args.model == "stereoanywhere":
        # First run monocular depth estimation, doesn't require padding
        with torch.no_grad():
            cat = torch.cat([left_image, right_image], dim=0)  # [2, 3, H, W]
            mono_depths = depth_model.infer_image(
                cat, input_size_width=cat.shape[-1], input_size_height=cat.shape[-2]
            )  # [2, 1, H, W]
            # normalize per pair 0-1
            mono_depths = (mono_depths - mono_depths.min()) / (
                mono_depths.max() - mono_depths.min()
            )
            mono_left, mono_right = mono_depths[0:1], mono_depths[1:2]
        # Pad everything to multiple of 32
        left_image, pad_left = pad_to_multiple(left_image)  # [1, 3, H, W]
        right_image, _ = pad_to_multiple(right_image)  # [1, 3, H, W]
        mono_left, _ = pad_to_multiple(mono_left)
        mono_right, _ = pad_to_multiple(mono_right)
        # Stereo inference
        with torch.no_grad():
            disparity, _ = stereo_model(
                left_image,
                right_image,
                mono_left,
                mono_right,
                test_mode=True,
                iters=stereo_model.args.iters,
            )
            disparity = -disparity
            disparity = unpad(disparity, pad_left).squeeze([0, 1]).cpu().numpy()
        left_image = unpad(left_image, pad_left)[0].cpu().numpy()
        right_image = unpad(right_image, pad_left)[0].cpu().numpy()

        left_image = (left_image * 255.0).astype(np.uint8)
        right_image = (right_image * 255.0).astype(np.uint8)
        left_image = np.transpose(left_image, (1, 2, 0))
        right_image = np.transpose(right_image, (1, 2, 0))

    elif args.model == "foundationstereo":
        # Foundation stereo requires input in 0 255 range
        left_image = (left_image * 255.0).float()
        right_image = (right_image * 255.0).float()

        # Pad to multiple of 32
        padder = thirdparty.FsInputPadder(
            left_image.shape, divis_by=32, force_square=False
        )
        left_image, right_image = padder.pad(left_image, right_image)

        # Inference
        with torch.no_grad():
            with torch.cuda.amp.autocast(True):
                disparity = stereo_model.run_hierachical(
                    left_image, right_image, iters=32, test_mode=True, small_ratio=0.5
                )
                # disp = stereo_model(left_image, right_image, iters=32, test_mode=True)
        disparity = padder.unpad(disparity).squeeze(0).cpu().numpy()
        left_image = padder.unpad(left_image)[0].cpu().numpy()
        right_image = padder.unpad(right_image)[0].cpu().numpy()
        left_image = (left_image).astype(np.uint8)
        right_image = (right_image).astype(np.uint8)
        left_image = np.transpose(left_image, (1, 2, 0))
        right_image = np.transpose(right_image, (1, 2, 0))

    pred_dsm = None
    # If we have a DSM and RPCs, we can evaluate the triangulated DSM
    if (
        args.dsm is not None
        and args.left_rpc is not None
        and args.right_rpc is not None
        and args.homographies is not None
    ):
        homographies = np.load(args.homographies)
        hleft = homographies["Hleft"]
        hright = homographies["Hright"]
        left_rpc = misc.rpc_from_json(args.left_rpc)
        right_rpc = misc.rpc_from_json(args.right_rpc)
        with rasterio.open(args.dsm) as src:
            dsm_meta = src.meta.copy()
            dsm = src.read(1).astype(np.float32)
            dsm_shape = dsm.shape
        # If center crop, we need to apply that crop to the homographies as well
        if center_crop:
            start_h = (orig_H - left_image.shape[0]) // 2
            start_w = (orig_W - left_image.shape[1]) // 2
            T_crop = np.array(
                [[1, 0, -start_w], [0, 1, -start_h], [0, 0, 1]], dtype=float
            )
            hleft = T_crop @ hleft
            hright = T_crop @ hright

        # Remove borders to account for "neural semantic aperture problem"
        disparity, hleft, hright = trim_border_disparity(
            disparity,
            hleft,
            hright,
        )

        # First compute an altitude image from the disparity
        altitude_image = misc.altitude_image_from_disparity_vectorized(
            disparity,
            left_rpc,
            right_rpc,
            hleft,
            hright,
        )
        pred_dsm, pred_meta = misc.rectified_altitude_to_dsm(
            altitude_image, left_rpc, hleft, dsm_shape, dsm_meta
        )

    # Save results
    iio.write(os.path.join(args.output_dir, subdir, "left_image.png"), left_image)
    iio.write(os.path.join(args.output_dir, subdir, "right_image.png"), right_image)
    iio.write(os.path.join(args.output_dir, subdir, "disparity.iio"), disparity)
    if args.ground_truth:
        ground_truth = read_disparity(args.ground_truth)
        if center_crop:
            ground_truth = ground_truth[h0:h1, w0:w1]
        residual = np.abs(disparity - ground_truth)
        residual[ground_truth <= 0] = 0  # Ignore invalid pixels
        iio.write(os.path.join(args.output_dir, subdir, "residual.iio"), residual)
        iio.write(
            os.path.join(args.output_dir, subdir, "ground_truth.iio"), ground_truth
        )
    if pred_dsm is not None:
        with rasterio.open(
            os.path.join(args.output_dir, subdir, "pred_dsm.tif"),
            "w",
            driver="GTiff",
            height=pred_dsm.shape[0],
            width=pred_dsm.shape[1],
            count=1,
            dtype=pred_meta["dtype"],
            crs=pred_meta["crs"],
            transform=pred_meta["transform"],
            nodata=pred_meta["nodata"],
        ) as dst:
            dst.write(pred_dsm, 1)
        mae = misc.align_dsm_with_gt(
            args.dsm,
            os.path.join(args.output_dir, subdir, "pred_dsm.tif"),
            filter_foliage_and_water=False,
        )
        # save mae as txt
        with open(os.path.join(args.output_dir, subdir, "dsm_mae.txt"), "w") as f:
            f.write(f"MAE: {mae}\n")
        mae_filtered = misc.align_dsm_with_gt(
            args.dsm,
            os.path.join(args.output_dir, subdir, "pred_dsm.tif"),
            filter_foliage_and_water=True,
        )
        with open(
            os.path.join(args.output_dir, subdir, "dsm_mae_filtered.txt"), "w"
        ) as f:
            f.write(f"MAE (filtered): {mae_filtered}\n")


if __name__ == "__main__":
    main()
