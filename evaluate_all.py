"""
Batch-evaluate stereo models and DSM reconstruction for DFC/SatNeRF/IARPA layouts.

Workflow:
  - discover rectified stereo pairs via a dataset subcommand
  - run a selected model (MonSter, StereoAnywhere, FoundationStereo, or RAFT)
  - save per-pair outputs (left/right PNGs and disparity.iio)
  - if homographies + RPCs + GT DSM are available, triangulate a DSM and compute
    MAE metrics (raw, filtered, and no-water) against ground truth
  - append a row to output_dir/summary.csv

Summary CSV columns (created/appended at --output-dir/summary.csv):
  model, pair_id, aoi, left_path, right_path, homography_path,
  left_rpc_json, right_rpc_json, gt_dsm_path, pred_dsm_path,
  mae, mae_filtered, mae_no_water, status, error
"""

from __future__ import annotations

# --- MUST be the very first lines in train_monster.py ---
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
monster_root = root / "thirdparty" / "MonSter"
depth_anything_root = monster_root / "Depth-Anything-V2-list3"

for p in (monster_root, depth_anything_root):
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)
# -----------------------------------------


import argparse
import csv
import os
import random
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import iio
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import misc
import thirdparty

from thirdparty.FoundationStereo.fs_core.utils.utils import InputPadder as FsInputPadder

# --------------------------- Small I/O helpers ---------------------------


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_image_any(path: Path) -> torch.Tensor:
    """
    Load PNG/JPG/GeoTIFF → [1,3,H,W] float in [0,1].
    1-band → 3, >3 bands → first 3, NaNs→0. Map dynamic range to [0,1] if not 8-bit.
    """
    arr = iio.read(str(path))
    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=0.0)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr.squeeze(-1)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]
    if arr.max() == 255:
        arr = arr.astype(np.float32) / 255.0
    else:
        m, M = float(arr.min()), float(arr.max())
        arr = (arr - m) / (M - m + 1e-12)
    t = torch.from_numpy(arr).permute(2, 0, 1).float()
    return t.unsqueeze(0)


def pad_to_multiple(
    x: torch.Tensor, multiple: int = 32
) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    h, w = x.shape[-2:]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)  # left,right,top,bottom
    return F.pad(x, pad, mode="replicate"), pad


def unpad(x: torch.Tensor, pad: Tuple[int, int, int, int]) -> torch.Tensor:
    l, r, t, b = pad
    return x[..., t : x.shape[-2] - b, l : x.shape[-1] - r]


def center_crop_if_large(imgL: torch.Tensor, imgR: torch.Tensor, max_side: int = 1024):
    """
    If any side > max_side, center-crop both to <= max_side.
    Returns (Lc, Rc, (h0,h1,w0,w1) or None)
    """
    H, W = imgL.shape[-2:]
    if H <= max_side and W <= max_side:
        return imgL, imgR, None
    new_H = min(H, max_side)
    new_W = min(W, max_side)
    h0 = (H - new_H) // 2
    w0 = (W - new_W) // 2
    h1, w1 = h0 + new_H, w0 + new_W
    return imgL[..., h0:h1, w0:w1], imgR[..., h0:h1, w0:w1], (h0, h1, w0, w1)


def trim_border_disparity(
    disp: np.ndarray,
    H_left: Optional[np.ndarray],
    H_right: Optional[np.ndarray],
    k: int = 32,
):
    """
    Trim k pixels on each border of the left rectified canvas and shift both homographies:
      H' = T(-k,-k) @ H
    """
    h, w = disp.shape[:2]
    out = disp[k : h - k, k : w - k]
    if H_left is None and H_right is None:
        return out, None, None
    T = np.array([[1, 0, -float(k)], [0, 1, -float(k)], [0, 0, 1]], dtype=float)
    Hl = (T @ H_left) if H_left is not None else None
    Hr = (T @ H_right) if H_right is not None else None
    return out, Hl, Hr


# --------------------------- CSV summary helpers ---------------------------

SUMMARY_COLS = [
    "model",
    "pair_id",
    "aoi",
    "left_path",
    "right_path",
    "homography_path",
    "left_rpc_json",
    "right_rpc_json",
    "gt_dsm_path",
    "pred_dsm_path",
    "mae",
    "mae_filtered",
    "mae_no_water",
    "status",
    "error",
]


def _summary_path(out_dir: Path) -> Path:
    return out_dir / "summary.csv"


def _init_summary(out_dir: Path) -> None:
    p = _summary_path(out_dir)
    if not p.exists():
        with open(p, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=SUMMARY_COLS).writeheader()


def _append_summary(out_dir: Path, row: Dict) -> None:
    p = _summary_path(out_dir)
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        # ensure all keys exist as strings
        safe = {k: ("" if row.get(k) is None else row.get(k)) for k in SUMMARY_COLS}
        w.writerow(safe)


# --------------------------- Data model ---------------------------


@dataclass(frozen=True)
class PairSample:
    pair_id: str
    left_path: Path
    right_path: Path
    out_dir: Path
    homography_path: Optional[Path] = None
    left_rpc_json: Optional[Path] = None
    right_rpc_json: Optional[Path] = None
    gt_dsm_path: Optional[Path] = None

    def pair_outdir(self) -> Path:
        return self.out_dir / self.pair_id

    def aoi(self) -> str:
        # For IARPA we prefix pair_id with '<AOI>/', for DFC use city from tokens
        if "/" in self.pair_id:
            return self.pair_id.split("/")[0]
        # DFC: take first two parts of left token
        left_token = self.pair_id.split("-")[0]
        ps = left_token.split("_")
        return "_".join(ps[:2]) if len(ps) >= 2 else ""


# --------------------------- MonSter predictor ---------------------------


class MonSterPredictor:
    """
    Wraps thirdparty MonSter with predict_batch().
    Inputs: batch_* in [0,1] → internally scaled to [0,255], pad-to-32, iters=32.
    """

    def __init__(self, ckpt: str, depth_anything_v2_path: Optional[str], device: str):
        self.device = device
        self.model = thirdparty.build_monster(
            monster_ckpt=ckpt,
            depth_anything_v2_path=depth_anything_v2_path,
            device=device,
        )

    @torch.no_grad()
    def predict_batch(self, batch_L: torch.Tensor, batch_R: torch.Tensor) -> np.ndarray:
        L255, R255 = batch_L * 255.0, batch_R * 255.0
        Lp, pad = pad_to_multiple(L255, 32)
        Rp, _ = pad_to_multiple(R255, 32)
        disp = self.model(Lp, Rp, iters=32, test_mode=True)  # [B,1,H,W]
        disp = unpad(disp, pad).squeeze(1).cpu().numpy()
        return disp


# --------------------------- StereoAnywhere predictor ---------------------------
class StereoAnywherePredictor:
    """
    Wraps thirdparty StereoAnywhere with predict_batch().
    Inputs: batch_* in [0,1], pad-to-32 internally, iters from model args.
    """

    def __init__(self, ckpt: str, depth_anything_v2_path: Optional[str], device: str):
        self.device = device
        self.stereo_model, self.depth_model = thirdparty.build_stereoanywhere(
            ckpt, depth_anything_v2_path, device
        )
        self.stereo_model.eval()
        self.depth_model.eval()

    @torch.no_grad()
    def predict_batch(self, batch_L: torch.Tensor, batch_R: torch.Tensor) -> np.ndarray:
        # Move to device (expects float in [0,1])
        L = batch_L.to(self.device)
        R = batch_R.to(self.device)

        # ---- Monocular priors from DepthAnythingV2 (no padding for the mono pass) ----
        B, _, H, W = L.shape
        mono_depths = self.depth_model.infer_image(
            torch.cat([L, R], dim=0),  # [2B, C, H, W]
            input_size_width=W,
            input_size_height=H,
        )
        # Normalize to [0,1] (safe denom)
        md_min = mono_depths.min()
        md_max = mono_depths.max()
        mono_depths = (mono_depths - md_min) / (md_max - md_min + 1e-8)

        # Split back into left/right batches
        mono_left = mono_depths[:B]  # [B, 1, H, W] typically
        mono_right = mono_depths[B : 2 * B]  # [B, 1, H, W]

        # ---- Pad all inputs to multiple of 32 for the stereo model ----
        Lp, pad = pad_to_multiple(L, 32)
        Rp, _ = pad_to_multiple(R, 32)
        mlp, _ = pad_to_multiple(mono_left, 32)
        mrp, _ = pad_to_multiple(mono_right, 32)

        # ---- Stereo inference ----
        disp, _ = self.stereo_model(
            Lp,
            Rp,
            mlp,
            mrp,
            test_mode=True,
            iters=self.stereo_model.args.iters,
        )
        # Model outputs negative disparities; flip sign to match convention
        disp = -disp

        # Unpad back to original size and return [B, H, W] numpy
        disp = unpad(disp, pad).squeeze(1).cpu().numpy()
        return disp


# --------------------------- FoundationStereo predictor ---------------------------
class FoundationStereoPredictor:
    """
    Wraps thirdparty FoundationStereo with predict_batch().
    Inputs: batch_* in [0,1] → internally scaled to [0,255], FsInputPadder(divis_by=32), iters=32.
    """

    def __init__(self, ckpt: str, device: str):
        self.device = device
        self.stereo_model = thirdparty.build_foundation_stereo(ckpt, device)
        self.stereo_model.eval()

    @torch.no_grad()
    def predict_batch(self, batch_L: torch.Tensor, batch_R: torch.Tensor) -> np.ndarray:
        # Move to device; expects floats in [0,1]
        L = batch_L.to(self.device) * 255.0
        R = batch_R.to(self.device) * 255.0
        # L = batch_L.to(self.device)
        # R = batch_R.to(self.device)

        # Pad to multiples of 32 (non-square allowed)
        # padder = thirdparty.FsInputPadder(L.shape, divis_by=32, force_square=False)
        padder = FsInputPadder(L.shape, divis_by=32, force_square=False)
        Lp, Rp = padder.pad(L, R)

        # Hierarchical inference (autocast on CUDA only)
        # Print input range of inputs to model (Lp, Rp)
        print("Input range Lp:", Lp.min().item(), Lp.max().item())
        print("Input range Rp:", Rp.min().item(), Rp.max().item())
        with torch.autocast(
            device_type="cuda", enabled=str(self.device).startswith("cuda")
        ):
            disp = self.stereo_model.run_hierachical(
                Lp, Rp, iters=32, test_mode=True, small_ratio=0.5
            )  # [B,1,H',W']

        # Unpad back to original size and return [B,H,W] numpy
        disp = padder.unpad(disp).squeeze(1).cpu().numpy()
        return disp


# --------------------------- RAFT-Stereo predictor ---------------------------
class RAFTPredictor:
    """
    Wraps thirdparty RAFT-Stereo with predict_batch().
    Inputs: batch_* in [0,1]. We pad to multiple-of-32 here (replicate)
    to match the training/eval behavior, then unpad back.
    """

    def __init__(self, ckpt: str, device: str):
        self.device = device
        # This returns a module with signature:
        #   disp = model(left, right, iters=32, test_mode=True) -> [B,1,H,W]
        self.model = thirdparty.build_raft_stereo(ckpt=ckpt, device=device)
        self.model.eval()

    @torch.no_grad()
    def predict_batch(self, batch_L: torch.Tensor, batch_R: torch.Tensor) -> np.ndarray:
        L = batch_L.to(self.device)  # [0,1] as in your RAFT code
        R = batch_R.to(self.device)

        Lp, pad = pad_to_multiple(L, 32)
        Rp, _ = pad_to_multiple(R, 32)

        # Keep same default iters (32) as the other predictors
        disp = self.model(Lp, Rp, iters=32, test_mode=True)  # [B,1,H,W]
        disp = unpad(disp, pad).squeeze(1).detach().cpu().numpy()
        disp = -disp
        return disp


# --------------------------- Collation & batching ---------------------------


def _collate_minibatch(samples: List[PairSample], max_side: int, device: str):
    """
    Read → optional center-crop → pad to batch max H/W (replicate) → stack → device.
    Returns: batch_L, batch_R, metas (crop & pad per item).
    """
    imgs_L, imgs_R, metas = [], [], []
    for s in samples:
        L = _read_image_any(s.left_path)
        R = _read_image_any(s.right_path)
        Lc, Rc, crop = center_crop_if_large(L, R, max_side=max_side)
        imgs_L.append(Lc)
        imgs_R.append(Rc)
        metas.append({"pair": s, "crop": crop})

    Hmax = max(t.shape[-2] for t in imgs_L)
    Wmax = max(t.shape[-1] for t in imgs_L)

    def _pad_to(Hb, Wb, x):
        ph = Hb - x.shape[-2]
        pw = Wb - x.shape[-1]
        pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)  # l r t b
        return F.pad(x, pad, mode="replicate"), pad

    out_L, out_R = [], []
    for i, (L, R) in enumerate(zip(imgs_L, imgs_R)):
        Lp, pad = _pad_to(Hmax, Wmax, L)
        Rp, _ = _pad_to(Hmax, Wmax, R)
        metas[i]["pad"] = pad
        out_L.append(Lp)
        out_R.append(Rp)

    batch_L = torch.cat(out_L, dim=0).to(device, non_blocking=True)
    batch_R = torch.cat(out_R, dim=0).to(device, non_blocking=True)
    return batch_L, batch_R, metas


def _to_png(x3hw: torch.Tensor) -> np.ndarray:
    """[3,H,W] in [0,1] → HWC uint8."""
    x = (x3hw * 255.0).clip(0, 255).byte().cpu().numpy()
    return np.transpose(x, (1, 2, 0))


# --------------------------- Per-pair post (save + DSM + CSV row) ---------------------------


def _save_pair_outputs_and_row(
    meta: Dict,
    disp: np.ndarray,
    imgL_3hw: torch.Tensor,
    imgR_3hw: torch.Tensor,
    border_trim_k: int,
    model_name: str,
) -> Dict:
    """
    Save left/right PNG, disparity.iio, DSM & MAE (if data available).
    Returns a dict suitable for CSV summary.
    """
    s: PairSample = meta["pair"]
    crop = meta["crop"]
    pair_dir = s.pair_outdir()
    _ensure_dir(pair_dir)

    # Always save PNGs + disparity
    iio.write(str(pair_dir / "left_image.png"), _to_png(imgL_3hw))
    iio.write(str(pair_dir / "right_image.png"), _to_png(imgR_3hw))
    iio.write(str(pair_dir / "disparity.iio"), disp.astype(np.float32))

    row = {
        "model": model_name,
        "pair_id": s.pair_id,
        "aoi": s.aoi(),
        "left_path": str(s.left_path),
        "right_path": str(s.right_path),
        "homography_path": str(s.homography_path) if s.homography_path else "",
        "left_rpc_json": str(s.left_rpc_json) if s.left_rpc_json else "",
        "right_rpc_json": str(s.right_rpc_json) if s.right_rpc_json else "",
        "gt_dsm_path": str(s.gt_dsm_path) if s.gt_dsm_path else "",
        "pred_dsm_path": "",
        "mae": "",
        "mae_filtered": "",
        "mae_no_water": "",
        "status": "skipped",
        "error": "",
    }

    # Triangulate DSM if possible
    if s.homography_path and s.left_rpc_json and s.right_rpc_json and s.gt_dsm_path:
        try:
            hom = np.load(str(s.homography_path))
            Hleft, Hright = hom["Hleft"], hom["Hright"]

            # shift homographies if we center-cropped
            if crop is not None:
                h0, h1, w0, w1 = crop
                T_crop = np.array(
                    [[1, 0, -float(w0)], [0, 1, -float(h0)], [0, 0, 1]], dtype=float
                )
                Hleft = T_crop @ Hleft
                Hright = T_crop @ Hright

            # border trim + shift homographies
            disp_t, Hl2, Hr2 = trim_border_disparity(
                disp, Hleft, Hright, k=border_trim_k
            )

            left_rpc = misc.rpc_from_json(str(s.left_rpc_json))
            right_rpc = misc.rpc_from_json(str(s.right_rpc_json))

            with rasterio.open(str(s.gt_dsm_path)) as src:
                dsm_meta = src.meta.copy()
                dsm = src.read(1).astype(np.float32)
                dsm_shape = dsm.shape

            alt_img = misc.altitude_image_from_disparity_vectorized(
                disp_t, left_rpc, right_rpc, Hl2, Hr2
            )
            pred_dsm, pred_meta = misc.rectified_altitude_to_dsm(
                alt_img, left_rpc, Hl2, dsm_shape, dsm_meta
            )

            pred_path = pair_dir / "pred_dsm.tif"
            with rasterio.open(
                str(pred_path),
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
                str(s.gt_dsm_path),
                str(pred_path),
                filter_foliage=False,
                filter_water=False,
            )
            mae_f = misc.align_dsm_with_gt(
                str(s.gt_dsm_path),
                str(pred_path),
                filter_foliage=True,
                filter_water=True,
            )
            mae_no_water = misc.align_dsm_with_gt(
                str(s.gt_dsm_path),
                str(pred_path),
                filter_foliage=False,
                filter_water=True,
            )
            print(
                f"[{s.pair_id}] MAE between GT and aligned DSM: {mae:.6f}, filtered: {mae_f:.6f}, no water: {mae_no_water:.6f}"
            )

            # fill row
            row.update(
                {
                    "pred_dsm_path": str(pred_path),
                    "mae": f"{mae:.6f}",
                    "mae_filtered": f"{mae_f:.6f}",
                    "mae_no_water": f"{mae_no_water:.6f}",
                    "status": "ok",
                    "error": "",
                }
            )

        except Exception as e:
            row.update(
                {
                    "status": "error",
                    "error": str(e),
                }
            )
            # also drop a text file for easier debugging
            with open(pair_dir / "triangulation_error.txt", "w") as f:
                f.write(str(e) + "\n")

    return row


# --------------------------- Runner ---------------------------


def run_batches(
    samples: List[PairSample],
    predictor: MonSterPredictor,
    batch_size: int = 1,
    max_side: int = 1024,
    border_trim_k: int = 32,
):
    """
    Mini-batch evaluation:
      - read + optional center crop
      - pad to batch max H/W
      - MonSter inference (internal pad-to-32)
      - save images, disparity, DSM + MAE when possible
      - append a row to summary CSV per pair
    """
    # Initialize CSV once (top-level output_dir used by all pairs)
    if samples:
        _init_summary(samples[0].out_dir)

    device = predictor.device
    for i in tqdm(
        range(0, len(samples), batch_size),
        total=(len(samples) + (batch_size - 1)) // batch_size,
        desc="Eval",
    ):
        chunk = samples[i : i + batch_size]
        batch_L, batch_R, metas = _collate_minibatch(
            chunk, max_side=max_side, device=device
        )
        disp_np = predictor.predict_batch(batch_L, batch_R)  # [B,H,W]
        for bi, meta in enumerate(metas):
            # remove in-batch padding before saving PNGs
            lpad, rpad, tpad, bpad = meta["pad"]
            L = batch_L[
                bi, :, tpad : batch_L.shape[-2] - bpad, lpad : batch_L.shape[-1] - rpad
            ]
            R = batch_R[
                bi, :, tpad : batch_R.shape[-2] - bpad, lpad : batch_R.shape[-1] - rpad
            ]
            disp_full = disp_np[bi]
            disp_unpadded = disp_full[
                tpad : disp_full.shape[0] - bpad, lpad : disp_full.shape[1] - rpad
            ]
            _ensure_dir(meta["pair"].pair_outdir())
            row = _save_pair_outputs_and_row(
                meta=meta,
                disp=disp_unpadded,
                imgL_3hw=L,
                imgR_3hw=R,
                border_trim_k=border_trim_k,
                model_name="monster",
            )
            _append_summary(meta["pair"].out_dir, row)


# --------------------------- Dataset adapters ---------------------------


def discover_satnerf_dfc(dataset_dir: Path, output_dir: Path) -> List[PairSample]:
    """
    SatNeRF DFC layout:
      stereo_pairs_ba/{L,R,homography}
      root_dir/crops_rpcs_ba_v2/<CITY>/*.json
      Track3-Truth/<CITY>_DSM.tif
    """
    L = dataset_dir / "stereo_pairs_ba" / "L"
    R = dataset_dir / "stereo_pairs_ba" / "R"
    Hdir = dataset_dir / "stereo_pairs_ba" / "homography"
    rpc_root = dataset_dir / "root_dir" / "crops_rpcs_ba_v2"
    truth_root = dataset_dir / "Track3-Truth"

    assert L.exists() and R.exists() and Hdir.exists(), "Missing L/R/homography dirs."

    samples: List[PairSample] = []
    for lp in sorted(L.glob("*.iio")):
        pid = lp.stem
        rp = R / f"{pid}.iio"
        hp = Hdir / f"{pid}.npz"
        if not rp.exists():
            continue
        left_token, right_token = pid.split("-")

        def token_city(tok: str) -> str:
            ps = tok.split("_")
            return "_".join(ps[:2])

        def rpc_path(tok: str) -> Optional[Path]:
            c = token_city(tok)
            p = rpc_root / c / f"{tok}.json"
            return p if p.exists() else None

        city = token_city(left_token)
        gt_dsm = truth_root / f"{city}_DSM.tif"
        gt_dsm = gt_dsm if gt_dsm.exists() else None

        samples.append(
            PairSample(
                pair_id=pid,
                left_path=lp,
                right_path=rp,
                out_dir=output_dir,
                homography_path=hp if hp.exists() else None,
                left_rpc_json=rpc_path(left_token),
                right_rpc_json=rpc_path(right_token),
                gt_dsm_path=gt_dsm,
            )
        )
    return samples


def discover_satnerf_iarpa(dataset_dir: Path, output_dir: Path) -> List[PairSample]:
    """
    SatNeRF IARPA layout (per-AOI subfolders):
      stereo_pairs_ba/<AOI>/{L,R,homography}
      root_dir/rpcs_ba/<AOI>/*.json
      Truth/<AOI>_DSM.tif

    pair_id is '<AOI>/<LEFTTOK>-<RIGHTTOK>' so outputs group by AOI.
    """
    import re

    stereo_root = dataset_dir / "stereo_pairs_ba"
    rpc_root = dataset_dir / "root_dir" / "rpcs_ba"
    truth_root = dataset_dir / "Truth"

    assert stereo_root.exists(), "Missing 'stereo_pairs_ba'."

    def split_with_valid(pid: str, valid_tokens: set[str]) -> tuple[str, str] | None:
        s = pid  # '<LEFT>-<RIGHT>'
        # Try every hyphen split: prefer exact match against known RPC tokens
        for i, ch in enumerate(s):
            if ch == "-":
                left, right = s[:i], s[i + 1 :]
                if left in valid_tokens and right in valid_tokens:
                    return left, right
        # Fallback: hyphen right before a new WV token (e.g., -03OCT15WV0...)
        m = re.search(r"-(?=\d{2}[A-Z]{3}\d{2}WV0\d)", s)
        if m:
            return s[: m.start()], s[m.end() :]
        return None

    samples: List[PairSample] = []
    for aoi_dir in sorted(p for p in stereo_root.iterdir() if p.is_dir()):
        aoi = aoi_dir.name
        Ldir, Rdir, Hdir = aoi_dir / "L", aoi_dir / "R", aoi_dir / "homography"
        if not (Ldir.exists() and Rdir.exists() and Hdir.exists()):
            continue

        # Build the validation set from RPC JSON stems for this AOI
        rpc_dir = rpc_root / aoi
        valid = {p.stem for p in rpc_dir.glob("*.json")} if rpc_dir.exists() else set()

        gt_dsm = truth_root / f"{aoi}_DSM.tif"
        gt_dsm = gt_dsm if gt_dsm.exists() else None

        for lp in sorted(Ldir.glob("*.iio")):
            pid = lp.stem  # '<LEFTTOK>-<RIGHTTOK>' but each token can contain hyphens
            rp = Rdir / f"{pid}.iio"
            hp = Hdir / f"{pid}.npz"
            if not rp.exists():
                continue

            split = split_with_valid(pid, valid)
            if split is None:
                # cannot split robustly; skip this pair
                # (optional) print(f"[warn] cannot split pair id for {aoi}: {pid}")
                continue
            left_token, right_token = split

            def rpc_path(tok: str) -> Optional[Path]:
                p = rpc_dir / f"{tok}.json"
                return p if p.exists() else None

            pair_id = f"{aoi}/{pid}"
            samples.append(
                PairSample(
                    pair_id=pair_id,
                    left_path=lp,
                    right_path=rp,
                    out_dir=output_dir,
                    homography_path=hp if hp.exists() else None,
                    left_rpc_json=rpc_path(left_token),
                    right_rpc_json=rpc_path(right_token),
                    gt_dsm_path=gt_dsm,
                )
            )
    return samples


def discover_dfc_flat(
    dataset_dir: Path,
    output_dir: Path,
    aoi_csv: Optional[Path] = None,
    max_pairs_per_aoi: Optional[int] = None,
    sync: bool = True,
) -> List[PairSample]:
    """
    DFC (flat) layout (no per-AOI subdirs):
      dataset_dir/
        stereo_pairs/{L,R,homography}
        root_dir/*.json
        Track3-Truth/<AOI>_DSM.tif

    Returns List[PairSample].
    pair_id is '<LEFT>-<RIGHT>' (no AOI prefix), like SatNeRF DFC.
    """
    # stereo_root = dataset_dir / "stereo_pairs"
    if sync:
        print("Discovering synchronic pairs...")
        stereo_root = dataset_dir / "test_stereo_pairs/sinchronic"
    else:
        print("Discovering diachronic pairs...")
        stereo_root = dataset_dir / "test_stereo_pairs/diachronic"
    Ldir, Rdir, Hdir = stereo_root / "L", stereo_root / "R", stereo_root / "homography"
    rpc_root = dataset_dir / "root_dir"
    truth_root = dataset_dir / "Track3-Truth"

    assert Ldir.exists() and Rdir.exists(), "Missing stereo_pairs/L or stereo_pairs/R."

    def _parse_pair_name(stem: str) -> tuple[str, str]:
        left, right = stem.split("-", 1)
        return left, right

    def _aoi_from_image_id(img_id: str) -> str:
        parts = img_id.split("_")
        return f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else parts[0]

    def _read_allowed_aois(csv_path: Optional[Path]) -> Optional[set[str]]:
        if not csv_path:
            return None
        allowed: set[str] = set()
        with csv_path.open(newline="") as f:
            sniff = f.read(1024)
            f.seek(0)
            if "aoi" in sniff.lower():
                for row in csv.DictReader(f):
                    val = (row.get("aoi_id") or row.get("aoi") or "").strip()
                    if val:
                        allowed.add(val)
            else:
                for line in f:
                    val = line.strip()
                    if val and not val.lower().startswith("aoi"):
                        allowed.add(val)
        return allowed

    allowed_aois = _read_allowed_aois(aoi_csv)

    by_aoi: dict[str, List[PairSample]] = {}

    for lp in sorted(Ldir.glob("*.iio")):
        stem = lp.stem  # '<LEFT>-<RIGHT>'
        if "-" not in stem:
            continue
        left_id, right_id = _parse_pair_name(stem)
        aoi = _aoi_from_image_id(left_id)

        if allowed_aois is not None and aoi not in allowed_aois:
            continue

        rp = Rdir / f"{stem}.iio"
        if not rp.exists():
            continue

        lrpc = rpc_root / f"{left_id}.json"
        rrpc = rpc_root / f"{right_id}.json"
        if not (lrpc.exists() and rrpc.exists()):
            continue

        hp = (Hdir / f"{stem}.npz") if Hdir.exists() else None
        if hp is not None and not hp.exists():
            hp = None

        gt = truth_root / f"{aoi}_DSM.tif"
        if not gt.exists():
            continue

        # Match SatNeRF DFC style: pair_id = '<LEFT>-<RIGHT>' (no AOI prefix)
        sample = PairSample(
            pair_id=stem,
            left_path=lp,
            right_path=rp,
            out_dir=output_dir,
            homography_path=hp,
            left_rpc_json=lrpc,
            right_rpc_json=rrpc,
            gt_dsm_path=gt,
        )
        by_aoi.setdefault(aoi, []).append(sample)

    samples: List[PairSample] = []
    for aoi, lst in sorted(by_aoi.items()):
        lst_sorted = sorted(lst, key=lambda s: s.pair_id)
        if isinstance(max_pairs_per_aoi, int) and max_pairs_per_aoi > 0:
            lst_sorted = lst_sorted[:max_pairs_per_aoi]
        samples.extend(lst_sorted)

    return samples


# --------------------------- CLI ---------------------------


def build_argparser():
    p = argparse.ArgumentParser(
        description="Batch DSM evaluation (SatNeRF DFC & IARPA) with CSV summary."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(ap):
        ap.add_argument("--dataset-dir", type=Path, required=True)
        ap.add_argument("--output-dir", type=Path, required=True)
        ap.add_argument("--stereo-ckpt", type=Path, required=True)
        ap.add_argument("--depth-anything-v2-path", type=Path, required=True)
        ap.add_argument("--device", type=str, default="cuda:0")
        ap.add_argument("--batch-size", type=int, default=1)
        ap.add_argument("--max-side", type=int, default=1024)
        ap.add_argument("--border-trim-k", type=int, default=32)
        ap.add_argument("--limit", type=int, default=0, help="0 = all")
        ap.add_argument(
            "--model",
            type=str,
            default="monster",
            choices=["monster", "stereoanywhere", "foundationstereo", "raft"],
        )
        ap.add_argument(
            "--sync", action="store_true", help="Evaluate only synchronic pairs"
        )
        ap.add_argument(
            "--no-sync",
            dest="sync",
            action="store_false",
            help="Evaluate diachronic pairs",
        )
        ap.set_defaults(sync=True)

    add_common(
        sub.add_parser("discover-satnerf-dfc", help="Run on SatNeRF DFC layout.")
    )
    add_common(
        sub.add_parser(
            "discover-satnerf-iarpa", help="Run on SatNeRF IARPA layout (per-AOI)."
        )
    )
    add_common(sub.add_parser("discover-dfc", help="Run DFC2019 layout."))
    return p


def set_reproducible(seed: int = 42):
    # must be set BEFORE first CUDA op
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # cuBLAS determinism
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def main():
    set_reproducible(42)
    args = build_argparser().parse_args()

    if args.model == "monster":
        predictor = MonSterPredictor(
            ckpt=str(args.stereo_ckpt),
            depth_anything_v2_path=str(args.depth_anything_v2_path),
            device=args.device,
        )
    elif args.model == "stereoanywhere":
        predictor = StereoAnywherePredictor(
            ckpt=str(args.stereo_ckpt),
            depth_anything_v2_path=str(args.depth_anything_v2_path),
            device=args.device,
        )
    elif args.model == "foundationstereo":
        predictor = FoundationStereoPredictor(
            ckpt=str(args.stereo_ckpt),
            device=args.device,
        )
    elif args.model == "raft":
        predictor = RAFTPredictor(
            ckpt=str(args.stereo_ckpt),
            device=args.device,
        )
    else:
        raise RuntimeError(f"Unknown model {args.model}")

    if args.cmd == "discover-satnerf-dfc":
        samples = discover_satnerf_dfc(args.dataset_dir, args.output_dir)
    elif args.cmd == "discover-satnerf-iarpa":
        samples = discover_satnerf_iarpa(args.dataset_dir, args.output_dir)
    elif args.cmd == "discover-dfc":
        samples = discover_dfc_flat(
            args.dataset_dir,
            args.output_dir,
            aoi_csv=Path(
                "/home/emasquil/diachronic-stereo/splits/dfc_v0/v0/test_aois_only_OMA.csv"
            ),
            max_pairs_per_aoi=20,
            sync=args.sync,
        )
    else:
        raise RuntimeError("Unknown command")

    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    # Ensure pair outdirs exist, init summary once
    for s in samples:
        _ensure_dir(s.pair_outdir())
    if samples:
        _init_summary(samples[0].out_dir)

    run_batches(
        samples=samples,
        predictor=predictor,
        batch_size=max(1, int(args.batch_size)),
        max_side=int(args.max_side),
        border_trim_k=int(args.border_trim_k),
    )


if __name__ == "__main__":
    main()
