#!/usr/bin/env python3
"""
Batch evaluation for aerial stereo datasets (disparity metrics only, no DSM).

Runs a selected stereo model (MonSter, StereoAnywhere, or FoundationStereo) on
paired aerial imagery, compares predicted disparities against GT over valid
pixels, and writes global and per-pair metrics. Optional top-K saving stores
best pairs (by EPE) with left/right images and predicted/GT disparity arrays.

Metrics:
  - EPE: mean |pred - gt| (pixels)
  - RMSE: sqrt(mean (pred - gt)^2) (pixels)
  - Good@{1,3,5}: fraction with |error| <= tau (pixels)

Outputs:
  - <out_dir>/global_metrics.json
  - <out_dir>/pair_metrics.csv
  - <out_dir>/best_pairs/ (optional, via --save_top)

Supported datasets (from datasets.aerial_datasets):
  - EnschedeStereo
  - EuroSDRVaihingenStereo
  - ToulouseMetroStereo
  - ToulouseUMBRAStereo
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import iio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import thirdparty  # model builders, FsInputPadder, etc.

# -------------------------- datasets --------------------------
from datasets.aerial_datasets import (
    EnschedeStereo,
    EuroSDRVaihingenStereo,
    ToulouseMetroStereo,
    ToulouseUMBRAStereo,
)

# =====================================================================
# Utils
# =====================================================================


def pad_to_multiple(x: torch.Tensor, multiple: int = 32):
    h, w = x.shape[-2:]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    pad = [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2]  # l r t b
    return F.pad(x, pad, mode="replicate"), pad


def unpad(x: torch.Tensor, pad):
    _, _, h, w = x.shape
    return x[..., pad[2] : h - pad[3], pad[0] : w - pad[1]]


def _tensor_to_uint8(img_3hw: torch.Tensor) -> np.ndarray:
    """[3,H,W] in [0,1] → HWC uint8"""
    arr = (img_3hw.clamp(0, 1) * 255.0).byte().cpu().numpy()
    return np.transpose(arr, (1, 2, 0))


# -------------------------- metrics ---------------------------


def _update_running(running: Dict[str, float], batch: Dict[str, float], n_pix: int):
    for k, v in batch.items():
        running[k] += float(v) * float(n_pix)


def _finalise(running: Dict[str, float], total_pix: int) -> Dict[str, float]:
    if total_pix == 0:
        return {k: float("nan") for k in running.keys()}
    return {k: v / float(total_pix) for k, v in running.items()}


# =====================================================================
# Predictors (self-contained, no external infer_* helpers)
# =====================================================================


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
        self.model.eval()

    @torch.no_grad()
    def predict_batch(
        self, batch_L: torch.Tensor, batch_R: torch.Tensor
    ) -> torch.Tensor:
        L255, R255 = batch_L.to(self.device) * 255.0, batch_R.to(self.device) * 255.0
        Lp, pad = pad_to_multiple(L255, 32)
        Rp, _ = pad_to_multiple(R255, 32)
        disp = self.model(Lp, Rp, iters=32, test_mode=True)  # [B,1,H,W]
        disp = unpad(disp, pad)
        return disp  # [B,1,H,W]


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
    def predict_batch(
        self, batch_L: torch.Tensor, batch_R: torch.Tensor
    ) -> torch.Tensor:
        L = batch_L.to(self.device)
        R = batch_R.to(self.device)

        # Monocular priors
        B, _, H, W = L.shape
        mono_depths = self.depth_model.infer_image(
            torch.cat([L, R], dim=0),
            input_size_width=W,
            input_size_height=H,
        )
        md_min = mono_depths.min()
        md_max = mono_depths.max()
        mono_depths = (mono_depths - md_min) / (md_max - md_min + 1e-8)
        mono_left, mono_right = mono_depths[:B], mono_depths[B : 2 * B]

        # Pad everything to /32
        Lp, pad = pad_to_multiple(L, 32)
        Rp, _ = pad_to_multiple(R, 32)
        mlp, _ = pad_to_multiple(mono_left, 32)
        mrp, _ = pad_to_multiple(mono_right, 32)

        disp, _ = self.stereo_model(
            Lp,
            Rp,
            mlp,
            mrp,
            test_mode=True,
            iters=self.stereo_model.args.iters,
        )
        disp = -disp  # model outputs negative disparities
        disp = unpad(disp, pad)
        return disp  # [B,1,H,W]


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
    def predict_batch(
        self, batch_L: torch.Tensor, batch_R: torch.Tensor
    ) -> torch.Tensor:
        L = batch_L.to(self.device) * 255.0
        R = batch_R.to(self.device) * 255.0
        padder = thirdparty.FsInputPadder(L.shape, divis_by=32, force_square=False)
        Lp, Rp = padder.pad(L, R)
        with torch.autocast(
            device_type="cuda", enabled=str(self.device).startswith("cuda")
        ):
            disp = self.stereo_model.run_hierachical(
                Lp, Rp, iters=32, test_mode=True, small_ratio=0.5
            )
        disp = padder.unpad(disp)
        return disp  # [B,1,H,W]


# =====================================================================
# Dataset factory
# =====================================================================


def get_aerial_dataset(
    name: str,
    root: Path,
    split: str,
    *,
    crop_size: int = 1024,
) -> torch.utils.data.Dataset:
    name = name.lower()
    if name == "eurosdr":
        return EuroSDRVaihingenStereo(
            root, split=split, crop_size=crop_size, transforms=None
        )
    if name == "toulouseumbra":
        return ToulouseUMBRAStereo(
            root, split=split, crop_size=crop_size, transforms=None
        )
    if name == "toulousemetro":
        return ToulouseMetroStereo(
            root, split=split, crop_size=crop_size, transforms=None
        )
    if name == "enschede":
        return EnschedeStereo(root, split=split, crop_size=crop_size, transforms=None)
    raise ValueError(f"Unknown aerial dataset '{name}'")


# =====================================================================
# Main
# =====================================================================


def build_argparser():
    p = argparse.ArgumentParser("Aerial stereo evaluation (no DSM)")
    p.add_argument(
        "--dataset",
        required=True,
        choices=["eurosdr", "toulouseumbra", "toulousemetro", "enschede"],
    )
    p.add_argument("--root", required=True, type=Path, help="Dataset root path")
    p.add_argument("--split", default="testing", help="Dataset split")
    p.add_argument(
        "--model",
        choices=["monster", "stereoanywhere", "foundationstereo"],
        default="monster",
    )
    p.add_argument(
        "--stereo_ckpt", required=True, type=Path, help="Path to stereo checkpoint"
    )
    p.add_argument(
        "--depth_anything_v2_path",
        type=Path,
        help="(StereoAnywhere) path to Depth-Anything V2 weights",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out_dir", required=True, type=Path, help="Where to save results")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--max_eval_pairs", type=int, default=0, help="0 = all; if >0, caps evaluation"
    )
    p.add_argument(
        "--save_top", type=int, default=0, help="Save this many best pairs (by EPE)"
    )
    p.add_argument(
        "--crop_size", type=int, default=1024, help="Dataset crop size if supported"
    )
    return p


def main():
    args = build_argparser().parse_args()
    device = torch.device(args.device)

    # ---------------- Build predictor ----------------
    if args.model == "monster":
        predictor = MonSterPredictor(
            ckpt=str(args.stereo_ckpt),
            depth_anything_v2_path=(
                str(args.depth_anything_v2_path)
                if args.depth_anything_v2_path
                else None
            ),
            device=args.device,
        )
    elif args.model == "stereoanywhere":
        predictor = StereoAnywherePredictor(
            ckpt=str(args.stereo_ckpt),
            depth_anything_v2_path=(
                str(args.depth_anything_v2_path)
                if args.depth_anything_v2_path
                else None
            ),
            device=args.device,
        )
    elif args.model == "foundationstereo":
        predictor = FoundationStereoPredictor(
            ckpt=str(args.stereo_ckpt),
            device=args.device,
        )
    else:
        raise RuntimeError(args.model)

    # ---------------- Dataset / loader ----------------
    ds = get_aerial_dataset(
        args.dataset, args.root, args.split, crop_size=args.crop_size
    )

    if args.max_eval_pairs and args.max_eval_pairs > 0:
        ds = torch.utils.data.Subset(ds, range(min(len(ds), args.max_eval_pairs)))

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Running, weighted by valid pixels
    running = {
        "epe": 0.0,
        "rmse": 0.0,
        "good1": 0.0,
        "good3": 0.0,
        "good5": 0.0,
    }
    total_pix = 0

    # Optional: keep best pairs by EPE
    import heapq

    top_heap: List[Tuple[float, int, Dict[str, object]]] = []
    counter = 0
    best_k = max(0, int(args.save_top))

    pair_rows: List[List[object]] = []  # → CSV rows

    for batch in tqdm(loader, total=len(loader), desc="Evaluating", unit="batch"):
        # Expected keys from dataset: filename (list of len B), left [B,3,H,W], right [B,3,H,W], disparity [B,1,H,W], valid optional
        filenames = batch["filename"]
        left = batch["left"].to(device)  # [B,3,H,W] in [0,1]
        right = batch["right"].to(device)  # [B,3,H,W] in [0,1]
        gt = batch["disparity"].to(device)  # [B,1,H,W]
        valid = batch["valid"].to(device) if "valid" in batch else (gt > 0).float()

        # Predict disparities
        pred = predictor.predict_batch(left, right)  # [B,1,H,W]

        B = pred.shape[0]
        for i in range(B):
            fname = str(filenames[i])
            pred_i = pred[i]  # [1,H,W]
            gt_i = gt[i]
            valid_i = valid[i]

            err = pred_i - gt_i
            abs_err = err.abs()
            n_pix = int(valid_i.sum().item())
            if n_pix == 0:
                continue

            # EPE / RMSE over valid pixels
            epe = (abs_err * valid_i).sum().item() / n_pix
            rmse = ((err * err) * valid_i).sum().item() / n_pix
            rmse = rmse**0.5

            # Good@τ: compare first, then mask
            good1 = (((abs_err <= 1).float() * valid_i).sum().item()) / n_pix
            good3 = (((abs_err <= 3).float() * valid_i).sum().item()) / n_pix
            good5 = (((abs_err <= 5).float() * valid_i).sum().item()) / n_pix

            _update_running(
                running,
                {
                    "epe": epe,
                    "rmse": rmse,
                    "good1": good1,
                    "good3": good3,
                    "good5": good5,
                },
                n_pix,
            )
            total_pix += n_pix

            pair_rows.append([fname, epe, rmse, good1, good3, good5])

            if best_k > 0:
                payload = {
                    "fname": fname,
                    "left": _tensor_to_uint8(left[i].cpu()),
                    "right": _tensor_to_uint8(right[i].cpu()),
                    "pred_disp": pred_i.cpu().squeeze(0).numpy(),
                    "gt_disp": gt_i.cpu().squeeze(0).numpy(),
                }
                heapq.heappush(top_heap, (epe, counter, payload))
                if len(top_heap) > best_k:
                    heapq._heapify_max(
                        top_heap
                    )  # ensure we can pop worst (largest EPE)
                    # remove the worst among top set
                    worst_idx = max(range(len(top_heap)), key=lambda j: top_heap[j][0])
                    top_heap.pop(worst_idx)
                counter += 1

    metrics = _finalise(running, total_pix)

    # ---------------- Save JSON ----------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_out = args.out_dir / "global_metrics.json"
    with open(json_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics to", json_out)
    print(metrics)

    # ---------------- Save per-pair CSV ----------------
    csv_out = args.out_dir / "pair_metrics.csv"
    with open(csv_out, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["file", "epe", "rmse", "good1", "good3", "good5"])
        writer.writerows(pair_rows)
    print("Saved per-pair metrics CSV to", csv_out)

    # ---------------- Save top-k pairs (optional) -------------
    if best_k > 0 and len(top_heap) > 0:
        save_dir = args.out_dir / "best_pairs"
        save_dir.mkdir(parents=True, exist_ok=True)
        # sort by EPE ascending
        top_sorted = sorted(top_heap, key=lambda t: t[0])
        for rank, (epe_val, _, data) in enumerate(top_sorted, 1):
            out_sub = (
                save_dir / f"{rank:02d}_{Path(data['fname']).stem}_EPE_{epe_val:.3f}"
            )
            out_sub.mkdir(exist_ok=True)
            iio.write(str(out_sub / "left.png"), data["left"])
            iio.write(str(out_sub / "right.png"), data["right"])
            np.save(out_sub / "pred_disp.npy", data["pred_disp"])
            np.save(out_sub / "gt_disp.npy", data["gt_disp"])
        print(f"Saved top-{best_k} pairs to {save_dir}")


if __name__ == "__main__":
    main()
