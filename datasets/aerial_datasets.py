from __future__ import annotations

"""dataloaders/aerial_stereo_datasets.py

Stereo‑only aerial benchmarks that share the **exact same preprocessing**
(cropping, multi‑scale, transforms) used our `StereoDFC`.

Each sample dict therefore has the same keys:
    * ``left``   –      C×H×W float32 in [0,1]
    * ``right``  –      C×H×W float32 in [0,1]
    * ``disparity`` – 1×H×W float32 (pixel disparities)
    * ``valid``      – 1×H×W float32 mask (1 for valid disparity, 0 for invalid)
    * ``filename``   – str (basename, no path)

The cropping strategy:
    * Training   → random crop such that every dimension ≤ *crop_size*.
    * Testing    → centre crop with the same size rule.

Scaling:
    * Same semantics: a single float or a list of floats for random multi‑scale.
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable

import cv2
import iio
import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    "EuroSDRVaihingenStereo",
    "ToulouseUMBRAStereo",
    "ToulouseMetroStereo",
    "EnschedeStereo",
]

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def _normalize_image(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    max_v = float(img.max())
    min_v = float(img.min())
    if max_v == 0:
        return img
    if max_v > 1.01:
        img /= 255.0
    else:
        rng = max_v - min_v
        if rng > 1e-5:
            img -= min_v
            img /= rng
    return img


def _resize(img: np.ndarray, scale: float, *, is_map: bool) -> np.ndarray:
    if scale == 1:
        return img
    new_h = max(1, int(round(img.shape[0] / scale)))
    new_w = max(1, int(round(img.shape[1] / scale)))
    interp = cv2.INTER_NEAREST if is_map else cv2.INTER_AREA
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def _center_crop(imgs: List[np.ndarray], crop_size: int) -> List[np.ndarray]:
    h, w = imgs[0].shape[:2]
    ch = min(crop_size, h)
    cw = min(crop_size, w)
    top = (h - ch) // 2
    left = (w - cw) // 2
    return [img[top : top + ch, left : left + cw] for img in imgs]


def _random_crop(imgs: List[np.ndarray], crop_size: int) -> List[np.ndarray]:
    h, w = imgs[0].shape[:2]
    if h <= crop_size and w <= crop_size:
        return imgs
    ch = min(h, crop_size)
    cw = min(w, crop_size)
    top = random.randint(0, h - ch) if h > ch else 0
    left = random.randint(0, w - cw) if w > cw else 0
    bottom = top + ch
    right = left + cw
    return [img[top:bottom, left:right] for img in imgs]


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------


def _read_rgb(p: Path) -> np.ndarray:
    img = iio.read(str(p))
    return _normalize_image(img)


def _read_disp(p: Path) -> Tuple[np.ndarray, np.ndarray]:
    disp = iio.read(str(p))
    if disp.ndim == 3:
        disp = disp[:, :, 0]
    disp = disp.astype(np.float32) / 256.0  # EuroSDR encoding
    valid = (disp > 0).astype(np.float32)
    return disp, valid


# -----------------------------------------------------------------------------
# Base class with cropping / scaling / augment hooks
# -----------------------------------------------------------------------------


class _AerialStereoBase(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        *,
        split: str = "testing",
        crop_size: int = 1024,
        scale: Union[float, List[float]] = 1,
        transforms: Optional[
            Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
        ] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.train = split.lower() in {"training"}
        self.crop_size = crop_size
        self.transforms = transforms

        if isinstance(scale, (list, tuple)):
            if not scale:
                raise ValueError("Scale list must not be empty")
            self.scale_list = list(map(float, scale))
            self.scale_single = None
        else:
            self.scale_single = float(scale)
            self.scale_list = None

        self.samples: List[Tuple[Path, Path, Path]] = []
        self._crawl()

    # subclasses implement _crawl ------------------------------------------------

    # ------------------------- helper methods ----------------------------------
    def _pick_scale(self) -> float:
        if self.scale_list is not None:
            return random.choice(self.scale_list) if self.train else self.scale_list[0]
        return self.scale_single or 1.0

    # ------------------------- dataset protocol --------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # noqa: Dunder Getitem
        left_p, right_p, disp_p = self.samples[idx]

        # ---------- load np arrays ----------
        left_np = _read_rgb(left_p)
        right_np = _read_rgb(right_p)
        disp_np, valid_np = _read_disp(disp_p)

        # ---------- cropping ---------------
        if max(left_np.shape[0], left_np.shape[1]) > self.crop_size:
            crop_f = _random_crop if self.train else _center_crop
            imgs = crop_f([left_np, right_np, disp_np, valid_np], self.crop_size)
            left_np, right_np, disp_np, valid_np = imgs  # unpack

        # ---------- scaling ---------------
        scale_factor = self._pick_scale()
        if scale_factor != 1:
            left_np = _resize(left_np, scale_factor, is_map=False)
            right_np = _resize(right_np, scale_factor, is_map=False)
            disp_np = _resize(disp_np, scale_factor, is_map=True)
            valid_np = _resize(valid_np, scale_factor, is_map=True)

        # ---------- to torch --------------
        sample = {
            "left": torch.from_numpy(left_np.transpose(2, 0, 1)).float(),  # C×H×W
            "right": torch.from_numpy(right_np.transpose(2, 0, 1)).float(),
            "disparity": torch.from_numpy(disp_np).unsqueeze(0).float(),
            "valid": torch.from_numpy(valid_np).unsqueeze(0).float(),
            "filename": left_p.name,
        }

        if self.transforms and self.train:
            sample = self.transforms(sample)
        return sample


# -----------------------------------------------------------------------------
# Concrete datasets (directory layouts identical to EuroSDR)
# -----------------------------------------------------------------------------


class EuroSDRVaihingenStereo(_AerialStereoBase):
    def _crawl(self):
        for seq in (self.root / self.split).iterdir():
            if not seq.is_dir():
                continue
            ldir, rdir, ddir = seq / "colored_0", seq / "colored_1", seq / "disp_occ"
            for lp in sorted(ldir.glob("*.png")):
                rp, dp = rdir / lp.name, ddir / lp.name
                if rp.exists() and dp.exists():
                    self.samples.append((lp, rp, dp))


class ToulouseUMBRAStereo(EuroSDRVaihingenStereo):
    """Same directory layout as EuroSDR Vaihingen."""


class ToulouseMetroStereo(EuroSDRVaihingenStereo):
    """Toulouse ‘TlseMetro‑Stereo’ benchmark."""


class EnschedeStereo(EuroSDRVaihingenStereo):
    """Enschede‑Stereo‑echo‑new benchmark."""
