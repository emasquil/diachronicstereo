import csv
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import iio
import numpy as np
import torch
from torch.utils.data import Dataset

CROP_SIZE = 1024  # Default crop size for images


def _normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize an image **in‑place** to the [0, 1] range.

    If the image is already in that range, it is returned unchanged.
    If the image has a weird range (e.g. [0, 1000]), it is scaled to [0, 1].
    If it's in 255 range, it is scaled to [0, 1] in place.
    """
    img = img.astype(np.float32, copy=False)
    max_v = float(img.max())
    min_v = float(img.min())

    if max_v == 255 and min_v == 0:
        img /= 255.0
    else:
        # Weird ranges: min‑max to [0, 1]
        rng = max_v - min_v
        if rng > 1e-5:
            img -= min_v
            img /= rng
    return img


def _pad_to_size(img: np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
    """Edge-replicate pad *img* so both dims == size."""
    h, w = img.shape[:2]
    pad_h = max(0, size - h)
    pad_w = max(0, size - w)
    if pad_h == 0 and pad_w == 0:
        return img
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    if img.ndim == 2:
        return np.pad(img, ((top, bottom), (left, right)), mode="edge")
    return np.pad(img, ((top, bottom), (left, right), (0, 0)), mode="edge")


def _center_crop(imgs: List[np.ndarray], size: int = CROP_SIZE) -> List[np.ndarray]:
    """Centre crop to *size×size*; replicate-pad if image is smaller."""
    outputs: List[np.ndarray] = []
    for img in imgs:
        h, w = img.shape[:2]
        top = max(0, (h - size) // 2)
        left = max(0, (w - size) // 2)
        bottom = min(h, top + size)
        right = min(w, left + size)
        cropped = img[top:bottom, left:right]
        cropped = _pad_to_size(cropped, size)
        outputs.append(cropped)
    return outputs


class StereoDFC(Dataset):
    def make_pair_list(self, csv_path):
        out = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                left = Path(row["left_image"]).stem  # remove .tif
                right = Path(row["right_image"]).stem
                out.append(f"{left}-{right}.iio")
        return out

    def __init__(
        self,
        left_dir: Optional[str] = None,
        right_dir: Optional[str] = None,
        disparity_dir: Optional[str] = None,
        disparity_dir_no_trees: Optional[str] = None,
        train: bool = True,
        aois_csv: Optional[str] = None,
        diachronic_list_csv: Optional[str] = None,
        syncronic_list_csv: Optional[str] = None,
        crop_size: int = CROP_SIZE,
        transforms: Optional[
            Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
        ] = None,
    ) -> None:
        super().__init__()

        self.train = train
        self.crop_size = crop_size
        self.transforms = transforms

        # Load CSVs
        self.aois = None
        if aois_csv is not None:
            with open(aois_csv, "r") as f:
                aois = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(aois)} AOIs from {aois_csv}")
            self.aois = set(aois)

        self.diachronic_pairs = None
        if diachronic_list_csv is not None:
            self.diachronic_pairs = self.make_pair_list(diachronic_list_csv)

        self.syncronic_pairs = None
        if syncronic_list_csv is not None:
            self.syncronic_pairs = self.make_pair_list(syncronic_list_csv)

        # File enumeration ----------------------------------------------------------
        self.left_dir = os.path.expanduser(left_dir)  # type: ignore[arg-type]
        files = sorted(
            [
                f
                for f in os.listdir(self.left_dir)
                if f[:7] in self.aois
                if self.aois is not None
            ]
        )
        # Ensure files exist in the companion dirs
        self.right_dir = os.path.expanduser(right_dir)  # type: ignore[arg-type]
        self.disp_dir = os.path.expanduser(disparity_dir)  # type: ignore[arg-type]
        self.disp_dir_no_trees = os.path.expanduser(disparity_dir_no_trees)
        for fname in files:
            for d in (self.right_dir, self.disp_dir, self.disp_dir_no_trees):
                if not os.path.exists(os.path.join(d, fname)):
                    raise FileNotFoundError(f"{fname} missing from {d}")
        self.filenames = files

    def __len__(self) -> int:  # noqa: Dunder Length
        return len(self.filenames)

    # ---------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # noqa: Dunder Getitem
        fname = self.filenames[idx]

        left = iio.read(os.path.join(self.left_dir, fname))
        right = iio.read(os.path.join(self.right_dir, fname))
        disp = iio.read(os.path.join(self.disp_dir, fname))  # disparity – left aligned
        disp = np.nan_to_num(disp, copy=False)  # replace NaNs with 0
        disp_no_trees = iio.read(os.path.join(self.disp_dir_no_trees, fname))
        disp_no_trees = np.nan_to_num(disp_no_trees, copy=False)
        left = _normalize_image(left)
        right = _normalize_image(right)
        sample_np = {
            "left": left,
            "right": right,
            "disparity": disp.astype(np.float32, copy=False),
            "disparity_no_trees": disp_no_trees.astype(np.float32, copy=False),
        }

        # ----------------- Crop (for val, in train we do it in aug) -----------------
        if not self.train:
            cropped = _center_crop(list(sample_np.values()), self.crop_size)
            for key, arr in zip(sample_np.keys(), cropped):
                sample_np[key] = arr

        # --------------------------- Numpy → Tensor ------------------------------
        tensor_sample: Dict[str, torch.Tensor] = {}
        for k, arr in sample_np.items():
            if arr.ndim == 2:  # H x W (map)
                tensor = torch.from_numpy(arr).unsqueeze(0)  # 1×H×W
            else:  # H x W x C (image)
                tensor = torch.from_numpy(arr.transpose(2, 0, 1))  # C×H×W
            tensor_sample[k] = tensor.float()

        # --------------------------- Augmentations -------------------------------
        if self.transforms is not None and self.train:
            tensor_sample = self.transforms(tensor_sample)

        if self.diachronic_pairs is not None and self.syncronic_pairs is not None:
            if fname in self.diachronic_pairs:
                tensor_sample["diachronic_pair"] = torch.tensor(1)
            elif fname in self.syncronic_pairs:
                tensor_sample["diachronic_pair"] = torch.tensor(0)
            else:
                raise ValueError(
                    f"File {fname} not found in either diachronic or syncronic pairs list."
                )

        tensor_sample["filename"] = fname  # keep track for debugging
        return tensor_sample
