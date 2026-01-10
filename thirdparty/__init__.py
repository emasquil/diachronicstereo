# TODO: Review later and streamline all this code.
# thirdparty/__init__.py
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union

import torch
from omegaconf import OmegaConf

from ._paths import ensure_monster_paths

# ---------- small helpers ----------


def _device_from_maybe_str(d: Optional[Union[str, torch.device]]) -> torch.device:
    if d is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


# ---------- defaults ----------

DEFAULT_MONSTER_ARGS = SimpleNamespace(
    encoder="vitl",
    hidden_dims=[128, 128, 128],
    corr_implementation="reg",
    shared_backbone=False,
    corr_levels=2,
    corr_radius=4,
    n_downsample=2,
    n_gru_layers=3,
    max_disp=192,
    mixed_precision=False,
    slow_fast_gru=False,
)

DEFAULT_STEROANYWHERE_ARGS = SimpleNamespace(
    maxdisp=192,
    n_downsample=2,
    n_additional_hourglass=0,
    volume_channels=8,
    vol_downsample=0,
    use_truncate_vol=False,
    mirror_conf_th=0.98,
    mirror_attenuation=0.9,
    use_aggregate_stereo_vol=False,
    use_aggregate_mono_vol=False,
    iters=32,
)

# ---------- builders (lazy imports only) ----------


def build_monster(
    monster_ckpt: str,
    depth_anything_v2_path: str,
    device: Optional[Union[str, torch.device]] = None,
    args: SimpleNamespace = DEFAULT_MONSTER_ARGS,
    eval_only: bool = True,
) -> torch.nn.Module:
    """
    Build MonSter with MonSter + Depth-Anything paths managed in one place.
    """
    device = _device_from_maybe_str(device)
    if depth_anything_v2_path is not None:
        try:
            args.depth_anything_v2_path = depth_anything_v2_path
        except Exception:
            if getattr(args, "depth_anything_v2_path", None) != depth_anything_v2_path:
                raise
    ensure_monster_paths()

    from core.monster import Monster  # MonSter uses top-level "core.*" imports

    model = Monster(args)

    if device.type == "cuda" and torch.cuda.device_count() > 1 and device.index is None:
        print(f"✅ Using DataParallel across {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model)

    model.to(device)
    if monster_ckpt:
        state = torch.load(monster_ckpt, map_location=device)
        state = state["state_dict"] if "state_dict" in state else state
        if isinstance(model, torch.nn.DataParallel):
            if not any(k.startswith("module.") for k in state.keys()):
                state = {f"module.{k}": v for k, v in state.items()}
        else:
            state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
    if eval_only:
        model.eval()
    return model


def build_stereoanywhere(
    stereo_ckpt: str,
    depth_anything_v2_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    args: SimpleNamespace = DEFAULT_STEROANYWHERE_ARGS,
    eval_only: bool = True,
):
    """
    Build StereoAnywhere (and DepthAnythingV2). These import cleanly as package modules,
    so we keep package-qualified imports and avoid sys.path pollution.
    """
    device = _device_from_maybe_str(device)

    # Package-qualified imports inside the function (lazy)
    from .stereoanywhere.models.stereoanywhere import StereoAnywhere
    from .stereoanywhere.models.depth_anything_v2 import get_depth_anything_v2

    stereo_model: torch.nn.Module = StereoAnywhere(args)
    if device.type == "cuda" and torch.cuda.device_count() > 1 and device.index is None:
        print(
            f"✅ Using DataParallel across {torch.cuda.device_count()} GPUs for StereoAnywhere."
        )
        stereo_model = torch.nn.DataParallel(stereo_model)
    stereo_model.to(device)

    try:
        state = torch.load(stereo_ckpt, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(stereo_ckpt, map_location=device)
    state = state["state_dict"] if "state_dict" in state else state
    state = {k.replace("module.", ""): v for k, v in state.items()}
    stereo_model.load_state_dict(state, strict=True)
    if eval_only:
        stereo_model.eval()

    mono_model: torch.nn.Module = get_depth_anything_v2(depth_anything_v2_path)
    if device.type == "cuda" and torch.cuda.device_count() > 1 and device.index is None:
        mono_model = torch.nn.DataParallel(mono_model)
    mono_model.to(device)
    if eval_only:
        mono_model.eval()
    return stereo_model, mono_model


def build_foundation_stereo(
    foundation_ckpt: str,
    device: Optional[Union[str, torch.device]] = None,
    eval_only: bool = True,
) -> torch.nn.Module:
    """
    Build FoundationStereo with package-qualified imports.
    """
    device = _device_from_maybe_str(device)

    from .FoundationStereo.fs_core.foundation_stereo import FoundationStereo

    ckpt_dir = Path(foundation_ckpt).parent
    cfg = OmegaConf.load(ckpt_dir / "cfg.yaml")
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    args_ns = OmegaConf.create(cfg)

    model = FoundationStereo(args_ns)
    ckpt = torch.load(foundation_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device)
    if eval_only:
        model.eval()
    return model


def build_raft_stereo(
    ckpt: Union[str, Path],
    device: str = "cuda:0",
    args: Optional[SimpleNamespace] = None,
) -> torch.nn.Module:
    """
    Defer to the dedicated builder (which safely handles RAFT's own 'core/').
    """
    from .raft_stereo_builder import build_raft_stereo as _builder

    return _builder(ckpt=ckpt, device=device, args=args)


def __getattr__(name: str):
    if name == "MonsterInputPadder":
        ensure_monster_paths()
        from core.utils.utils import InputPadder as MonsterInputPadder

        globals()["MonsterInputPadder"] = MonsterInputPadder
        return MonsterInputPadder
    if name == "FsInputPadder":
        from .FoundationStereo.fs_core.utils.utils import InputPadder as FsInputPadder

        globals()["FsInputPadder"] = FsInputPadder
        return FsInputPadder
    raise AttributeError(f"module {__name__} has no attribute {name}")
