# thirdparty/raft_stereo_builder.py
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Union

import torch

from ._paths import ensure_raft_paths


def _default_args() -> SimpleNamespace:
    """Minimal arg namespace for RAFTStereo constructor."""
    return SimpleNamespace(
        hidden_dims=[128, 128, 128],
        corr_implementation="reg",  # use "reg_cuda"/"alt_cuda" if you compiled CUDA corr
        shared_backbone=False,
        corr_levels=4,
        corr_radius=4,
        n_downsample=2,
        context_norm="batch",
        slow_fast_gru=False,
        n_gru_layers=3,
        # some repos read this field
        mixed_precision=False,
    )


class _RaftWrapper(torch.nn.Module):
    """
    Call like MonSter:
        disp = model(left, right, iters=32, test_mode=True)
    Expects inputs in [0,1]. No padding inside (do it outside).
    """

    def __init__(self, raft: torch.nn.Module):
        super().__init__()
        self.raft = raft

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        iters: int = 32,
        test_mode: bool = True,
    ) -> torch.Tensor:
        out = self.raft(left, right, iters=iters, test_mode=test_mode)
        # Some forks return (flow_low, flow_up), others a single tensor
        if isinstance(out, (tuple, list)):
            return out[-1]
        return out


def build_raft_stereo(
    ckpt: Union[str, Path],
    device: str = "cuda:0",
    args: SimpleNamespace | None = None,
) -> torch.nn.Module:
    """
    Build + load RAFT-Stereo and return a wrapped module with call:
        disp = model(left, right, iters=32, test_mode=True)
    """
    ensure_raft_paths()

    from core.raft_stereo import RAFTStereo  # import after sys.path injection

    args = _default_args() if args is None else args

    raft = RAFTStereo(args)

    # Robust checkpoint load
    ckpt_obj = torch.load(str(ckpt), map_location="cpu")
    state = ckpt_obj.get("state_dict", ckpt_obj)
    # strip potential DataParallel prefixes
    state = {k.replace("module.", ""): v for k, v in state.items()}
    raft.load_state_dict(state, strict=True)

    raft.to(device)
    raft.eval()

    return _RaftWrapper(raft)
