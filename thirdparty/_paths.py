from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

# Centralized sys.path handling for vendor repos.
_THIRDPARTY_DIR = Path(__file__).resolve().parent


def _ensure_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def ensure_monster_paths() -> Tuple[Path, Path]:
    monster_root = _THIRDPARTY_DIR / "MonSter"
    depth_anything_root = monster_root / "Depth-Anything-V2-list3"
    if not monster_root.exists():
        raise FileNotFoundError(f"MonSter repo not found at {monster_root}")
    if not depth_anything_root.exists():
        raise FileNotFoundError(
            f"Depth-Anything V2 not found at {depth_anything_root}"
        )
    _ensure_sys_path(monster_root)
    _ensure_sys_path(depth_anything_root)
    return monster_root, depth_anything_root


def ensure_raft_paths() -> Path:
    repo_root = _THIRDPARTY_DIR / "RAFT-Stereo"
    core_dir = repo_root / "core"
    if not core_dir.exists():
        raise FileNotFoundError(
            f"Could not find RAFT-Stereo at {repo_root}. "
            f"Expected {core_dir} to exist."
        )
    _ensure_sys_path(repo_root)
    return repo_root
