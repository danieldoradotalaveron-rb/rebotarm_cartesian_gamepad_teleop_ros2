"""Locate vendored reBotArm_control_py and add it to sys.path."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def teleop_workspace_root() -> Path:
    """Root of this overlay workspace (rebotarm_cartesian_gamepad_teleop_ros2/)."""
    return Path(__file__).resolve().parents[3]


def driver_workspace_root() -> Path:
    """Root of the driver fork that hosts rebotarm_msgs / rebotarm_bringup."""
    env = os.environ.get("REBOTARM_DRIVER_WS", "").strip()
    if env:
        return Path(env).resolve()
    return teleop_workspace_root().parent


def sdk_candidates() -> list[Path]:
    driver = driver_workspace_root()
    teleop = teleop_workspace_root()
    return [
        driver / "third_party" / "reBotArm_control_py",
        driver / "sdk" / "reBotArm_control_py",
        teleop / "third_party" / "reBotArm_control_py",
        Path.home() / "seeed" / "cameraws" / "sdk" / "reBotArm_control_py",
    ]


def ensure_rebot_sdk_in_syspath() -> Path:
    for root in sdk_candidates():
        if (root / "reBotArm_control_py").is_dir():
            root_str = str(root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
            return root
    candidates = "\n".join(f"  - {path}" for path in sdk_candidates())
    raise FileNotFoundError(f"Cannot find reBotArm_control_py. Clone it into one of:\n{candidates}")
