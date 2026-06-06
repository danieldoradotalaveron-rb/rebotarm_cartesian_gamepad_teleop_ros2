"""Pure FK model init and pose computation using reBotArm_control_py."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from geometry_msgs.msg import Pose

from .fk_pose import fk_arrays_to_pose
from .sdk_path import ensure_rebot_sdk_in_syspath


@dataclass
class FkContext:
    ok: bool
    error: str
    model: Any | None
    data: Any | None
    end_frame_id: int | None
    q_current: np.ndarray | None
    ee_frame: str


def resolve_urdf_path(urdf_path: str) -> str | None:
    """Return absolute URDF path, or None to use the SDK default URDF."""
    urdf_path = (urdf_path or "").strip()
    if urdf_path:
        return os.path.abspath(os.path.expanduser(urdf_path))

    try:
        from ament_index_python.packages import get_package_share_directory

        share = get_package_share_directory("rebotarm_bringup")
        # Arm-only core URDF (no D405 frames) so FK/IK keep exactly 6 active joints.
        return os.path.join(share, "description", "urdf", "reBot-DevArm_fixend_core.urdf")
    except Exception:
        return None


def init_fk_context(
    urdf_path: str,
    ee_frame: str,
    initial_q: list[float],
) -> FkContext:
    ee_frame = (ee_frame or "end_link").strip()
    try:
        ensure_rebot_sdk_in_syspath()
    except FileNotFoundError:
        return FkContext(False, "SDK_NOT_FOUND", None, None, None, None, ee_frame)

    from reBotArm_control_py.kinematics import load_robot_model

    resolved = resolve_urdf_path(urdf_path)
    try:
        if resolved is None:
            model = load_robot_model()
        else:
            if not os.path.isfile(resolved):
                return FkContext(False, "FK_MODEL_LOAD_FAILED", None, None, None, None, ee_frame)
            model = load_robot_model(resolved)
    except Exception:
        return FkContext(False, "FK_MODEL_LOAD_FAILED", None, None, None, None, ee_frame)

    if not model.existFrame(ee_frame):
        return FkContext(False, "MISSING_EE_FRAME", model, None, None, None, ee_frame)

    q_list = [float(v) for v in initial_q]
    if len(q_list) != model.nq:
        return FkContext(False, "INVALID_INITIAL_Q", model, None, None, None, ee_frame)

    q_current = np.array(q_list, dtype=np.float64)
    data = model.createData()
    end_frame_id = int(model.getFrameId(ee_frame))
    return FkContext(True, "", model, data, end_frame_id, q_current, ee_frame)


@dataclass(frozen=True)
class InitialTargetPose:
    x: float
    y: float
    z: float
    from_fk: bool
    fallback_reason: str


def initial_target_pose_from_fk(
    fk_ctx: FkContext,
    fallback_x: float,
    fallback_y: float,
    fallback_z: float,
) -> InitialTargetPose:
    """Initialize conceptual target position from FK(q_current) or YAML fallback."""
    pose, fk_error = compute_fk_pose(fk_ctx)
    if pose is not None and not fk_error:
        return InitialTargetPose(
            x=float(pose.position.x),
            y=float(pose.position.y),
            z=float(pose.position.z),
            from_fk=True,
            fallback_reason="",
        )

    reason = fk_error or fk_ctx.error or "FK_NOT_READY"
    return InitialTargetPose(
        x=float(fallback_x),
        y=float(fallback_y),
        z=float(fallback_z),
        from_fk=False,
        fallback_reason=reason,
    )


def compute_fk_pose(ctx: FkContext) -> tuple[Pose | None, str]:
    if ctx.q_current is None:
        return None, ctx.error or "FK_NOT_READY"
    pose, _, err = compute_fk_pose_for_q(ctx, ctx.q_current)
    return pose, err


def compute_fk_pose_for_q(
    ctx: FkContext,
    q: np.ndarray,
) -> tuple[Pose | None, np.ndarray | None, str]:
    """Compute FK at q without mutating ctx.q_current."""
    if not ctx.ok or ctx.model is None:
        return None, None, ctx.error or "FK_NOT_READY"

    try:
        from reBotArm_control_py.kinematics import compute_fk

        q_arr = np.asarray(q, dtype=np.float64).reshape(ctx.model.nq)
        position, rotation, _ = compute_fk(ctx.model, q_arr, frame_name=ctx.ee_frame)
        rot = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        return fk_arrays_to_pose(position, rot), rot, ""
    except Exception:
        return None, None, "FK_COMPUTE_FAILED"
