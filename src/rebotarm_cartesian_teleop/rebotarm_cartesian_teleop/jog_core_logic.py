"""Pure Cartesian jog core logic (state machine, integration, clamps)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from geometry_msgs.msg import Pose
from rebotarm_msgs.msg import CartesianJogCmd, CartesianJogState

from .fk_kinematics import FkContext, compute_fk_pose_for_q
from .fk_pose import fk_arrays_to_pose
from .ik_kinematics import compute_ik_for_pose, compute_ik_for_position, joint_delta_within_limit

VALID_IK_TASK_MODES = frozenset({"full_6d", "position_only"})


def parse_ik_task_mode(value: str) -> str:
    mode = (value or "full_6d").strip().lower()
    if mode not in VALID_IK_TASK_MODES:
        allowed = ", ".join(sorted(VALID_IK_TASK_MODES))
        raise ValueError(f"Invalid ik_task_mode: {value!r}; allowed: {allowed}")
    return mode


@dataclass(frozen=True)
class IkConfig:
    max_iterations: int
    tolerance: float
    max_ik_error: float
    max_joint_delta_rad: float
    task_mode: str = "full_6d"


@dataclass(frozen=True)
class IkNoEffectConfig:
    candidate_step_min_m: float = 0.0005
    reached_step_min_m: float = 0.0001
    q_step_min_norm: float = 1e-6


@dataclass(frozen=True)
class IkNoEffectMetrics:
    candidate_step_m: float
    reached_step_m: float
    q_step_norm: float


@dataclass(frozen=True)
class JointLimitRejectConfig:
    reject_margin_rad: float = 0.05


@dataclass(frozen=True)
class JointNearLimitInfo:
    joint: str
    nearest_margin: float
    nearest_side: str


@dataclass(frozen=True)
class Joint1GlobalOperationalLimitConfig:
    enabled: bool = False
    min_rad: float = -1.60
    max_rad: float = 1.60


@dataclass(frozen=True)
class Joint1AnchorWindowConfig:
    enabled: bool = False
    hard_window_rad: float = 1.20


@dataclass(frozen=True)
class Joint1GlobalOperationalLimitInfo:
    joint1_q: float
    min_rad: float
    max_rad: float
    violation_side: str


@dataclass(frozen=True)
class Joint1AnchorWindowInfo:
    joint1_q: float
    base_anchor_q: float
    hard_window_rad: float
    abs_delta_from_anchor: float


@dataclass(frozen=True)
class IkFailureDiagnostics:
    rejection_reason: str
    candidate_target: tuple[float, float, float]
    seed_q: tuple[float, ...]
    ik_error: float | None
    ik_iterations: int | None
    max_ik_error: float
    max_joint_delta_rad: float
    clamp_reason: str
    state: str
    target_rotation_from_fk: bool
    committed_target: tuple[float, float, float] | None = None


def format_ik_failure_log(diag: IkFailureDiagnostics) -> str:
    cx, cy, cz = diag.committed_target if diag.committed_target is not None else (0.0, 0.0, 0.0)
    tx, ty, tz = diag.candidate_target
    committed_str = (
        f"({cx:.4f}, {cy:.4f}, {cz:.4f})" if diag.committed_target is not None else "n/a"
    )
    seed_str = ", ".join(f"{v:.4f}" for v in diag.seed_q)
    ik_error_str = f"{diag.ik_error:.6f}" if diag.ik_error is not None else "n/a"
    ik_iter_str = str(diag.ik_iterations) if diag.ik_iterations is not None else "n/a"
    clamp_str = diag.clamp_reason if diag.clamp_reason else "(none)"
    rot_src = "FK(q_sim)" if diag.target_rotation_from_fk else "other"
    return (
        f"IK failure: reason={diag.rejection_reason} state={diag.state} "
        f"committed={committed_str} candidate=({tx:.4f}, {ty:.4f}, {tz:.4f}) "
        f"seed_q=[{seed_str}] ik_error={ik_error_str} ik_iterations={ik_iter_str} "
        f"max_ik_error={diag.max_ik_error:.6f} max_joint_delta_rad={diag.max_joint_delta_rad:.4f} "
        f"clamp_reason={clamp_str} target_rotation={rot_src}"
    )


@dataclass(frozen=True)
class WorkspaceLimits:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


COMMAND_FRAME_LOCAL_WINDOW = "local_window_frame"


@dataclass(frozen=True)
class LocalWindowLimits:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    global_z_min: float
    global_z_max: float


@dataclass(frozen=True)
class LocalWindowState:
    base_anchor_q: float
    anchor_position: tuple[float, float, float]
    local_offset: tuple[float, float, float]


@dataclass(frozen=True)
class LocalWindowClampResult:
    local_offset: tuple[float, float, float]
    target_base_link: tuple[float, float, float]
    clamp_active: bool
    clamped_axes: tuple[str, ...]


@dataclass(frozen=True)
class BaseJogResult:
    q_after: list[float]
    delta_rad: float
    before_q: float
    after_q: float


def compute_local_axes(
    base_anchor_q: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    c = math.cos(float(base_anchor_q))
    s = math.sin(float(base_anchor_q))
    forward = (c, s, 0.0)
    lateral = (-s, c, 0.0)
    return forward, lateral


def integrate_local_target_offset(
    local_offset: Sequence[float],
    v_local: Sequence[float],
    dt: float,
) -> tuple[float, float, float]:
    ox, oy, oz = (float(local_offset[i]) for i in range(3))
    vx, vy, vz = (float(v_local[i]) for i in range(3))
    return (ox + vx * dt, oy + vy * dt, oz + vz * dt)


def clamp_local_target_offset(
    local_offset: Sequence[float],
    limits: LocalWindowLimits,
    anchor_z: float,
) -> tuple[tuple[float, float, float], bool, tuple[str, ...]]:
    """Clamp local x/y/z offsets; adjust oz if anchor_z+oz violates global Z."""
    ox, oy, oz = (float(local_offset[i]) for i in range(3))
    axes: list[str] = []

    ox, cx = clamp(ox, limits.x_min, limits.x_max)
    if cx:
        axes.append("LOCAL_X")
    oy, cy = clamp(oy, limits.y_min, limits.y_max)
    if cy:
        axes.append("LOCAL_Y")
    oz, cz = clamp(oz, limits.z_min, limits.z_max)
    if cz:
        axes.append("LOCAL_Z")

    tz_raw = float(anchor_z) + oz
    if tz_raw < limits.global_z_min:
        oz = limits.global_z_min - float(anchor_z)
        axes.append("GLOBAL_Z")
    elif tz_raw > limits.global_z_max:
        oz = limits.global_z_max - float(anchor_z)
        axes.append("GLOBAL_Z")

    return (ox, oy, oz), bool(axes), tuple(axes)


def local_target_to_base_link(
    anchor_position: Sequence[float],
    base_anchor_q: float,
    local_offset: Sequence[float],
    limits: LocalWindowLimits,
) -> LocalWindowClampResult:
    """Convert local offset to base_link IK target using explicit axis semantics."""
    ax, ay, az = (float(anchor_position[i]) for i in range(3))
    ox, oy, oz = (float(local_offset[i]) for i in range(3))
    axes: list[str] = []

    ox, cx = clamp(ox, limits.x_min, limits.x_max)
    if cx:
        axes.append("LOCAL_X")
    oy, cy = clamp(oy, limits.y_min, limits.y_max)
    if cy:
        axes.append("LOCAL_Y")
    oz, cz = clamp(oz, limits.z_min, limits.z_max)
    if cz:
        axes.append("LOCAL_Z")

    forward, lateral = compute_local_axes(base_anchor_q)
    tx = ax + forward[0] * ox + lateral[0] * oy
    ty = ay + forward[1] * ox + lateral[1] * oy
    tz_raw = az + oz
    tz, gz = clamp(tz_raw, limits.global_z_min, limits.global_z_max)
    if gz:
        axes.append("GLOBAL_Z")
        oz = tz - az

    return LocalWindowClampResult(
        local_offset=(ox, oy, oz),
        target_base_link=(tx, ty, tz),
        clamp_active=bool(axes),
        clamped_axes=tuple(axes),
    )


def reanchor_local_window_from_fk(
    *,
    fk_position: Sequence[float],
    joint1_q: float,
) -> LocalWindowState:
    return LocalWindowState(
        base_anchor_q=float(joint1_q),
        anchor_position=(
            float(fk_position[0]),
            float(fk_position[1]),
            float(fk_position[2]),
        ),
        local_offset=(0.0, 0.0, 0.0),
    )


def apply_base_joint1_jog(
    q: Sequence[float],
    *,
    joint1_index: int,
    velocity_rad_s: float,
    dt: float,
    min_rad: float,
    max_rad: float,
) -> BaseJogResult:
    qb = [float(v) for v in q]
    before_q = float(qb[joint1_index])
    after_q, _ = clamp(before_q + float(velocity_rad_s) * float(dt), min_rad, max_rad)
    qb[joint1_index] = after_q
    return BaseJogResult(
        q_after=qb,
        delta_rad=after_q - before_q,
        before_q=before_q,
        after_q=after_q,
    )


def integrate_local_window_candidate(
    local_state: LocalWindowState,
    v_local: Sequence[float],
    dt: float,
    limits: LocalWindowLimits,
) -> tuple[LocalWindowState, LocalWindowClampResult]:
    new_offset = integrate_local_target_offset(local_state.local_offset, v_local, dt)
    clamp_result = local_target_to_base_link(
        local_state.anchor_position,
        local_state.base_anchor_q,
        new_offset,
        limits,
    )
    updated = LocalWindowState(
        base_anchor_q=local_state.base_anchor_q,
        anchor_position=local_state.anchor_position,
        local_offset=clamp_result.local_offset,
    )
    return updated, clamp_result


def compute_state_name(
    latest_cmd: CartesianJogCmd | None,
    command_age: float,
    command_timeout_s: float,
) -> str:
    if latest_cmd is None:
        return "IDLE"

    if command_age > command_timeout_s:
        return "TIMEOUT"

    if latest_cmd.soft_stop:
        return "SOFT_STOP"

    if not latest_cmd.deadman:
        return "DEADMAN_UP"

    return "ACTIVE"


def resolve_rejection_reason(state_name: str, fk_error: str, ik_reason: str) -> str:
    state_reason = rejection_reason_for_state(state_name)
    if state_reason:
        return state_reason
    if fk_error:
        return fk_error
    return ik_reason


def rejection_reason_for_state(state_name: str) -> str:
    if state_name == "TIMEOUT":
        return "COMMAND_TIMEOUT"
    if state_name == "DEADMAN_UP":
        return "DEADMAN_UP"
    if state_name == "SOFT_STOP":
        return "SOFT_STOP"
    return ""


def update_base_anchor_q_on_deadman_rising(
    *,
    deadman_pressed: bool,
    prev_deadman_pressed: bool,
    joint1_q: float,
    base_anchor_q: float,
) -> tuple[float, bool]:
    """Diagnostic-only anchor policy: reset joint1 anchor on deadman rising edge."""
    new_anchor = base_anchor_q
    if deadman_pressed and not prev_deadman_pressed:
        new_anchor = float(joint1_q)
    return new_anchor, deadman_pressed


def clamp(value: float, min_value: float, max_value: float) -> tuple[float, bool]:
    if value < min_value:
        return min_value, True
    if value > max_value:
        return max_value, True
    return value, False


def compute_candidate_target(
    sim_x: float,
    sim_y: float,
    sim_z: float,
    latest_cmd: CartesianJogCmd,
    dt: float,
    workspace: WorkspaceLimits,
) -> tuple[float, float, float, str]:
    """Integrate joystick delta from FK(q_sim) position into a candidate target."""
    vx = float(latest_cmd.linear.x)
    vy = float(latest_cmd.linear.y)
    vz = float(latest_cmd.linear.z)

    candidate_x = sim_x + vx * dt
    candidate_y = sim_y + vy * dt
    candidate_z = sim_z + vz * dt

    clamp_reasons: list[str] = []

    candidate_x, clamped_x = clamp(candidate_x, workspace.x_min, workspace.x_max)
    candidate_y, clamped_y = clamp(candidate_y, workspace.y_min, workspace.y_max)
    candidate_z, clamped_z = clamp(candidate_z, workspace.z_min, workspace.z_max)

    if clamped_x:
        clamp_reasons.append("WORKSPACE_X")
    if clamped_y:
        clamp_reasons.append("WORKSPACE_Y")
    if clamped_z:
        clamp_reasons.append("WORKSPACE_Z")

    return candidate_x, candidate_y, candidate_z, ",".join(clamp_reasons)


def commit_target_on_ik_success(
    committed_x: float,
    committed_y: float,
    committed_z: float,
    candidate_x: float,
    candidate_y: float,
    candidate_z: float,
    ik_success: bool,
) -> tuple[float, float, float]:
    """Legacy helper: commit candidate on success (superseded by resync_committed_from_q_sim)."""
    if ik_success:
        return candidate_x, candidate_y, candidate_z
    return committed_x, committed_y, committed_z


def update_q_sim_on_ik_success(
    q_sim: np.ndarray,
    candidate_q: list[float],
    ik_success: bool,
) -> np.ndarray:
    if ik_success and candidate_q:
        return np.asarray(candidate_q, dtype=np.float64).reshape(q_sim.shape)
    return np.asarray(q_sim, dtype=np.float64)


def compute_candidate_drift_m(
    fk_ctx: FkContext,
    candidate_x: float,
    candidate_y: float,
    candidate_z: float,
    candidate_q: list[float],
) -> float:
    """Position drift between ideal candidate and FK(candidate_q) (diagnostics only)."""
    if not candidate_q:
        return 0.0
    pose, _, err = compute_fk_pose_for_q(fk_ctx, np.asarray(candidate_q, dtype=np.float64))
    if err or pose is None:
        return 0.0
    dx = candidate_x - float(pose.position.x)
    dy = candidate_y - float(pose.position.y)
    dz = candidate_z - float(pose.position.z)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def resync_committed_from_q_sim(
    fk_ctx: FkContext,
    q_sim: np.ndarray,
) -> tuple[float, float, float, np.ndarray | None, Pose | None, str]:
    """Set committed pose from FK(q_sim). Always used after accepted IK."""
    pose, rotation, err = compute_fk_pose_for_q(fk_ctx, q_sim)
    if err or pose is None or rotation is None:
        return 0.0, 0.0, 0.0, None, None, err or "FK_NOT_READY"
    return (
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
        rotation,
        pose,
        "",
    )


def build_committed_target_pose(
    x: float,
    y: float,
    z: float,
    rotation: np.ndarray | None,
) -> Pose:
    if rotation is not None:
        pos = np.array([x, y, z], dtype=np.float64)
        return fk_arrays_to_pose(pos, rotation)
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    pose.orientation.w = 1.0
    return pose


def integrate_target_pose(
    target_x: float,
    target_y: float,
    target_z: float,
    latest_cmd: CartesianJogCmd | None,
    dt: float,
    state_name: str,
    workspace: WorkspaceLimits,
) -> tuple[float, float, float, str]:
    if state_name != "ACTIVE" or latest_cmd is None:
        return target_x, target_y, target_z, ""

    return compute_candidate_target(target_x, target_y, target_z, latest_cmd, dt, workspace)


def _ik_failure_diagnostics(
    *,
    reason: str,
    candidate_x: float,
    candidate_y: float,
    candidate_z: float,
    q_seed: np.ndarray,
    ik_config: IkConfig,
    state_name: str,
    clamp_reason: str,
    committed_x: float | None = None,
    committed_y: float | None = None,
    committed_z: float | None = None,
    ik_error: float | None = None,
    ik_iterations: int | None = None,
) -> IkFailureDiagnostics:
    committed = None
    if committed_x is not None and committed_y is not None and committed_z is not None:
        committed = (committed_x, committed_y, committed_z)
    return IkFailureDiagnostics(
        rejection_reason=reason,
        candidate_target=(candidate_x, candidate_y, candidate_z),
        seed_q=tuple(float(v) for v in q_seed),
        ik_error=ik_error,
        ik_iterations=ik_iterations,
        max_ik_error=ik_config.max_ik_error,
        max_joint_delta_rad=ik_config.max_joint_delta_rad,
        clamp_reason=clamp_reason,
        state=state_name,
        target_rotation_from_fk=True,
        committed_target=committed,
    )


def solve_target_ik(
    *,
    fk_ctx: FkContext,
    state_name: str,
    target_x: float,
    target_y: float,
    target_z: float,
    target_rotation: np.ndarray,
    q_seed: np.ndarray,
    ik_config: IkConfig,
    clamp_reason: str = "",
    committed_x: float | None = None,
    committed_y: float | None = None,
    committed_z: float | None = None,
) -> tuple[list[float], bool, str, IkFailureDiagnostics | None]:
    """Compute q_target from candidate position and FK(q_sim) orientation."""
    if state_name != "ACTIVE" or not fk_ctx.ok:
        return [], False, "", None

    if (
        fk_ctx.model is None
        or fk_ctx.data is None
        or fk_ctx.end_frame_id is None
    ):
        return [], False, "", None

    target_pos = np.array([target_x, target_y, target_z], dtype=np.float64)
    target_rot = np.asarray(target_rotation, dtype=np.float64).reshape(3, 3)
    q_seed_arr = np.asarray(q_seed, dtype=np.float64).reshape(fk_ctx.model.nq)

    if ik_config.task_mode == "position_only":
        ik_result = compute_ik_for_position(
            fk_ctx.model,
            fk_ctx.data,
            fk_ctx.end_frame_id,
            target_pos,
            q_seed_arr,
            ik_config.max_iterations,
            ik_config.tolerance,
            ik_config.max_ik_error,
        )
    else:
        ik_result = compute_ik_for_pose(
            fk_ctx.model,
            fk_ctx.data,
            fk_ctx.end_frame_id,
            target_pos,
            target_rot,
            q_seed_arr,
            ik_config.max_iterations,
            ik_config.tolerance,
            ik_config.max_ik_error,
        )

    if not ik_result.success:
        diag = _ik_failure_diagnostics(
            reason=ik_result.reason,
            candidate_x=target_x,
            candidate_y=target_y,
            candidate_z=target_z,
            q_seed=q_seed_arr,
            ik_config=ik_config,
            state_name=state_name,
            clamp_reason=clamp_reason,
            committed_x=committed_x,
            committed_y=committed_y,
            committed_z=committed_z,
            ik_error=ik_result.error,
            ik_iterations=ik_result.iterations,
        )
        return [], False, ik_result.reason, diag

    if len(ik_result.q_target) != fk_ctx.model.nq:
        diag = _ik_failure_diagnostics(
            reason="INVALID_IK_RESULT",
            candidate_x=target_x,
            candidate_y=target_y,
            candidate_z=target_z,
            q_seed=q_seed_arr,
            ik_config=ik_config,
            state_name=state_name,
            clamp_reason=clamp_reason,
            committed_x=committed_x,
            committed_y=committed_y,
            committed_z=committed_z,
            ik_error=ik_result.error,
            ik_iterations=ik_result.iterations,
        )
        return [], False, "INVALID_IK_RESULT", diag

    if not joint_delta_within_limit(
        ik_result.q_target,
        q_seed_arr,
        ik_config.max_joint_delta_rad,
    ):
        diag = _ik_failure_diagnostics(
            reason="JOINT_DELTA_TOO_LARGE",
            candidate_x=target_x,
            candidate_y=target_y,
            candidate_z=target_z,
            q_seed=q_seed_arr,
            ik_config=ik_config,
            state_name=state_name,
            clamp_reason=clamp_reason,
            committed_x=committed_x,
            committed_y=committed_y,
            committed_z=committed_z,
            ik_error=ik_result.error,
            ik_iterations=ik_result.iterations,
        )
        return [], False, "JOINT_DELTA_TOO_LARGE", diag

    return list(ik_result.q_target), True, "", None


def _position_step_m(
    from_pos: tuple[float, float, float],
    to_x: float,
    to_y: float,
    to_z: float,
) -> float:
    dx = to_x - from_pos[0]
    dy = to_y - from_pos[1]
    dz = to_z - from_pos[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def compute_ik_no_effect_metrics(
    fk_ctx: FkContext,
    q_sim_before: np.ndarray,
    candidate_q: list[float] | np.ndarray,
    candidate_x: float,
    candidate_y: float,
    candidate_z: float,
    fk_position_before: tuple[float, float, float] | None = None,
) -> IkNoEffectMetrics:
    """Measure candidate vs reached motion after solver-reported IK success."""
    qb = np.asarray(q_sim_before, dtype=np.float64).reshape(-1)
    qc = np.asarray(candidate_q, dtype=np.float64).reshape(-1)

    if fk_position_before is None:
        pose, _, err = compute_fk_pose_for_q(fk_ctx, qb)
        if err or pose is None:
            return IkNoEffectMetrics(0.0, 0.0, 0.0)
        fk_position_before = (
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
        )

    cand_step = _position_step_m(fk_position_before, candidate_x, candidate_y, candidate_z)

    fk_cand_pose, _, fk_err = compute_fk_pose_for_q(fk_ctx, qc)
    if fk_err or fk_cand_pose is None:
        reached_step = 0.0
    else:
        reached_step = _position_step_m(
            fk_position_before,
            float(fk_cand_pose.position.x),
            float(fk_cand_pose.position.y),
            float(fk_cand_pose.position.z),
        )

    return IkNoEffectMetrics(
        candidate_step_m=cand_step,
        reached_step_m=reached_step,
        q_step_norm=float(np.linalg.norm(qc - qb)),
    )


def is_ik_no_effect(metrics: IkNoEffectMetrics, config: IkNoEffectConfig) -> bool:
    return (
        metrics.candidate_step_m > config.candidate_step_min_m
        and metrics.reached_step_m < config.reached_step_min_m
        and metrics.q_step_norm < config.q_step_min_norm
    )


def reject_ik_if_no_effect(
    fk_ctx: FkContext,
    q_sim_before: np.ndarray,
    candidate_q: list[float],
    candidate_x: float,
    candidate_y: float,
    candidate_z: float,
    config: IkNoEffectConfig,
    fk_position_before: tuple[float, float, float] | None = None,
) -> tuple[list[float], bool, str, IkNoEffectMetrics | None]:
    """After solver success: reject phantom-success IK that produces no joint motion."""
    if not candidate_q:
        return [], False, "", None

    metrics = compute_ik_no_effect_metrics(
        fk_ctx,
        q_sim_before,
        candidate_q,
        candidate_x,
        candidate_y,
        candidate_z,
        fk_position_before=fk_position_before,
    )
    if is_ik_no_effect(metrics, config):
        return [], False, "IK_NO_EFFECT", metrics
    return candidate_q, True, "", metrics


def compute_nearest_joint_limit_margin(
    q: float,
    lower: float,
    upper: float,
) -> tuple[float, str]:
    margin_to_lower = q - lower
    margin_to_upper = upper - q
    if margin_to_lower <= margin_to_upper:
        return float(margin_to_lower), "lower"
    return float(margin_to_upper), "upper"


def find_joint_near_limit_violation(
    candidate_q: Sequence[float],
    joint_names: Sequence[str],
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    reject_margin_rad: float,
) -> JointNearLimitInfo | None:
    """Return info for the joint closest to a hard limit below the reject threshold."""
    qt = np.asarray(candidate_q, dtype=np.float64).reshape(-1)
    lo = np.asarray(lower_limits, dtype=np.float64).reshape(-1)
    hi = np.asarray(upper_limits, dtype=np.float64).reshape(-1)
    if not (len(joint_names) == len(qt) == len(lo) == len(hi)):
        return None

    worst: JointNearLimitInfo | None = None
    for i, name in enumerate(joint_names):
        margin, side = compute_nearest_joint_limit_margin(float(qt[i]), float(lo[i]), float(hi[i]))
        if margin >= float(reject_margin_rad):
            continue
        if worst is None or margin < worst.nearest_margin:
            worst = JointNearLimitInfo(joint=str(name), nearest_margin=margin, nearest_side=side)
    return worst


def _joint1_index(joint_names: Sequence[str]) -> int | None:
    try:
        return list(joint_names).index("joint1")
    except ValueError:
        return None


def reject_ik_if_joint1_global_operational_limit(
    candidate_q: list[float],
    joint_names: Sequence[str],
    config: Joint1GlobalOperationalLimitConfig,
) -> tuple[list[float], bool, str, Joint1GlobalOperationalLimitInfo | None]:
    """After solver success: reject joint1 outside operational cap."""
    if not config.enabled or not candidate_q:
        return candidate_q, True, "", None

    j1_idx = _joint1_index(joint_names)
    if j1_idx is None or j1_idx >= len(candidate_q):
        return candidate_q, True, "", None

    joint1_q = float(candidate_q[j1_idx])
    if joint1_q < float(config.min_rad):
        return (
            [],
            False,
            "JOINT1_GLOBAL_OPERATIONAL_LIMIT",
            Joint1GlobalOperationalLimitInfo(
                joint1_q=joint1_q,
                min_rad=float(config.min_rad),
                max_rad=float(config.max_rad),
                violation_side="lower",
            ),
        )
    if joint1_q > float(config.max_rad):
        return (
            [],
            False,
            "JOINT1_GLOBAL_OPERATIONAL_LIMIT",
            Joint1GlobalOperationalLimitInfo(
                joint1_q=joint1_q,
                min_rad=float(config.min_rad),
                max_rad=float(config.max_rad),
                violation_side="upper",
            ),
        )
    return candidate_q, True, "", None


def reject_ik_if_joint1_anchor_window(
    candidate_q: list[float],
    joint_names: Sequence[str],
    base_anchor_q: float,
    config: Joint1AnchorWindowConfig,
) -> tuple[list[float], bool, str, Joint1AnchorWindowInfo | None]:
    """After solver success: reject joint1 drift beyond anchor hard window."""
    if not config.enabled or not candidate_q:
        return candidate_q, True, "", None

    j1_idx = _joint1_index(joint_names)
    if j1_idx is None or j1_idx >= len(candidate_q):
        return candidate_q, True, "", None

    joint1_q = float(candidate_q[j1_idx])
    abs_delta = abs(joint1_q - float(base_anchor_q))
    if abs_delta > float(config.hard_window_rad):
        return (
            [],
            False,
            "JOINT1_ANCHOR_WINDOW",
            Joint1AnchorWindowInfo(
                joint1_q=joint1_q,
                base_anchor_q=float(base_anchor_q),
                hard_window_rad=float(config.hard_window_rad),
                abs_delta_from_anchor=abs_delta,
            ),
        )
    return candidate_q, True, "", None


def reject_ik_if_near_joint_limit(
    candidate_q: list[float],
    joint_names: Sequence[str],
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    config: JointLimitRejectConfig,
) -> tuple[list[float], bool, str, JointNearLimitInfo | None]:
    """After solver success: reject IK candidates too close to any joint hard limit."""
    if not candidate_q or not joint_names:
        return candidate_q, True, "", None

    violation = find_joint_near_limit_violation(
        candidate_q,
        joint_names,
        lower_limits,
        upper_limits,
        config.reject_margin_rad,
    )
    if violation is not None:
        return [], False, "JOINT_NEAR_LIMIT", violation
    return candidate_q, True, "", None


@dataclass(frozen=True)
class IkGateSequenceInput:
    """Inputs for post-IK safety gate sequence (pure, no ROS)."""

    fk_ctx: FkContext
    q_candidate: list[float]
    q_before: np.ndarray
    joint_names: list[str]
    lower_limits: list[float]
    upper_limits: list[float]
    base_anchor_q: float
    candidate_x: float
    candidate_y: float
    candidate_z: float
    fk_position_before: tuple[float, float, float] | None
    joint1_global_config: Joint1GlobalOperationalLimitConfig
    joint1_anchor_config: Joint1AnchorWindowConfig
    joint_limit_config: JointLimitRejectConfig
    ik_no_effect_config: IkNoEffectConfig


@dataclass(frozen=True)
class IkGateSequenceResult:
    """Outcome of post-IK gate sequence."""

    accepted: bool
    q_candidate: list[float]
    rejection_reason: str
    rejection_source: str
    gate_name: str
    global_cap_info: Joint1GlobalOperationalLimitInfo | None = None
    anchor_info: Joint1AnchorWindowInfo | None = None
    joint_limit_info: JointNearLimitInfo | None = None
    no_effect_metrics: IkNoEffectMetrics | None = None
    q_from_ik: list[float] | None = None


def apply_ik_gate_sequence(inp: IkGateSequenceInput) -> IkGateSequenceResult:
    """Apply post-IK gates in fixed order; first failure wins.

    Order (must not change):
    1. JOINT1_GLOBAL_OPERATIONAL_LIMIT
    2. JOINT1_ANCHOR_WINDOW
    3. JOINT_NEAR_LIMIT
    4. IK_NO_EFFECT
    """
    q = list(inp.q_candidate)
    q_from_ik = list(q)

    q, ok, reason, global_cap_info = reject_ik_if_joint1_global_operational_limit(
        q,
        inp.joint_names,
        inp.joint1_global_config,
    )
    if not ok:
        return IkGateSequenceResult(
            accepted=False,
            q_candidate=[],
            rejection_reason=reason,
            rejection_source="JOINT1_GLOBAL_OPERATIONAL_LIMIT",
            gate_name="JOINT1_GLOBAL_OPERATIONAL_LIMIT",
            global_cap_info=global_cap_info,
            q_from_ik=q_from_ik,
        )

    q_from_ik = list(q)
    q, ok, reason, anchor_info = reject_ik_if_joint1_anchor_window(
        q,
        inp.joint_names,
        inp.base_anchor_q,
        inp.joint1_anchor_config,
    )
    if not ok:
        return IkGateSequenceResult(
            accepted=False,
            q_candidate=[],
            rejection_reason=reason,
            rejection_source="JOINT1_ANCHOR_WINDOW",
            gate_name="JOINT1_ANCHOR_WINDOW",
            anchor_info=anchor_info,
            q_from_ik=q_from_ik,
        )

    q_from_ik = list(q)
    q, ok, reason, joint_limit_info = reject_ik_if_near_joint_limit(
        q,
        inp.joint_names,
        inp.lower_limits,
        inp.upper_limits,
        inp.joint_limit_config,
    )
    if not ok:
        return IkGateSequenceResult(
            accepted=False,
            q_candidate=[],
            rejection_reason=reason,
            rejection_source="JOINT_NEAR_LIMIT",
            gate_name="JOINT_NEAR_LIMIT",
            joint_limit_info=joint_limit_info,
            q_from_ik=q_from_ik,
        )

    q_from_ik = list(q)
    q, ok, reason, no_effect_metrics = reject_ik_if_no_effect(
        inp.fk_ctx,
        inp.q_before,
        q,
        inp.candidate_x,
        inp.candidate_y,
        inp.candidate_z,
        inp.ik_no_effect_config,
        fk_position_before=inp.fk_position_before,
    )
    if not ok:
        return IkGateSequenceResult(
            accepted=False,
            q_candidate=[],
            rejection_reason=reason,
            rejection_source="IK_NO_EFFECT",
            gate_name="IK_NO_EFFECT",
            no_effect_metrics=no_effect_metrics,
            q_from_ik=q_from_ik,
        )

    return IkGateSequenceResult(
        accepted=True,
        q_candidate=q,
        rejection_reason="",
        rejection_source="",
        gate_name="",
        no_effect_metrics=no_effect_metrics,
        q_from_ik=None,
    )


def format_joint_near_limit_log(info: JointNearLimitInfo) -> str:
    return (
        "IK rejection: reason=JOINT_NEAR_LIMIT "
        f"nearest_limit_joint={info.joint} "
        f"nearest_limit_margin={info.nearest_margin:.4f} "
        f"nearest_limit_side={info.nearest_side}"
    )


def format_joint1_global_operational_limit_log(info: Joint1GlobalOperationalLimitInfo) -> str:
    return (
        "IK rejection: reason=JOINT1_GLOBAL_OPERATIONAL_LIMIT "
        f"joint1_q={info.joint1_q:.4f} "
        f"range=[{info.min_rad:.2f},{info.max_rad:.2f}] "
        f"violation_side={info.violation_side}"
    )


def format_joint1_anchor_window_log(info: Joint1AnchorWindowInfo) -> str:
    return (
        "IK rejection: reason=JOINT1_ANCHOR_WINDOW "
        f"joint1_q={info.joint1_q:.4f} "
        f"base_anchor_q={info.base_anchor_q:.4f} "
        f"hard_window_rad={info.hard_window_rad:.4f} "
        f"abs_delta_from_anchor={info.abs_delta_from_anchor:.4f}"
    )


def build_cartesian_jog_state(
    *,
    state_name: str,
    target_x: float,
    target_y: float,
    target_z: float,
    latest_cmd: CartesianJogCmd | None,
    clamp_reason: str,
    dry_run: bool,
    output_mode: str,
    command_age: float,
    current_pose: Pose | None = None,
    target_pose: Pose | None = None,
    q_current: list[float] | None = None,
    q_target: list[float] | None = None,
    ik_success: bool = False,
    fk_error: str = "",
    ik_reason: str = "",
) -> CartesianJogState:
    msg = CartesianJogState()
    msg.header.frame_id = "base_link"
    msg.state = state_name

    if current_pose is not None:
        msg.current_pose = current_pose
    else:
        msg.current_pose.position.x = target_x
        msg.current_pose.position.y = target_y
        msg.current_pose.position.z = target_z
        msg.current_pose.orientation.w = 1.0

    if target_pose is not None:
        msg.target_pose = target_pose
    else:
        msg.target_pose.position.x = target_x
        msg.target_pose.position.y = target_y
        msg.target_pose.position.z = target_z
        msg.target_pose.orientation.w = 1.0

    if latest_cmd is not None:
        msg.commanded_twist.linear = latest_cmd.linear
        msg.commanded_twist.angular = latest_cmd.angular

    msg.q_current = [float(v) for v in q_current] if q_current is not None else []
    msg.q_target = [float(v) for v in q_target] if q_target is not None else []
    msg.ik_success = bool(ik_success)
    msg.rejection_reason = resolve_rejection_reason(state_name, fk_error, ik_reason)
    msg.clamp_reason = clamp_reason
    msg.dry_run = dry_run
    msg.output_mode = output_mode
    msg.command_age_s = command_age if command_age != math.inf else -1.0

    return msg
