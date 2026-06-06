"""Pure IK solution quality diagnostics (logging only, no acceptance policy)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_JOINT_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
)


@dataclass(frozen=True)
class IkQualityLogConfig:
    """Thresholds for warning logs only; never gate IK acceptance."""

    joint_limit_warn_margin_rad: float = 0.35
    joint5_warn_abs_rad: float = 1.0
    joint4_warn_abs_rad: float = 1.0
    q_delta_warn_rad: float = 0.15
    candidate_drift_warn_m: float = 0.003
    reached_step_warn_min_m: float = 0.0001
    enable_cartesian_joint1_window_diagnostics: bool = True
    cartesian_joint1_window_warning_rad: float = 0.25
    cartesian_joint1_window_hard_rad: float = 1.20
    enable_joint1_anchor_hard_gate: bool = False
    enable_joint1_global_operational_cap: bool = False
    joint1_global_operational_min_rad: float = -1.60
    joint1_global_operational_max_rad: float = 1.60
    joint1_large_delta_from_anchor_rad: float = 0.15


@dataclass(frozen=True)
class BaseSectorDiagnostics:
    """Diagnostic-only joint1/base sector fields (never gate IK acceptance)."""

    command_frame: str
    command_frame_kind: str
    workspace_frame: str
    cartesian_command_linear_x: float
    cartesian_command_linear_y: float
    cartesian_command_linear_z: float
    base_anchor_q: float
    joint1_current_q: float
    joint1_candidate_q: float
    joint1_min_limit: float
    joint1_max_limit: float
    joint1_margin_to_lower: float
    joint1_margin_to_upper: float
    joint1_delta_from_anchor: float
    abs_joint1_delta_from_anchor: float
    joint1_would_violate_warning_window: bool
    joint1_warning_window_rad: float
    joint1_warning_window_error_rad: float
    joint1_would_violate_hard_window: bool
    joint1_hard_window_rad: float
    joint1_hard_window_error_rad: float
    joint1_would_violate_global_cap: bool
    joint1_global_operational_min_rad: float
    joint1_global_operational_max_rad: float
    joint1_global_cap_error_rad: float
    enable_joint1_anchor_hard_gate: bool
    enable_joint1_global_operational_cap: bool
    joint1_delta_this_tick: float
    joint1_candidate_moved_this_tick: bool
    q_before_joint1: float
    q_committed_or_current_joint1: float
    workspace_clamp_active: bool
    workspace_clamped_axes: tuple[str, ...]
    raw_candidate_position: tuple[float, float, float]
    clamped_candidate_position: tuple[float, float, float]
    accepted_fk_position: tuple[float, float, float] | None
    rejected_candidate_position: tuple[float, float, float] | None
    workspace_clamp_with_large_joint1_delta: bool
    outcome: str
    rejection_source: str
    nearest_limit_joint: str
    nearest_limit_margin: float
    posture_distance_from_initial_q: float


@dataclass(frozen=True)
class JointQualityDiagnostic:
    name: str
    q_before: float
    q_target: float
    q_delta: float
    abs_q_delta: float
    lower_limit: float
    upper_limit: float
    margin_to_lower: float
    margin_to_upper: float
    nearest_margin: float
    nearest_side: str


@dataclass(frozen=True)
class IkQualityDiagnostics:
    joints: tuple[JointQualityDiagnostic, ...]
    max_abs_q_delta: float
    max_abs_q_delta_joint: str
    nearest_limit_joint: str
    nearest_limit_margin: float
    nearest_limit_side: str
    any_joint_near_limit: bool
    candidate_drift_m: float
    ik_error: float
    fk_position_before: tuple[float, float, float]
    fk_position_target: tuple[float, float, float]
    reached_step_m: float
    candidate_step_m: float
    q_step_norm: float
    posture_distance_from_initial_q: float
    joint4: JointQualityDiagnostic
    joint5: JointQualityDiagnostic
    log_reasons: tuple[str, ...]


def joint_names_from_model(model) -> list[str]:
    """Return revolute joint names in Pinocchio q order."""
    return [str(model.names[i + 1]) for i in range(model.nq)]


def joint_limits_from_model(model) -> tuple[list[float], list[float]]:
    lower = [float(model.lowerPositionLimit[i]) for i in range(model.nq)]
    upper = [float(model.upperPositionLimit[i]) for i in range(model.nq)]
    return lower, upper


def _nearest_side(margin_to_lower: float, margin_to_upper: float) -> str:
    return "lower" if margin_to_lower <= margin_to_upper else "upper"


def joint1_index(joint_names: Sequence[str]) -> int | None:
    try:
        return list(joint_names).index("joint1")
    except ValueError:
        return None


def _window_error_rad(abs_delta: float, window_rad: float) -> float:
    return max(0.0, abs_delta - float(window_rad))


def _global_cap_error_rad(
    joint1_q: float,
    *,
    min_rad: float,
    max_rad: float,
) -> float:
    if joint1_q < min_rad:
        return float(min_rad - joint1_q)
    if joint1_q > max_rad:
        return float(joint1_q - max_rad)
    return 0.0


def resolve_joint1_warning_window_rad(
    *,
    warning_rad: float,
    legacy_window_rad: float,
    default_rad: float = 0.25,
) -> float:
    """Backward-compatible alias: cartesian_joint1_window_rad -> warning threshold."""
    if warning_rad != default_rad or legacy_window_rad == default_rad:
        return float(warning_rad)
    return float(legacy_window_rad)


def compute_base_sector_diagnostics(
    *,
    joint_names: Sequence[str],
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    q_before: Sequence[float],
    q_target_or_current: Sequence[float],
    base_anchor_q: float,
    warning_window_rad: float,
    hard_window_rad: float,
    global_cap_min_rad: float,
    global_cap_max_rad: float,
    enable_joint1_anchor_hard_gate: bool = False,
    enable_joint1_global_operational_cap: bool = False,
    command_frame: str,
    workspace_frame: str,
    cartesian_command_linear_x: float,
    cartesian_command_linear_y: float,
    cartesian_command_linear_z: float,
    raw_candidate_position: Sequence[float],
    clamped_candidate_position: Sequence[float],
    workspace_clamp_active: bool,
    workspace_clamped_axes: Sequence[str],
    ik_accepted: bool,
    rejection_source: str,
    nearest_limit_joint: str,
    nearest_limit_margin: float,
    posture_distance_from_initial_q: float,
    q_candidate_from_ik: Sequence[float] | None = None,
    accepted_fk_position: Sequence[float] | None = None,
    large_delta_threshold_rad: float = 0.15,
) -> BaseSectorDiagnostics | None:
    """Build joint1/base sector diagnostics without affecting control decisions."""
    j1_idx = joint1_index(joint_names)
    if j1_idx is None:
        return None

    qb = np.asarray(q_before, dtype=np.float64).reshape(-1)
    qt = np.asarray(q_target_or_current, dtype=np.float64).reshape(-1)
    lo = float(lower_limits[j1_idx])
    hi = float(upper_limits[j1_idx])

    joint1_current_q = float(qb[j1_idx])
    q_before_joint1 = joint1_current_q
    q_committed_or_current_joint1 = float(qt[j1_idx])

    if q_candidate_from_ik is not None and len(q_candidate_from_ik) > j1_idx:
        joint1_candidate_q = float(q_candidate_from_ik[j1_idx])
    elif ik_accepted:
        joint1_candidate_q = q_committed_or_current_joint1
    else:
        joint1_candidate_q = joint1_current_q

    joint1_delta_this_tick = joint1_candidate_q - q_before_joint1
    joint1_candidate_moved_this_tick = abs(joint1_delta_this_tick) > 1e-9

    joint1_delta_from_anchor = joint1_candidate_q - float(base_anchor_q)
    abs_joint1_delta_from_anchor = abs(joint1_delta_from_anchor)
    joint1_would_violate_warning = (
        abs_joint1_delta_from_anchor > float(warning_window_rad)
    )
    joint1_warning_window_error_rad = _window_error_rad(
        abs_joint1_delta_from_anchor,
        warning_window_rad,
    )
    joint1_would_violate_hard = abs_joint1_delta_from_anchor > float(hard_window_rad)
    joint1_hard_window_error_rad = _window_error_rad(
        abs_joint1_delta_from_anchor,
        hard_window_rad,
    )
    joint1_global_cap_error_rad = _global_cap_error_rad(
        joint1_candidate_q,
        min_rad=global_cap_min_rad,
        max_rad=global_cap_max_rad,
    )
    joint1_would_violate_global_cap = joint1_global_cap_error_rad > 0.0

    margin_to_lower = joint1_candidate_q - lo
    margin_to_upper = hi - joint1_candidate_q

    raw_pos = tuple(float(v) for v in raw_candidate_position)
    clamped_pos = tuple(float(v) for v in clamped_candidate_position)
    axes = tuple(str(axis) for axis in workspace_clamped_axes)

    accepted_pos = None
    if accepted_fk_position is not None:
        accepted_pos = (
            float(accepted_fk_position[0]),
            float(accepted_fk_position[1]),
            float(accepted_fk_position[2]),
        )
    rejected_pos = clamped_pos if rejection_source else None

    return BaseSectorDiagnostics(
        command_frame=command_frame,
        command_frame_kind="base_frame",
        workspace_frame=workspace_frame,
        cartesian_command_linear_x=float(cartesian_command_linear_x),
        cartesian_command_linear_y=float(cartesian_command_linear_y),
        cartesian_command_linear_z=float(cartesian_command_linear_z),
        base_anchor_q=float(base_anchor_q),
        joint1_current_q=joint1_current_q,
        joint1_candidate_q=joint1_candidate_q,
        joint1_min_limit=lo,
        joint1_max_limit=hi,
        joint1_margin_to_lower=float(margin_to_lower),
        joint1_margin_to_upper=float(margin_to_upper),
        joint1_delta_from_anchor=float(joint1_delta_from_anchor),
        abs_joint1_delta_from_anchor=float(abs_joint1_delta_from_anchor),
        joint1_would_violate_warning_window=joint1_would_violate_warning,
        joint1_warning_window_rad=float(warning_window_rad),
        joint1_warning_window_error_rad=float(joint1_warning_window_error_rad),
        joint1_would_violate_hard_window=joint1_would_violate_hard,
        joint1_hard_window_rad=float(hard_window_rad),
        joint1_hard_window_error_rad=float(joint1_hard_window_error_rad),
        joint1_would_violate_global_cap=joint1_would_violate_global_cap,
        joint1_global_operational_min_rad=float(global_cap_min_rad),
        joint1_global_operational_max_rad=float(global_cap_max_rad),
        joint1_global_cap_error_rad=float(joint1_global_cap_error_rad),
        enable_joint1_anchor_hard_gate=bool(enable_joint1_anchor_hard_gate),
        enable_joint1_global_operational_cap=bool(enable_joint1_global_operational_cap),
        joint1_delta_this_tick=float(joint1_delta_this_tick),
        joint1_candidate_moved_this_tick=joint1_candidate_moved_this_tick,
        q_before_joint1=q_before_joint1,
        q_committed_or_current_joint1=q_committed_or_current_joint1,
        workspace_clamp_active=bool(workspace_clamp_active),
        workspace_clamped_axes=axes,
        raw_candidate_position=raw_pos,
        clamped_candidate_position=clamped_pos,
        accepted_fk_position=accepted_pos,
        rejected_candidate_position=rejected_pos,
        workspace_clamp_with_large_joint1_delta=(
            workspace_clamp_active
            and abs_joint1_delta_from_anchor > float(large_delta_threshold_rad)
        ),
        outcome="accepted" if ik_accepted else "rejected",
        rejection_source=str(rejection_source),
        nearest_limit_joint=str(nearest_limit_joint),
        nearest_limit_margin=float(nearest_limit_margin),
        posture_distance_from_initial_q=float(posture_distance_from_initial_q),
    )


def _joint_diagnostic(
    name: str,
    q_before: float,
    q_target: float,
    lower_limit: float,
    upper_limit: float,
) -> JointQualityDiagnostic:
    q_delta = q_target - q_before
    margin_to_lower = q_target - lower_limit
    margin_to_upper = upper_limit - q_target
    nearest_margin = min(margin_to_lower, margin_to_upper)
    return JointQualityDiagnostic(
        name=name,
        q_before=q_before,
        q_target=q_target,
        q_delta=q_delta,
        abs_q_delta=abs(q_delta),
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        margin_to_lower=margin_to_lower,
        margin_to_upper=margin_to_upper,
        nearest_margin=nearest_margin,
        nearest_side=_nearest_side(margin_to_lower, margin_to_upper),
    )


def compute_joint_quality_diagnostics(
    joint_names: Sequence[str],
    q_before: Sequence[float],
    q_target: Sequence[float],
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    initial_q: Sequence[float],
    *,
    fk_position_before: Sequence[float],
    fk_position_target: Sequence[float],
    candidate_drift_m: float = 0.0,
    ik_error: float = 0.0,
    candidate_step_m: float = 0.0,
    joint_limit_near_rad: float = 0.35,
) -> IkQualityDiagnostics:
    """Build per-joint and global IK quality diagnostics without mutating inputs."""
    qb = np.asarray(q_before, dtype=np.float64).reshape(-1)
    qt = np.asarray(q_target, dtype=np.float64).reshape(-1)
    q0 = np.asarray(initial_q, dtype=np.float64).reshape(-1)
    lo = np.asarray(lower_limits, dtype=np.float64).reshape(-1)
    hi = np.asarray(upper_limits, dtype=np.float64).reshape(-1)

    if not (len(joint_names) == len(qb) == len(qt) == len(lo) == len(hi) == len(q0)):
        raise ValueError("joint_names, q arrays, limits, and initial_q length mismatch")

    joints = tuple(
        _joint_diagnostic(
            str(joint_names[i]),
            float(qb[i]),
            float(qt[i]),
            float(lo[i]),
            float(hi[i]),
        )
        for i in range(len(joint_names))
    )

    max_idx = max(range(len(joints)), key=lambda i: joints[i].abs_q_delta)
    nearest_idx = min(range(len(joints)), key=lambda i: joints[i].nearest_margin)

    pos_before = np.asarray(fk_position_before, dtype=np.float64).reshape(3)
    pos_target = np.asarray(fk_position_target, dtype=np.float64).reshape(3)
    reached_step_m = float(np.linalg.norm(pos_target - pos_before))

    joint_by_name = {j.name: j for j in joints}
    j4_name = "joint4" if "joint4" in joint_by_name else joint_names[3]
    j5_name = "joint5" if "joint5" in joint_by_name else joint_names[4]

    any_near = any(j.nearest_margin < joint_limit_near_rad for j in joints)

    return IkQualityDiagnostics(
        joints=joints,
        max_abs_q_delta=joints[max_idx].abs_q_delta,
        max_abs_q_delta_joint=joints[max_idx].name,
        nearest_limit_joint=joints[nearest_idx].name,
        nearest_limit_margin=joints[nearest_idx].nearest_margin,
        nearest_limit_side=joints[nearest_idx].nearest_side,
        any_joint_near_limit=any_near,
        candidate_drift_m=float(candidate_drift_m),
        ik_error=float(ik_error),
        fk_position_before=(float(pos_before[0]), float(pos_before[1]), float(pos_before[2])),
        fk_position_target=(float(pos_target[0]), float(pos_target[1]), float(pos_target[2])),
        reached_step_m=reached_step_m,
        candidate_step_m=float(candidate_step_m),
        q_step_norm=float(np.linalg.norm(qt - qb)),
        posture_distance_from_initial_q=float(np.linalg.norm(qt - q0)),
        joint4=joint_by_name[j4_name],
        joint5=joint_by_name[j5_name],
        log_reasons=(),
    )


def _collect_base_sector_log_reasons(
    base_sector: BaseSectorDiagnostics | None,
    config: IkQualityLogConfig,
) -> tuple[str, ...]:
    if not config.enable_cartesian_joint1_window_diagnostics or base_sector is None:
        return ()
    reasons: list[str] = []
    if base_sector.joint1_would_violate_warning_window:
        reasons.append(
            "joint1_would_violate_warning_window="
            f"{base_sector.abs_joint1_delta_from_anchor:.4f}>"
            f"{config.cartesian_joint1_window_warning_rad}"
        )
    if base_sector.joint1_would_violate_hard_window:
        reasons.append(
            "joint1_would_violate_hard_window="
            f"{base_sector.abs_joint1_delta_from_anchor:.4f}>"
            f"{config.cartesian_joint1_window_hard_rad}"
        )
    if base_sector.joint1_would_violate_global_cap:
        reasons.append(
            "joint1_would_violate_global_cap="
            f"q={base_sector.joint1_candidate_q:.4f} "
            f"range=[{config.joint1_global_operational_min_rad:.2f},"
            f"{config.joint1_global_operational_max_rad:.2f}] "
            f"error={base_sector.joint1_global_cap_error_rad:.4f}"
        )
    if base_sector.workspace_clamp_with_large_joint1_delta:
        reasons.append(
            "workspace_clamp_with_large_joint1_delta="
            f"axes={','.join(base_sector.workspace_clamped_axes) or '(none)'} "
            f"abs_joint1_delta_from_anchor={base_sector.abs_joint1_delta_from_anchor:.4f}"
        )
    return tuple(reasons)


def _collect_log_reasons(
    diag: IkQualityDiagnostics,
    config: IkQualityLogConfig,
    *,
    base_sector: BaseSectorDiagnostics | None = None,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if diag.nearest_limit_margin < config.joint_limit_warn_margin_rad:
        reasons.append(
            f"nearest_limit_margin={diag.nearest_limit_margin:.4f}<{config.joint_limit_warn_margin_rad}"
        )
    if abs(diag.joint5.q_target) > config.joint5_warn_abs_rad:
        reasons.append(
            f"|joint5|={abs(diag.joint5.q_target):.4f}>{config.joint5_warn_abs_rad}"
        )
    if abs(diag.joint4.q_target) > config.joint4_warn_abs_rad:
        reasons.append(
            f"|joint4|={abs(diag.joint4.q_target):.4f}>{config.joint4_warn_abs_rad}"
        )
    if diag.max_abs_q_delta > config.q_delta_warn_rad:
        reasons.append(
            f"max_abs_q_delta={diag.max_abs_q_delta:.4f}>{config.q_delta_warn_rad}"
        )
    if diag.candidate_drift_m > config.candidate_drift_warn_m:
        reasons.append(
            f"candidate_drift_m={diag.candidate_drift_m:.6f}>{config.candidate_drift_warn_m}"
        )
    trivial_candidate = diag.candidate_step_m > 1e-4
    if trivial_candidate and diag.reached_step_m < config.reached_step_warn_min_m:
        reasons.append(
            f"reached_step_m={diag.reached_step_m:.6f}<{config.reached_step_warn_min_m}"
            f" while candidate_step_m={diag.candidate_step_m:.6f}"
        )
    reasons.extend(_collect_base_sector_log_reasons(base_sector, config))
    return tuple(reasons)


def should_log_ik_quality_diagnostics(
    diag: IkQualityDiagnostics,
    config: IkQualityLogConfig,
    *,
    ik_failure: bool = False,
    base_sector: BaseSectorDiagnostics | None = None,
) -> bool:
    if ik_failure:
        return True
    return bool(_collect_log_reasons(diag, config, base_sector=base_sector))


def with_log_reasons(
    diag: IkQualityDiagnostics,
    config: IkQualityLogConfig,
    *,
    ik_failure: bool = False,
    base_sector: BaseSectorDiagnostics | None = None,
) -> IkQualityDiagnostics:
    reasons = list(_collect_log_reasons(diag, config, base_sector=base_sector))
    if ik_failure:
        reasons.insert(0, "ik_failure")
    return IkQualityDiagnostics(
        joints=diag.joints,
        max_abs_q_delta=diag.max_abs_q_delta,
        max_abs_q_delta_joint=diag.max_abs_q_delta_joint,
        nearest_limit_joint=diag.nearest_limit_joint,
        nearest_limit_margin=diag.nearest_limit_margin,
        nearest_limit_side=diag.nearest_limit_side,
        any_joint_near_limit=diag.any_joint_near_limit,
        candidate_drift_m=diag.candidate_drift_m,
        ik_error=diag.ik_error,
        fk_position_before=diag.fk_position_before,
        fk_position_target=diag.fk_position_target,
        reached_step_m=diag.reached_step_m,
        candidate_step_m=diag.candidate_step_m,
        q_step_norm=diag.q_step_norm,
        posture_distance_from_initial_q=diag.posture_distance_from_initial_q,
        joint4=diag.joint4,
        joint5=diag.joint5,
        log_reasons=tuple(reasons),
    )


def format_base_sector_diagnostics(base_sector: BaseSectorDiagnostics) -> str:
    axes = ",".join(base_sector.workspace_clamped_axes) if base_sector.workspace_clamped_axes else "(none)"
    accepted = (
        f"({base_sector.accepted_fk_position[0]:.4f}, "
        f"{base_sector.accepted_fk_position[1]:.4f}, "
        f"{base_sector.accepted_fk_position[2]:.4f})"
        if base_sector.accepted_fk_position is not None
        else "n/a"
    )
    rejected = (
        f"({base_sector.rejected_candidate_position[0]:.4f}, "
        f"{base_sector.rejected_candidate_position[1]:.4f}, "
        f"{base_sector.rejected_candidate_position[2]:.4f})"
        if base_sector.rejected_candidate_position is not None
        else "n/a"
    )
    outcome_line = (
        f"  outcome: {base_sector.outcome}"
        if not base_sector.rejection_source
        else f"  outcome: rejection_source={base_sector.rejection_source}"
    )
    return "\n".join(
        [
            "Base/sector diagnostics:",
            f"  command_frame={base_sector.command_frame} "
            f"command_frame_kind={base_sector.command_frame_kind} "
            f"workspace_frame={base_sector.workspace_frame}",
            (
                "  cartesian_command_linear="
                f"({base_sector.cartesian_command_linear_x:.4f}, "
                f"{base_sector.cartesian_command_linear_y:.4f}, "
                f"{base_sector.cartesian_command_linear_z:.4f})"
            ),
            outcome_line,
            (
                "  global: nearest_limit="
                f"{base_sector.nearest_limit_joint} "
                f"margin={base_sector.nearest_limit_margin:.4f} "
                f"posture_dist_from_initial_q={base_sector.posture_distance_from_initial_q:.4f}"
            ),
            f"  base_anchor_q={base_sector.base_anchor_q:.4f}",
            f"  joint1_current_q={base_sector.joint1_current_q:.4f}",
            f"  joint1_candidate_q={base_sector.joint1_candidate_q:.4f}",
            f"  q_before_joint1={base_sector.q_before_joint1:.4f}",
            f"  q_committed_or_current_joint1={base_sector.q_committed_or_current_joint1:.4f}",
            f"  joint1_delta_this_tick={base_sector.joint1_delta_this_tick:+.4f}",
            (
                "  joint1_candidate_moved_this_tick="
                f"{base_sector.joint1_candidate_moved_this_tick}"
            ),
            (
                f"  joint1_limits=[{base_sector.joint1_min_limit:.2f},"
                f"{base_sector.joint1_max_limit:.2f}] "
                f"margin_to_lower={base_sector.joint1_margin_to_lower:.4f} "
                f"margin_to_upper={base_sector.joint1_margin_to_upper:.4f}"
            ),
            f"  joint1_delta_from_anchor={base_sector.joint1_delta_from_anchor:+.4f}",
            f"  abs_joint1_delta_from_anchor={base_sector.abs_joint1_delta_from_anchor:.4f}",
            (
                "  joint1_warning_window: "
                f"rad={base_sector.joint1_warning_window_rad:.4f} "
                f"would_violate={base_sector.joint1_would_violate_warning_window} "
                f"error_rad={base_sector.joint1_warning_window_error_rad:.4f}"
            ),
            (
                "  joint1_hard_window: "
                f"rad={base_sector.joint1_hard_window_rad:.4f} "
                f"would_violate={base_sector.joint1_would_violate_hard_window} "
                f"error_rad={base_sector.joint1_hard_window_error_rad:.4f} "
                f"gate_enabled={base_sector.enable_joint1_anchor_hard_gate}"
            ),
            (
                "  joint1_global_cap: "
                f"min={base_sector.joint1_global_operational_min_rad:.2f} "
                f"max={base_sector.joint1_global_operational_max_rad:.2f} "
                f"would_violate={base_sector.joint1_would_violate_global_cap} "
                f"error_rad={base_sector.joint1_global_cap_error_rad:.4f} "
                f"gate_enabled={base_sector.enable_joint1_global_operational_cap}"
            ),
            (
                "  workspace: raw_candidate="
                f"({base_sector.raw_candidate_position[0]:.4f}, "
                f"{base_sector.raw_candidate_position[1]:.4f}, "
                f"{base_sector.raw_candidate_position[2]:.4f}) "
                f"clamped_candidate="
                f"({base_sector.clamped_candidate_position[0]:.4f}, "
                f"{base_sector.clamped_candidate_position[1]:.4f}, "
                f"{base_sector.clamped_candidate_position[2]:.4f}) "
                f"active={base_sector.workspace_clamp_active} axes={axes}"
            ),
            (
                "  workspace_clamp_with_large_joint1_delta="
                f"{base_sector.workspace_clamp_with_large_joint1_delta}"
            ),
            f"  accepted_fk_position={accepted}",
            f"  rejected_candidate_position={rejected}",
        ]
    )


def format_ik_quality_diagnostics(
    diag: IkQualityDiagnostics,
    *,
    base_sector: BaseSectorDiagnostics | None = None,
    local_window: LocalWindowDiagnostics | None = None,
) -> str:
    lines = ["IK quality diagnostics:"]
    if diag.log_reasons:
        lines.append(f"  triggers: {', '.join(diag.log_reasons)}")
    lines.extend(
        [
            f"  global: max_abs_q_delta={diag.max_abs_q_delta:.4f} ({diag.max_abs_q_delta_joint})",
            (
                "  global: nearest_limit="
                f"{diag.nearest_limit_joint} margin={diag.nearest_limit_margin:.4f} "
                f"side={diag.nearest_limit_side} any_near={diag.any_joint_near_limit}"
            ),
            (
                "  motion: candidate_step_m="
                f"{diag.candidate_step_m:.6f} reached_step_m={diag.reached_step_m:.6f} "
                f"candidate_drift_m={diag.candidate_drift_m:.6f} ik_error={diag.ik_error:.6f}"
            ),
            (
                "  fk: before="
                f"({diag.fk_position_before[0]:.4f}, {diag.fk_position_before[1]:.4f}, "
                f"{diag.fk_position_before[2]:.4f}) target="
                f"({diag.fk_position_target[0]:.4f}, {diag.fk_position_target[1]:.4f}, "
                f"{diag.fk_position_target[2]:.4f})"
            ),
            (
                f"  posture: q_step_norm={diag.q_step_norm:.4f} "
                f"dist_from_initial_q={diag.posture_distance_from_initial_q:.4f}"
            ),
            (
                "  highlight joint4 (elbow): "
                f"q {diag.joint4.q_before:.4f}->{diag.joint4.q_target:.4f} "
                f"dq={diag.joint4.q_delta:+.4f} nearest_margin={diag.joint4.nearest_margin:.4f} "
                f"side={diag.joint4.nearest_side}"
            ),
            (
                "  highlight joint5 (wrist): "
                f"q {diag.joint5.q_before:.4f}->{diag.joint5.q_target:.4f} "
                f"dq={diag.joint5.q_delta:+.4f} nearest_margin={diag.joint5.nearest_margin:.4f} "
                f"side={diag.joint5.nearest_side}"
            ),
        ]
    )
    for joint in diag.joints:
        lines.append(
            f"  {joint.name}: q {joint.q_before:.4f}->{joint.q_target:.4f} "
            f"dq={joint.q_delta:+.4f} limits=[{joint.lower_limit:.2f},{joint.upper_limit:.2f}] "
            f"margins=(lo+{joint.margin_to_lower:.4f}, hi-{joint.margin_to_upper:.4f}) "
            f"nearest={joint.nearest_margin:.4f} ({joint.nearest_side})"
        )
    text = "\n".join(lines)
    if local_window is not None:
        text = f"{text}\n{format_local_window_diagnostics(local_window)}"
    if base_sector is not None:
        text = f"{text}\n{format_base_sector_diagnostics(base_sector)}"
    return text


@dataclass(frozen=True)
class LocalWindowDiagnostics:
    teleop_mode: str
    command_frame_kind: str
    base_jog_active: bool
    base_jog_delta_rad: float
    base_jog_before_q: float
    base_jog_after_q: float
    cartesian_mode_skipped_due_to_base_jog: bool
    base_anchor_q: float
    local_window_anchor_position: tuple[float, float, float]
    local_target_offset: tuple[float, float, float]
    local_window_limits: LocalWindowLimitsView
    local_window_clamp_active: bool
    local_window_clamped_axes: tuple[str, ...]
    target_base_link_from_local_window: tuple[float, float, float]
    target_base_link_before_ik: tuple[float, float, float]
    joint1_current_q: float
    joint1_candidate_q: float
    joint1_delta_from_anchor: float
    joint1_would_violate_global_cap: bool
    joint1_global_cap_error_rad: float
    joint1_would_violate_hard_window: bool
    joint1_hard_window_error_rad: float


@dataclass(frozen=True)
class LocalWindowLimitsView:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    global_z_min: float
    global_z_max: float


def format_local_window_diagnostics(local: LocalWindowDiagnostics) -> str:
    lim = local.local_window_limits
    axes = (
        ",".join(local.local_window_clamped_axes)
        if local.local_window_clamped_axes
        else "(none)"
    )
    anchor = local.local_window_anchor_position
    offset = local.local_target_offset
    target = local.target_base_link_from_local_window
    return "\n".join(
        [
            "Local/window diagnostics:",
            f"  teleop_mode={local.teleop_mode}",
            f"  command_frame_kind={local.command_frame_kind}",
            f"  base_jog_active={local.base_jog_active}",
            f"  base_jog_delta_rad={local.base_jog_delta_rad:+.4f}",
            f"  base_jog_before_q={local.base_jog_before_q:.4f}",
            f"  base_jog_after_q={local.base_jog_after_q:.4f}",
            (
                "  cartesian_mode_skipped_due_to_base_jog="
                f"{local.cartesian_mode_skipped_due_to_base_jog}"
            ),
            f"  base_anchor_q={local.base_anchor_q:.4f}",
            (
                "  local_window_anchor_position="
                f"({anchor[0]:.4f}, {anchor[1]:.4f}, {anchor[2]:.4f})"
            ),
            (
                "  local_target_offset="
                f"({offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f})"
            ),
            (
                "  local_window_limits="
                f"x=[{lim.x_min:.2f},{lim.x_max:.2f}] "
                f"y=[{lim.y_min:.2f},{lim.y_max:.2f}] "
                f"z=[{lim.z_min:.2f},{lim.z_max:.2f}] "
                f"global_z=[{lim.global_z_min:.2f},{lim.global_z_max:.2f}]"
            ),
            f"  local_window_clamp_active={local.local_window_clamp_active} axes={axes}",
            (
                "  target_base_link_from_local_window="
                f"({target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f})"
            ),
            (
                "  target_base_link_before_ik="
                f"({local.target_base_link_before_ik[0]:.4f}, "
                f"{local.target_base_link_before_ik[1]:.4f}, "
                f"{local.target_base_link_before_ik[2]:.4f})"
            ),
            f"  joint1_current_q={local.joint1_current_q:.4f}",
            f"  joint1_candidate_q={local.joint1_candidate_q:.4f}",
            f"  joint1_delta_from_anchor={local.joint1_delta_from_anchor:+.4f}",
            (
                "  joint1_global_cap: "
                f"would_violate={local.joint1_would_violate_global_cap} "
                f"error_rad={local.joint1_global_cap_error_rad:.4f}"
            ),
            (
                "  joint1_hard_window: "
                f"would_violate={local.joint1_would_violate_hard_window} "
                f"error_rad={local.joint1_hard_window_error_rad:.4f}"
            ),
        ]
    )


def pos3_from_pose(pose) -> tuple[float, float, float]:
    return (
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
    )


def candidate_step_m(
    fk_position_before: Sequence[float],
    candidate_position: Sequence[float],
) -> float:
    before = np.asarray(fk_position_before, dtype=np.float64).reshape(3)
    candidate = np.asarray(candidate_position, dtype=np.float64).reshape(3)
    return float(np.linalg.norm(candidate - before))
