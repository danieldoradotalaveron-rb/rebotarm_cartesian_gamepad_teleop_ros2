"""ROS parameter declaration and loading for CartesianJogCore."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .ik_quality_diagnostics import IkQualityLogConfig, resolve_joint1_warning_window_rad
from .jog_core_logic import (
    IkConfig,
    IkNoEffectConfig,
    Joint1AnchorWindowConfig,
    Joint1GlobalOperationalLimitConfig,
    JointLimitRejectConfig,
    LocalWindowLimits,
    WorkspaceLimits,
    parse_ik_task_mode,
)

if TYPE_CHECKING:
    from rclpy.node import Node


CARTESIAN_CORE_PARAMETER_DEFAULTS: dict[str, object] = {
    "cartesian_jog_cmd_topic": "/rebotarm/cartesian_jog_cmd",
    "cartesian_jog_state_topic": "/rebotarm/cartesian_jog_state",
    "output_mode": "dry_run",
    "dry_run": True,
    "command_timeout_s": 0.3,
    "servo_hz": 50.0,
    "initial_x": 0.30,
    "initial_y": 0.00,
    "initial_z": 0.20,
    "workspace_x_min": 0.150,
    "workspace_x_max": 0.450,
    "workspace_y_min": -0.250,
    "workspace_y_max": 0.250,
    "workspace_z_min": 0.020,
    "workspace_z_max": 0.450,
    "enable_local_teleop_window": True,
    "enable_base_joint_jog": True,
    "local_window_x_min_m": -0.12,
    "local_window_x_max_m": 0.18,
    "local_window_y_min_m": -0.25,
    "local_window_y_max_m": 0.25,
    "local_window_z_min_m": -0.25,
    "local_window_z_max_m": 0.18,
    "global_z_min_m": 0.020,
    "global_z_max_m": 0.450,
    "base_joint_jog_speed_rad_s": 0.5,
    "urdf_path": "",
    "ee_frame": "end_link",
    "initial_q": [0.0, -0.3, -0.3, 0.0, 0.0, 0.0],
    "ik_max_iterations": 100,
    "ik_tolerance": 0.001,
    "ik_task_mode": "position_only",
    "max_ik_error": 0.005,
    "max_joint_delta_rad": 0.25,
    "ik_failure_log_interval_s": 1.0,
    "candidate_drift_log_threshold_m": 0.001,
    "joint_limit_warn_margin_rad": 0.35,
    "joint_limit_reject_margin_rad": 0.05,
    "joint5_warn_abs_rad": 1.0,
    "joint4_warn_abs_rad": 1.0,
    "q_delta_warn_rad": 0.15,
    "candidate_drift_warn_m": 0.003,
    "reached_step_warn_min_m": 0.0001,
    "ik_quality_log_interval_s": 1.0,
    "enable_cartesian_joint1_window_diagnostics": True,
    "cartesian_joint1_window_warning_rad": 0.25,
    "cartesian_joint1_window_hard_rad": 1.20,
    "cartesian_joint1_window_rad": 0.25,
    "enable_joint1_anchor_hard_gate": True,
    "enable_joint1_global_operational_cap": True,
    "joint1_global_operational_min_rad": -1.60,
    "joint1_global_operational_max_rad": 1.60,
    "joint1_large_delta_from_anchor_rad": 0.15,
    "ik_no_effect_candidate_step_min_m": 0.0005,
    "ik_no_effect_reached_step_min_m": 0.0001,
    "ik_no_effect_q_step_min_norm": 1.0e-6,
    "publish_fake_joint_states": True,
    "fake_joint_states_topic": "/rebotarm/fake_joint_states",
    "fake_joint_state_hz": 50.0,
}


@dataclass(frozen=True)
class CartesianCoreParams:
    cmd_topic: str
    state_topic: str
    output_mode: str
    dry_run: bool
    command_timeout_s: float
    servo_hz: float
    ee_frame: str


@dataclass(frozen=True)
class CartesianOutputParams:
    publish_fake_joint_states: bool
    fake_joint_states_topic: str
    fake_joint_state_hz: float


@dataclass(frozen=True)
class CartesianTeleopGeometryParams:
    initial_x: float
    initial_y: float
    initial_z: float
    urdf_path: str
    initial_q: list[float]
    workspace: WorkspaceLimits
    enable_local_teleop_window: bool
    local_window_limits: LocalWindowLimits


@dataclass(frozen=True)
class CartesianIkParams:
    ik_config: IkConfig
    ik_failure_log_interval_s: float
    candidate_drift_log_threshold_m: float


@dataclass(frozen=True)
class CartesianSafetyParams:
    joint_limit_reject_config: JointLimitRejectConfig
    ik_no_effect_config: IkNoEffectConfig
    joint1_global_cap_config: Joint1GlobalOperationalLimitConfig
    joint1_anchor_window_config: Joint1AnchorWindowConfig


@dataclass(frozen=True)
class CartesianBaseJogParams:
    enable_base_joint_jog: bool
    base_joint_jog_speed_rad_s: float


@dataclass(frozen=True)
class CartesianDiagnosticsParams:
    ik_quality_log_config: IkQualityLogConfig
    ik_quality_log_interval_s: float


@dataclass(frozen=True)
class CartesianCoreParamsBundle:
    core: CartesianCoreParams
    output: CartesianOutputParams
    geometry: CartesianTeleopGeometryParams
    ik: CartesianIkParams
    safety: CartesianSafetyParams
    base_jog: CartesianBaseJogParams
    diagnostics: CartesianDiagnosticsParams


def declare_cartesian_core_parameters(node: Node) -> None:
    for name, default in CARTESIAN_CORE_PARAMETER_DEFAULTS.items():
        node.declare_parameter(name, default)


def load_cartesian_core_params(node: Node) -> CartesianCoreParamsBundle:
    declare_cartesian_core_parameters(node)

    cmd_topic = node.get_parameter("cartesian_jog_cmd_topic").value
    state_topic = node.get_parameter("cartesian_jog_state_topic").value
    output_mode = node.get_parameter("output_mode").value
    dry_run = bool(node.get_parameter("dry_run").value)
    command_timeout_s = float(node.get_parameter("command_timeout_s").value)
    servo_hz = float(node.get_parameter("servo_hz").value)
    ee_frame = str(node.get_parameter("ee_frame").value)

    workspace = WorkspaceLimits(
        x_min=float(node.get_parameter("workspace_x_min").value),
        x_max=float(node.get_parameter("workspace_x_max").value),
        y_min=float(node.get_parameter("workspace_y_min").value),
        y_max=float(node.get_parameter("workspace_y_max").value),
        z_min=float(node.get_parameter("workspace_z_min").value),
        z_max=float(node.get_parameter("workspace_z_max").value),
    )
    enable_local_teleop_window = bool(node.get_parameter("enable_local_teleop_window").value)
    enable_base_joint_jog = bool(node.get_parameter("enable_base_joint_jog").value)
    base_joint_jog_speed_rad_s = float(node.get_parameter("base_joint_jog_speed_rad_s").value)
    local_window_limits = LocalWindowLimits(
        x_min=float(node.get_parameter("local_window_x_min_m").value),
        x_max=float(node.get_parameter("local_window_x_max_m").value),
        y_min=float(node.get_parameter("local_window_y_min_m").value),
        y_max=float(node.get_parameter("local_window_y_max_m").value),
        z_min=float(node.get_parameter("local_window_z_min_m").value),
        z_max=float(node.get_parameter("local_window_z_max_m").value),
        global_z_min=float(node.get_parameter("global_z_min_m").value),
        global_z_max=float(node.get_parameter("global_z_max_m").value),
    )

    urdf_path = str(node.get_parameter("urdf_path").value)
    initial_q = [float(v) for v in node.get_parameter("initial_q").value]
    initial_x = float(node.get_parameter("initial_x").value)
    initial_y = float(node.get_parameter("initial_y").value)
    initial_z = float(node.get_parameter("initial_z").value)

    ik_failure_log_interval_s = float(node.get_parameter("ik_failure_log_interval_s").value)
    candidate_drift_log_threshold_m = float(
        node.get_parameter("candidate_drift_log_threshold_m").value
    )
    ik_quality_log_config = IkQualityLogConfig(
        joint_limit_warn_margin_rad=float(
            node.get_parameter("joint_limit_warn_margin_rad").value
        ),
        joint5_warn_abs_rad=float(node.get_parameter("joint5_warn_abs_rad").value),
        joint4_warn_abs_rad=float(node.get_parameter("joint4_warn_abs_rad").value),
        q_delta_warn_rad=float(node.get_parameter("q_delta_warn_rad").value),
        candidate_drift_warn_m=float(node.get_parameter("candidate_drift_warn_m").value),
        reached_step_warn_min_m=float(node.get_parameter("reached_step_warn_min_m").value),
        enable_cartesian_joint1_window_diagnostics=bool(
            node.get_parameter("enable_cartesian_joint1_window_diagnostics").value
        ),
        cartesian_joint1_window_warning_rad=resolve_joint1_warning_window_rad(
            warning_rad=float(node.get_parameter("cartesian_joint1_window_warning_rad").value),
            legacy_window_rad=float(node.get_parameter("cartesian_joint1_window_rad").value),
        ),
        cartesian_joint1_window_hard_rad=float(
            node.get_parameter("cartesian_joint1_window_hard_rad").value
        ),
        enable_joint1_anchor_hard_gate=bool(
            node.get_parameter("enable_joint1_anchor_hard_gate").value
        ),
        enable_joint1_global_operational_cap=bool(
            node.get_parameter("enable_joint1_global_operational_cap").value
        ),
        joint1_global_operational_min_rad=float(
            node.get_parameter("joint1_global_operational_min_rad").value
        ),
        joint1_global_operational_max_rad=float(
            node.get_parameter("joint1_global_operational_max_rad").value
        ),
        joint1_large_delta_from_anchor_rad=float(
            node.get_parameter("joint1_large_delta_from_anchor_rad").value
        ),
    )
    ik_quality_log_interval_s = float(node.get_parameter("ik_quality_log_interval_s").value)
    ik_no_effect_config = IkNoEffectConfig(
        candidate_step_min_m=float(
            node.get_parameter("ik_no_effect_candidate_step_min_m").value
        ),
        reached_step_min_m=float(node.get_parameter("ik_no_effect_reached_step_min_m").value),
        q_step_min_norm=float(node.get_parameter("ik_no_effect_q_step_min_norm").value),
    )
    joint_limit_reject_config = JointLimitRejectConfig(
        reject_margin_rad=float(node.get_parameter("joint_limit_reject_margin_rad").value),
    )
    joint1_global_cap_config = Joint1GlobalOperationalLimitConfig(
        enabled=ik_quality_log_config.enable_joint1_global_operational_cap,
        min_rad=ik_quality_log_config.joint1_global_operational_min_rad,
        max_rad=ik_quality_log_config.joint1_global_operational_max_rad,
    )
    joint1_anchor_window_config = Joint1AnchorWindowConfig(
        enabled=ik_quality_log_config.enable_joint1_anchor_hard_gate,
        hard_window_rad=ik_quality_log_config.cartesian_joint1_window_hard_rad,
    )

    publish_fake_joint_states = bool(node.get_parameter("publish_fake_joint_states").value)
    fake_joint_states_topic = str(node.get_parameter("fake_joint_states_topic").value)
    fake_joint_state_hz = float(node.get_parameter("fake_joint_state_hz").value)

    ik_task_mode = parse_ik_task_mode(str(node.get_parameter("ik_task_mode").value))
    ik_config = IkConfig(
        max_iterations=int(node.get_parameter("ik_max_iterations").value),
        tolerance=float(node.get_parameter("ik_tolerance").value),
        max_ik_error=float(node.get_parameter("max_ik_error").value),
        max_joint_delta_rad=float(node.get_parameter("max_joint_delta_rad").value),
        task_mode=ik_task_mode,
    )

    return CartesianCoreParamsBundle(
        core=CartesianCoreParams(
            cmd_topic=cmd_topic,
            state_topic=state_topic,
            output_mode=output_mode,
            dry_run=dry_run,
            command_timeout_s=command_timeout_s,
            servo_hz=servo_hz,
            ee_frame=ee_frame,
        ),
        output=CartesianOutputParams(
            publish_fake_joint_states=publish_fake_joint_states,
            fake_joint_states_topic=fake_joint_states_topic,
            fake_joint_state_hz=fake_joint_state_hz,
        ),
        geometry=CartesianTeleopGeometryParams(
            initial_x=initial_x,
            initial_y=initial_y,
            initial_z=initial_z,
            urdf_path=urdf_path,
            initial_q=initial_q,
            workspace=workspace,
            enable_local_teleop_window=enable_local_teleop_window,
            local_window_limits=local_window_limits,
        ),
        ik=CartesianIkParams(
            ik_config=ik_config,
            ik_failure_log_interval_s=ik_failure_log_interval_s,
            candidate_drift_log_threshold_m=candidate_drift_log_threshold_m,
        ),
        safety=CartesianSafetyParams(
            joint_limit_reject_config=joint_limit_reject_config,
            ik_no_effect_config=ik_no_effect_config,
            joint1_global_cap_config=joint1_global_cap_config,
            joint1_anchor_window_config=joint1_anchor_window_config,
        ),
        base_jog=CartesianBaseJogParams(
            enable_base_joint_jog=enable_base_joint_jog,
            base_joint_jog_speed_rad_s=base_joint_jog_speed_rad_s,
        ),
        diagnostics=CartesianDiagnosticsParams(
            ik_quality_log_config=ik_quality_log_config,
            ik_quality_log_interval_s=ik_quality_log_interval_s,
        ),
    )
