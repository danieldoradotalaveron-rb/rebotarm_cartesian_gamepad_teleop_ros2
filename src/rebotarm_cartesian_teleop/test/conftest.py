"""Shared fixtures for Cartesian teleop unit tests."""

from __future__ import annotations

import numpy as np
from rebotarm_msgs.msg import CartesianJogCmd
from sensor_msgs.msg import Joy

from rebotarm_cartesian_teleop.fk_pose import pose_to_rotation_matrix
from rebotarm_cartesian_teleop.jog_core_logic import IkConfig, WorkspaceLimits, solve_target_ik
from rebotarm_cartesian_teleop.joy_mapping import JoyMapperConfig

# Default teleop simulation posture (cartesian_teleop.yaml initial_q).
TELEOP_INITIAL_Q = [0.0, -0.3, -0.3, 0.0, 0.0, 0.0]


def default_mapper_config(**overrides) -> JoyMapperConfig:
    params = {
        "deadzone": 0.15,
        "joy_timeout_s": 0.3,
        "max_linear_velocity": 0.03,
        "speed_scale_default": 1.0,
        "speed_scale_boost": 1.5,
    }
    params.update(overrides)
    return JoyMapperConfig(**params)


def default_workspace(**overrides) -> WorkspaceLimits:
    params = {
        "x_min": 0.15,
        "x_max": 0.45,
        "y_min": -0.25,
        "y_max": 0.25,
        "z_min": 0.02,
        "z_max": 0.45,
    }
    params.update(overrides)
    return WorkspaceLimits(**params)


def make_joy(
    axes: list[float] | None = None,
    buttons: list[int] | None = None,
) -> Joy:
    msg = Joy()
    msg.axes = [float(v) for v in (axes or [])]
    msg.buttons = [int(v) for v in (buttons or [])]
    return msg


def make_joy_with_defaults(
    *,
    axis0: float = 0.0,
    axis1: float = 0.0,
    axis5: float = 0.0,
    deadman: bool = False,
    soft_stop: bool = False,
    speed_boost: bool = False,
) -> Joy:
    axes = [0.0] * 6
    axes[0] = axis0
    axes[1] = axis1
    axes[5] = axis5
    buttons = [0] * 6
    buttons[4] = 1 if deadman else 0
    buttons[2] = 1 if soft_stop else 0
    buttons[5] = 1 if speed_boost else 0
    return make_joy(axes=axes, buttons=buttons)


def make_cmd(
    *,
    deadman: bool = True,
    soft_stop: bool = False,
    linear_x: float = 0.0,
    linear_y: float = 0.0,
    linear_z: float = 0.0,
    base_jog_active: bool = False,
    joint1_jog_velocity_rad_s: float = 0.0,
    command_frame_kind: str = "local_window_frame",
) -> CartesianJogCmd:
    msg = CartesianJogCmd()
    msg.deadman = deadman
    msg.soft_stop = soft_stop
    msg.linear.x = linear_x
    msg.linear.y = linear_y
    msg.linear.z = linear_z
    msg.base_jog_active = base_jog_active
    msg.joint1_jog_velocity_rad_s = float(joint1_jog_velocity_rad_s)
    msg.command_frame_kind = command_frame_kind
    return msg


def call_solve_target_ik(
    fk_ctx,
    pose,
    q_seed,
    *,
    state_name: str = "ACTIVE",
    target_x: float,
    target_y: float,
    target_z: float,
    ik_config: IkConfig,
    **kwargs,
):
    return solve_target_ik(
        fk_ctx=fk_ctx,
        state_name=state_name,
        target_x=target_x,
        target_y=target_y,
        target_z=target_z,
        target_rotation=pose_to_rotation_matrix(pose),
        q_seed=np.asarray(q_seed, dtype=np.float64),
        ik_config=ik_config,
        **kwargs,
    )
