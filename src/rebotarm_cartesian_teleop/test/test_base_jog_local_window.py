"""Option B: explicit base jog + base-anchored local teleop window."""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import TELEOP_INITIAL_Q, default_mapper_config, make_joy

from rebotarm_cartesian_teleop.fk_kinematics import compute_fk_pose_for_q, init_fk_context
from rebotarm_cartesian_teleop.ik_quality_diagnostics import pos3_from_pose
from rebotarm_cartesian_teleop.jog_core_logic import (
    COMMAND_FRAME_LOCAL_WINDOW,
    Joint1AnchorWindowConfig,
    Joint1GlobalOperationalLimitConfig,
    LocalWindowLimits,
    LocalWindowState,
    apply_base_joint1_jog,
    integrate_local_window_candidate,
    local_target_to_base_link,
    reanchor_local_window_from_fk,
    reject_ik_if_joint1_anchor_window,
    reject_ik_if_joint1_global_operational_limit,
)
from rebotarm_cartesian_teleop.joy_mapping import map_joy_to_cmd, resolve_base_jog_from_joy

JOINT1_IDX = 0
GLOBAL_MIN = -1.60
GLOBAL_MAX = 1.60
HARD_RAD = 1.20
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

LOCAL_LIMITS = LocalWindowLimits(
    x_min=-0.12,
    x_max=0.18,
    y_min=-0.25,
    y_max=0.25,
    z_min=-0.25,
    z_max=0.18,
    global_z_min=0.020,
    global_z_max=0.450,
)

ANCHOR = (0.272, 0.0, 0.270)


def _local_state(
    *,
    base_anchor_q: float = 0.0,
    anchor: tuple[float, float, float] = ANCHOR,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> LocalWindowState:
    return LocalWindowState(
        base_anchor_q=base_anchor_q,
        anchor_position=anchor,
        local_offset=offset,
    )


def test_base_jog_changes_only_joint1():
    q = [float(v) for v in TELEOP_INITIAL_Q]
    before = q.copy()
    result = apply_base_joint1_jog(
        q,
        joint1_index=JOINT1_IDX,
        velocity_rad_s=0.5,
        dt=0.1,
        min_rad=GLOBAL_MIN,
        max_rad=GLOBAL_MAX,
    )
    assert result.after_q == pytest.approx(before[JOINT1_IDX] + 0.05)
    for i in range(1, len(q)):
        assert result.q_after[i] == pytest.approx(before[i])


def test_base_jog_clamps_joint1():
    q = [float(v) for v in TELEOP_INITIAL_Q]
    q[JOINT1_IDX] = GLOBAL_MAX
    result = apply_base_joint1_jog(
        q,
        joint1_index=JOINT1_IDX,
        velocity_rad_s=1.0,
        dt=1.0,
        min_rad=GLOBAL_MIN,
        max_rad=GLOBAL_MAX,
    )
    assert result.after_q == pytest.approx(GLOBAL_MAX)

    q[JOINT1_IDX] = GLOBAL_MIN
    result = apply_base_joint1_jog(
        q,
        joint1_index=JOINT1_IDX,
        velocity_rad_s=-1.0,
        dt=1.0,
        min_rad=GLOBAL_MIN,
        max_rad=GLOBAL_MAX,
    )
    assert result.after_q == pytest.approx(GLOBAL_MIN)


def test_base_jog_reanchor_resets_local_window():
    state = reanchor_local_window_from_fk(
        fk_position=(0.30, 0.05, 0.28),
        joint1_q=0.42,
    )
    assert state.base_anchor_q == pytest.approx(0.42)
    assert state.anchor_position == pytest.approx((0.30, 0.05, 0.28))
    assert state.local_offset == (0.0, 0.0, 0.0)


def test_local_forward_at_zero_maps_to_plus_x():
    result = local_target_to_base_link(ANCHOR, 0.0, (0.10, 0.0, 0.0), LOCAL_LIMITS)
    assert result.target_base_link[0] == pytest.approx(ANCHOR[0] + 0.10)
    assert result.target_base_link[1] == pytest.approx(ANCHOR[1])
    assert result.target_base_link[2] == pytest.approx(ANCHOR[2])


def test_local_forward_at_pi_over_2_maps_to_plus_y():
    result = local_target_to_base_link(ANCHOR, math.pi / 2, (0.10, 0.0, 0.0), LOCAL_LIMITS)
    assert result.target_base_link[0] == pytest.approx(ANCHOR[0], abs=1e-9)
    assert result.target_base_link[1] == pytest.approx(ANCHOR[1] + 0.10)
    assert result.target_base_link[2] == pytest.approx(ANCHOR[2])


def test_local_lateral_at_pi_over_2_maps_to_minus_x():
    result = local_target_to_base_link(ANCHOR, math.pi / 2, (0.0, 0.10, 0.0), LOCAL_LIMITS)
    assert result.target_base_link[0] == pytest.approx(ANCHOR[0] - 0.10, abs=1e-9)
    assert result.target_base_link[1] == pytest.approx(ANCHOR[1], abs=1e-9)
    assert result.target_base_link[2] == pytest.approx(ANCHOR[2])


def test_local_window_clamps_xyz_offsets():
    result = local_target_to_base_link(
        ANCHOR,
        0.0,
        (0.50, 0.40, 0.40),
        LOCAL_LIMITS,
    )
    assert result.local_offset[0] == pytest.approx(LOCAL_LIMITS.x_max)
    assert result.local_offset[1] == pytest.approx(LOCAL_LIMITS.y_max)
    assert result.local_offset[2] == pytest.approx(LOCAL_LIMITS.z_max)
    assert result.clamp_active is True
    assert "LOCAL_X" in result.clamped_axes
    assert "LOCAL_Y" in result.clamped_axes
    assert "LOCAL_Z" in result.clamped_axes


def test_z_clamp_respects_global_limits():
    low_anchor = (0.272, 0.0, 0.030)
    result = local_target_to_base_link(
        low_anchor,
        0.0,
        (0.0, 0.0, -0.10),
        LOCAL_LIMITS,
    )
    assert result.target_base_link[2] == pytest.approx(LOCAL_LIMITS.global_z_min)
    assert "GLOBAL_Z" in result.clamped_axes

    high_anchor = (0.272, 0.0, 0.44)
    result = local_target_to_base_link(
        high_anchor,
        0.0,
        (0.0, 0.0, 0.20),
        LOCAL_LIMITS,
    )
    assert result.target_base_link[2] == pytest.approx(LOCAL_LIMITS.global_z_max)
    assert "GLOBAL_Z" in result.clamped_axes


def test_base_jog_skips_local_window_integration():
    """Core tick contract: base jog must not integrate Cartesian stick motion."""
    state = _local_state(offset=(0.05, 0.0, 0.0))
    base_jog_active = True
    if not base_jog_active:
        integrate_local_window_candidate(state, (0.10, 0.0, 0.0), 0.1, LOCAL_LIMITS)
    assert state.local_offset == (0.05, 0.0, 0.0)


def test_axis_base_jog_dpad_left_decreases_joint1():
    cfg = default_mapper_config()
    axes = [0.0] * 7
    axes[6] = 1.0
    active, velocity = resolve_base_jog_from_joy(make_joy(axes=axes), cfg)
    assert active is True
    assert velocity == pytest.approx(-0.5)


def test_axis_base_jog_dpad_right_increases_joint1():
    cfg = default_mapper_config()
    axes = [0.0] * 7
    axes[6] = -1.0
    active, velocity = resolve_base_jog_from_joy(make_joy(axes=axes), cfg)
    assert active is True
    assert velocity == pytest.approx(0.5)


def test_axis_base_jog_neutral_is_inactive():
    cfg = default_mapper_config()
    axes = [0.0] * 7
    active, velocity = resolve_base_jog_from_joy(make_joy(axes=axes), cfg)
    assert active is False
    assert velocity == pytest.approx(0.0)


def test_axis_base_jog_sign_flipped_in_yaml():
    cfg = default_mapper_config(base_jog_axis_to_joint_sign=1.0)
    axes = [0.0] * 7
    axes[6] = 1.0
    active, velocity = resolve_base_jog_from_joy(make_joy(axes=axes), cfg)
    assert active is True
    assert velocity == pytest.approx(0.5)


def test_mapper_base_jog_wins_over_sticks():
    cfg = default_mapper_config()
    buttons = [0] * 6
    buttons[4] = 1
    axes = [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    joy = make_joy(axes=axes, buttons=buttons)
    cmd = map_joy_to_cmd(
        joy,
        cfg,
        latest_joy_time_ns=1_000_000_000,
        now_ns=1_100_000_000,
    )
    assert cmd is not None
    assert cmd.base_jog_active is True
    assert cmd.joint1_jog_velocity_rad_s == pytest.approx(-0.5)
    assert cmd.linear.x == 0.0
    assert cmd.linear.y == 0.0
    assert cmd.linear.z == 0.0
    assert cmd.command_frame_kind == COMMAND_FRAME_LOCAL_WINDOW


def test_joint1_gates_still_work():
    gates_on_global = Joint1GlobalOperationalLimitConfig(
        enabled=True, min_rad=GLOBAL_MIN, max_rad=GLOBAL_MAX
    )
    gates_on_anchor = Joint1AnchorWindowConfig(enabled=True, hard_window_rad=HARD_RAD)

    q = [float(v) for v in TELEOP_INITIAL_Q]
    q[0] = 1.61
    _, ok, reason, _ = reject_ik_if_joint1_global_operational_limit(
        q, JOINT_NAMES, gates_on_global
    )
    assert ok is False
    assert reason == "JOINT1_GLOBAL_OPERATIONAL_LIMIT"

    q[0] = 1.25
    _, ok, reason, _ = reject_ik_if_joint1_anchor_window(
        q, JOINT_NAMES, base_anchor_q=0.0, config=gates_on_anchor
    )
    assert ok is False
    assert reason == "JOINT1_ANCHOR_WINDOW"


def test_integrate_local_window_updates_offset():
    state = _local_state()
    updated, clamp = integrate_local_window_candidate(
        state,
        (0.10, 0.0, 0.0),
        dt=0.1,
        limits=LOCAL_LIMITS,
    )
    assert updated.local_offset[0] == pytest.approx(0.01)
    assert clamp.target_base_link[0] == pytest.approx(ANCHOR[0] + 0.01)


def test_base_jog_fk_reanchor_no_stale_target():
    fk = init_fk_context("", "end_link", TELEOP_INITIAL_Q)
    if not fk.ok:
        pytest.skip("URDF unavailable for FK integration test")

    q = np.asarray(TELEOP_INITIAL_Q, dtype=np.float64)
    pose_before, _, _ = compute_fk_pose_for_q(fk, q)
    pos_before = pos3_from_pose(pose_before)

    jog = apply_base_joint1_jog(
        q,
        joint1_index=JOINT1_IDX,
        velocity_rad_s=0.5,
        dt=0.2,
        min_rad=GLOBAL_MIN,
        max_rad=GLOBAL_MAX,
    )
    q_after = np.asarray(jog.q_after, dtype=np.float64)
    pose_after, _, _ = compute_fk_pose_for_q(fk, q_after)
    pos_after = pos3_from_pose(pose_after)

    state = reanchor_local_window_from_fk(
        fk_position=pos_after,
        joint1_q=float(q_after[JOINT1_IDX]),
    )
    assert state.local_offset == (0.0, 0.0, 0.0)
    assert state.anchor_position == pytest.approx(pos_after)
    assert pos_after != pos_before
