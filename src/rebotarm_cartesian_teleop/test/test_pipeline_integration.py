"""End-to-end pipeline test using pure mapper/core logic (no ROS graph)."""

from __future__ import annotations

import pytest
from conftest import default_mapper_config, default_workspace, make_joy_with_defaults

from rebotarm_cartesian_teleop.jog_core_logic import (
    build_cartesian_jog_state,
    commit_target_on_ik_success,
    compute_candidate_target,
    compute_state_name,
)
from rebotarm_cartesian_teleop.joy_mapping import map_joy_to_cmd


def _tick_core(
    *,
    target: tuple[float, float, float],
    latest_cmd,
    command_age: float,
    dt: float,
    command_timeout_s: float = 0.3,
    ik_success: bool = True,
):
    ws = default_workspace()
    state = compute_state_name(latest_cmd, command_age, command_timeout_s)
    cx, cy, cz = target
    clamp_reason = ""
    if state == "ACTIVE" and latest_cmd is not None:
        candidate_x, candidate_y, candidate_z, clamp_reason = compute_candidate_target(
            cx, cy, cz, latest_cmd, dt, ws
        )
        cx, cy, cz = commit_target_on_ik_success(
            cx, cy, cz, candidate_x, candidate_y, candidate_z, ik_success
        )
    state_msg = build_cartesian_jog_state(
        state_name=state,
        target_x=cx,
        target_y=cy,
        target_z=cz,
        latest_cmd=latest_cmd,
        clamp_reason=clamp_reason,
        dry_run=True,
        output_mode="dry_run",
        command_age=command_age,
    )
    return state_msg, (cx, cy, cz)


def test_mapper_to_core_deadman_active_soft_stop_timeout_sequence():
    cfg = default_mapper_config()
    target = (0.30, 0.0, 0.20)
    t_joy_ns = 1_000_000_000

    # 1) Deadman up, axes moved -> DEADMAN_UP, target unchanged
    joy_up = make_joy_with_defaults(axis1=1.0, axis0=0.5, deadman=False)
    cmd_up = map_joy_to_cmd(joy_up, cfg, latest_joy_time_ns=t_joy_ns, now_ns=t_joy_ns)
    assert cmd_up is not None
    assert cmd_up.deadman is False
    state_msg, target = _tick_core(target=target, latest_cmd=cmd_up, command_age=0.0, dt=0.02)
    assert state_msg.state == "DEADMAN_UP"
    assert target == pytest.approx((0.30, 0.0, 0.20))

    # 2) Deadman pressed -> ACTIVE, non-zero cmd linear
    joy_active = make_joy_with_defaults(axis1=1.0, axis0=0.5, axis5=-0.5, deadman=True)
    cmd_active = map_joy_to_cmd(joy_active, cfg, latest_joy_time_ns=t_joy_ns, now_ns=t_joy_ns)
    assert cmd_active is not None
    assert cmd_active.linear.x != 0.0
    state_msg, target_before = _tick_core(
        target=target, latest_cmd=cmd_active, command_age=0.0, dt=0.02
    )
    assert state_msg.state == "ACTIVE"

    # 3) Integrate again -> target moves
    state_msg, target_after = _tick_core(
        target=target_before, latest_cmd=cmd_active, command_age=0.0, dt=0.02
    )
    assert state_msg.state == "ACTIVE"
    assert target_after[0] > target_before[0]
    assert target_after[1] > target_before[1]
    assert target_after[2] < target_before[2]

    frozen = target_after

    # 4) Soft stop -> SOFT_STOP, target frozen
    joy_soft = make_joy_with_defaults(axis1=1.0, deadman=True, soft_stop=True)
    cmd_soft = map_joy_to_cmd(joy_soft, cfg, latest_joy_time_ns=t_joy_ns, now_ns=t_joy_ns)
    state_msg, target_soft = _tick_core(
        target=frozen, latest_cmd=cmd_soft, command_age=0.0, dt=0.02
    )
    assert state_msg.state == "SOFT_STOP"
    assert target_soft == pytest.approx(frozen)

    # 5) Stale joy -> mapper stops publishing; core times out
    stale_now = t_joy_ns + int(0.5 * 1e9)
    assert map_joy_to_cmd(joy_active, cfg, latest_joy_time_ns=t_joy_ns, now_ns=stale_now) is None
    state_msg, target_timeout = _tick_core(
        target=frozen, latest_cmd=cmd_soft, command_age=0.5, dt=0.02
    )
    assert state_msg.state == "TIMEOUT"
    assert state_msg.rejection_reason == "COMMAND_TIMEOUT"
    assert target_timeout == pytest.approx(frozen)
