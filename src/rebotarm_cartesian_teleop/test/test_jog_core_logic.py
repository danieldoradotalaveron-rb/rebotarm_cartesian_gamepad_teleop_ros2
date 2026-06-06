"""Unit tests for jog_core_logic pure logic."""

from __future__ import annotations

import math

import pytest
from conftest import default_workspace, make_cmd

from rebotarm_cartesian_teleop.jog_core_logic import (
    build_cartesian_jog_state,
    clamp,
    compute_state_name,
    integrate_target_pose,
    rejection_reason_for_state,
    resolve_rejection_reason,
)


def test_no_command_is_idle():
    assert compute_state_name(None, math.inf, 0.3) == "IDLE"


def test_deadman_false_is_deadman_up():
    cmd = make_cmd(deadman=False)
    assert compute_state_name(cmd, 0.0, 0.3) == "DEADMAN_UP"
    assert rejection_reason_for_state("DEADMAN_UP") == "DEADMAN_UP"


def test_deadman_true_is_active():
    cmd = make_cmd(deadman=True)
    assert compute_state_name(cmd, 0.0, 0.3) == "ACTIVE"
    assert rejection_reason_for_state("ACTIVE") == ""


def test_soft_stop_state():
    cmd = make_cmd(deadman=True, soft_stop=True)
    assert compute_state_name(cmd, 0.0, 0.3) == "SOFT_STOP"
    assert rejection_reason_for_state("SOFT_STOP") == "SOFT_STOP"


def test_command_timeout():
    cmd = make_cmd(deadman=True)
    assert compute_state_name(cmd, 0.31, 0.3) == "TIMEOUT"
    assert rejection_reason_for_state("TIMEOUT") == "COMMAND_TIMEOUT"


def test_target_pose_integration_while_active():
    ws = default_workspace()
    cmd = make_cmd(linear_x=0.1, linear_y=0.2, linear_z=-0.1)
    x, y, z, reason = integrate_target_pose(0.30, 0.0, 0.20, cmd, 0.1, "ACTIVE", ws)
    assert x == pytest.approx(0.31)
    assert y == pytest.approx(0.02)
    assert z == pytest.approx(0.19)
    assert reason == ""


@pytest.mark.parametrize(
    "state_name",
    ["DEADMAN_UP", "SOFT_STOP", "TIMEOUT", "IDLE"],
)
def test_no_integration_when_not_active(state_name):
    ws = default_workspace()
    cmd = make_cmd(linear_x=1.0, linear_y=1.0, linear_z=1.0)
    x, y, z, reason = integrate_target_pose(0.30, 0.0, 0.20, cmd, 0.1, state_name, ws)
    assert x == pytest.approx(0.30)
    assert y == pytest.approx(0.0)
    assert z == pytest.approx(0.20)
    assert reason == ""


def test_workspace_clamp_x_max():
    ws = default_workspace(x_max=0.45)
    cmd = make_cmd(linear_x=10.0)
    x, _, _, reason = integrate_target_pose(0.44, 0.0, 0.20, cmd, 1.0, "ACTIVE", ws)
    assert x == pytest.approx(0.45)
    assert "WORKSPACE_X" in reason


def test_workspace_clamp_y_min():
    ws = default_workspace(y_min=-0.25)
    cmd = make_cmd(linear_y=-10.0)
    _, y, _, reason = integrate_target_pose(0.30, -0.24, 0.20, cmd, 1.0, "ACTIVE", ws)
    assert y == pytest.approx(-0.25)
    assert "WORKSPACE_Y" in reason


def test_workspace_clamp_z():
    ws = default_workspace(z_min=0.05, z_max=0.45)
    cmd = make_cmd(linear_z=-10.0)
    _, _, z, reason = integrate_target_pose(0.30, 0.0, 0.06, cmd, 1.0, "ACTIVE", ws)
    assert z == pytest.approx(0.05)
    assert "WORKSPACE_Z" in reason


def test_workspace_clamp_z_sim_floor():
    """Simulation config uses workspace_z_min=0.02 for floor reach validation."""
    ws = default_workspace(z_min=0.02, z_max=0.45)
    cmd = make_cmd(linear_z=-10.0)
    _, _, z, reason = integrate_target_pose(0.30, 0.0, 0.03, cmd, 1.0, "ACTIVE", ws)
    assert z == pytest.approx(0.02)
    assert reason == "WORKSPACE_Z"


def test_clamp_helper():
    assert clamp(0.5, 0.0, 1.0) == (0.5, False)
    assert clamp(-1.0, 0.0, 1.0) == (0.0, True)
    assert clamp(2.0, 0.0, 1.0) == (1.0, True)


def test_output_fields_on_state_message():
    cmd = make_cmd(deadman=True, linear_x=0.01)
    msg = build_cartesian_jog_state(
        state_name="ACTIVE",
        target_x=0.30,
        target_y=0.0,
        target_z=0.20,
        latest_cmd=cmd,
        clamp_reason="",
        dry_run=True,
        output_mode="dry_run",
        command_age=0.05,
    )
    assert msg.dry_run is True
    assert msg.output_mode == "dry_run"
    assert len(msg.q_current) == 0
    assert len(msg.q_target) == 0
    assert msg.ik_success is False
    assert msg.rejection_reason == resolve_rejection_reason("ACTIVE", "", "")
    assert msg.command_age_s == pytest.approx(0.05)


def test_rejection_fk_error_before_ik_on_active():
    assert resolve_rejection_reason("ACTIVE", "FK_NOT_READY", "IK_FAILED") == "FK_NOT_READY"
