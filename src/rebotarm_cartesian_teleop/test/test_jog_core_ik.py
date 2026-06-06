"""Unit tests for IK integration logic in jog_core_logic."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import call_solve_target_ik, default_workspace, make_cmd

from rebotarm_cartesian_teleop.fk_kinematics import compute_fk_pose, init_fk_context
from rebotarm_cartesian_teleop.jog_core_logic import (
    IkConfig,
    build_cartesian_jog_state,
    integrate_target_pose,
    resolve_rejection_reason,
)
from rebotarm_cartesian_teleop.sdk_path import ensure_rebot_sdk_in_syspath


def _sdk_available() -> bool:
    try:
        ensure_rebot_sdk_in_syspath()
        return True
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(not _sdk_available(), reason="reBotArm_control_py not found")


@pytest.fixture
def fk_ctx():
    return init_fk_context("", "end_link", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


@pytest.fixture
def ik_config():
    return IkConfig(
        max_iterations=100,
        tolerance=1e-4,
        max_ik_error=0.005,
        max_joint_delta_rad=0.25,
    )


def test_solve_target_ik_at_fk_position_succeeds(fk_ctx, ik_config):
    pose, err = compute_fk_pose(fk_ctx)
    assert err == ""
    assert pose is not None

    q_target, ik_success, ik_reason, _ = call_solve_target_ik(
        fk_ctx,
        pose,
        fk_ctx.q_current,
        target_x=pose.position.x,
        target_y=pose.position.y,
        target_z=pose.position.z,
        ik_config=ik_config,
    )
    assert ik_success is True
    assert ik_reason == ""
    assert len(q_target) == 6


def test_solve_target_ik_not_run_when_deadman_up(fk_ctx, ik_config):
    pose, _ = compute_fk_pose(fk_ctx)
    q_target, ik_success, ik_reason, diag = call_solve_target_ik(
        fk_ctx,
        pose,
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        state_name="DEADMAN_UP",
        target_x=0.3,
        target_y=0.0,
        target_z=0.2,
        ik_config=ik_config,
    )
    assert ik_success is False
    assert q_target == []
    assert ik_reason == ""
    assert diag is None


def test_joint_delta_rejects_large_step(fk_ctx, ik_config):
    pose, _ = compute_fk_pose(fk_ctx)
    strict = IkConfig(
        max_iterations=ik_config.max_iterations,
        tolerance=ik_config.tolerance,
        max_ik_error=ik_config.max_ik_error,
        max_joint_delta_rad=1e-6,
    )
    q_target, ik_success, ik_reason, _ = call_solve_target_ik(
        fk_ctx,
        pose,
        fk_ctx.q_current,
        target_x=0.45,
        target_y=0.25,
        target_z=0.40,
        ik_config=strict,
    )
    assert ik_success is False
    assert q_target == []
    assert ik_reason == "JOINT_DELTA_TOO_LARGE"


def test_rejection_priority_deadman_over_ik():
    assert resolve_rejection_reason("DEADMAN_UP", "", "IK_FAILED") == "DEADMAN_UP"


def test_rejection_priority_timeout_over_ik():
    assert resolve_rejection_reason("TIMEOUT", "", "IK_FAILED") == "COMMAND_TIMEOUT"


def test_rejection_priority_soft_stop_over_ik():
    assert resolve_rejection_reason("SOFT_STOP", "", "IK_FAILED") == "SOFT_STOP"


def test_rejection_active_uses_ik_reason():
    assert resolve_rejection_reason("ACTIVE", "", "IK_FAILED") == "IK_FAILED"


def test_build_state_active_ik_success_fields(fk_ctx, ik_config):
    pose, _ = compute_fk_pose(fk_ctx)
    q_target, ik_success, ik_reason, _ = call_solve_target_ik(
        fk_ctx,
        pose,
        fk_ctx.q_current,
        target_x=pose.position.x,
        target_y=pose.position.y,
        target_z=pose.position.z,
        ik_config=ik_config,
    )
    msg = build_cartesian_jog_state(
        state_name="ACTIVE",
        target_x=pose.position.x,
        target_y=pose.position.y,
        target_z=pose.position.z,
        latest_cmd=make_cmd(deadman=True),
        clamp_reason="",
        dry_run=True,
        output_mode="dry_run",
        command_age=0.0,
        current_pose=pose,
        q_current=[float(v) for v in fk_ctx.q_current],
        q_target=q_target,
        ik_success=ik_success,
        ik_reason=ik_reason,
    )
    assert msg.ik_success is True
    assert len(msg.q_target) == 6
    assert msg.rejection_reason == ""


def test_q_current_unchanged_after_integration(fk_ctx):
    q_before = fk_ctx.q_current.copy()
    integrate_target_pose(
        0.30, 0.0, 0.20, make_cmd(linear_x=1.0), 0.1, "ACTIVE", default_workspace()
    )
    assert np.allclose(fk_ctx.q_current, q_before)
