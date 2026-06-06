"""Tests for IK failure diagnostics (logging-safe, no behavior change)."""

from __future__ import annotations

import pytest
from conftest import call_solve_target_ik

from rebotarm_cartesian_teleop.fk_kinematics import compute_fk_pose, init_fk_context
from rebotarm_cartesian_teleop.jog_core_logic import (
    IkConfig,
    IkFailureDiagnostics,
    format_ik_failure_log,
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


def test_format_ik_failure_log_contains_required_fields():
    diag = IkFailureDiagnostics(
        rejection_reason="IK_FAILED",
        candidate_target=(0.261, 0.001, 0.192),
        seed_q=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ik_error=0.012345,
        ik_iterations=100,
        max_ik_error=0.005,
        max_joint_delta_rad=0.25,
        clamp_reason="WORKSPACE_X",
        state="ACTIVE",
        target_rotation_from_fk=True,
        committed_target=(0.260, 0.0, 0.192),
    )
    line = format_ik_failure_log(diag)
    assert "reason=IK_FAILED" in line
    assert "state=ACTIVE" in line
    assert "committed=(0.2600, 0.0000, 0.1920)" in line
    assert "candidate=(0.2610, 0.0010, 0.1920)" in line
    assert "seed_q=[" in line
    assert "ik_error=0.012345" in line
    assert "ik_iterations=100" in line
    assert "max_ik_error=0.005000" in line
    assert "max_joint_delta_rad=0.2500" in line
    assert "clamp_reason=WORKSPACE_X" in line
    assert "target_rotation=FK(q_sim)" in line


def test_format_ik_failure_log_handles_missing_solver_metrics():
    diag = IkFailureDiagnostics(
        rejection_reason="IK_EXCEPTION",
        candidate_target=(0.3, 0.0, 0.2),
        seed_q=(0.0,) * 6,
        ik_error=None,
        ik_iterations=None,
        max_ik_error=0.005,
        max_joint_delta_rad=0.25,
        clamp_reason="",
        state="ACTIVE",
        target_rotation_from_fk=True,
    )
    line = format_ik_failure_log(diag)
    assert "ik_error=n/a" in line
    assert "ik_iterations=n/a" in line
    assert "clamp_reason=(none)" in line


def test_solve_target_ik_returns_diagnostics_on_joint_delta_rejection(fk_ctx, ik_config):
    pose, _ = compute_fk_pose(fk_ctx)
    strict = IkConfig(
        max_iterations=ik_config.max_iterations,
        tolerance=ik_config.tolerance,
        max_ik_error=ik_config.max_ik_error,
        max_joint_delta_rad=1e-6,
    )
    q_target, ik_success, ik_reason, diag = call_solve_target_ik(
        fk_ctx,
        pose,
        fk_ctx.q_current,
        target_x=0.45,
        target_y=0.25,
        target_z=0.40,
        ik_config=strict,
        clamp_reason="WORKSPACE_Z",
        committed_x=0.26,
        committed_y=0.0,
        committed_z=0.192,
    )
    assert ik_success is False
    assert q_target == []
    assert ik_reason == "JOINT_DELTA_TOO_LARGE"
    assert diag is not None
    assert diag.rejection_reason == "JOINT_DELTA_TOO_LARGE"
    assert diag.clamp_reason == "WORKSPACE_Z"
    assert diag.target_rotation_from_fk is True
    assert diag.ik_error is not None
    assert diag.ik_iterations is not None


def test_solve_target_ik_no_diagnostics_when_not_active(fk_ctx, ik_config):
    pose, _ = compute_fk_pose(fk_ctx)
    _, ik_success, _, diag = call_solve_target_ik(
        fk_ctx,
        pose,
        fk_ctx.q_current,
        state_name="DEADMAN_UP",
        target_x=0.3,
        target_y=0.0,
        target_z=0.2,
        ik_config=ik_config,
    )
    assert ik_success is False
    assert diag is None


def test_solve_target_ik_success_has_no_diagnostics(fk_ctx, ik_config):
    pose, _ = compute_fk_pose(fk_ctx)
    _, ik_success, ik_reason, diag = call_solve_target_ik(
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
    assert diag is None
