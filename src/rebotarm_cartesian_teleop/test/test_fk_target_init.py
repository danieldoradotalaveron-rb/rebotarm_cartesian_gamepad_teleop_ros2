"""Tests for initial target_pose from FK(q_current)."""

from __future__ import annotations

import pytest
from conftest import call_solve_target_ik

from rebotarm_cartesian_teleop.fk_kinematics import (
    compute_fk_pose,
    init_fk_context,
    initial_target_pose_from_fk,
)
from rebotarm_cartesian_teleop.jog_core_logic import IkConfig, commit_target_on_ik_success
from rebotarm_cartesian_teleop.sdk_path import ensure_rebot_sdk_in_syspath


def _sdk_available() -> bool:
    try:
        ensure_rebot_sdk_in_syspath()
        return True
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(not _sdk_available(), reason="reBotArm_control_py not found")


def test_initial_target_from_fk_matches_current_pose():
    fk_ctx = init_fk_context("", "end_link", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pose, err = compute_fk_pose(fk_ctx)
    assert err == ""
    assert pose is not None

    init = initial_target_pose_from_fk(fk_ctx, 0.30, 0.0, 0.20)
    assert init.from_fk is True
    assert init.fallback_reason == ""
    assert init.x == pytest.approx(pose.position.x)
    assert init.y == pytest.approx(pose.position.y)
    assert init.z == pytest.approx(pose.position.z)


def test_initial_target_yaml_fallback_when_fk_not_ready():
    fk_ctx = init_fk_context("", "end_link", [0.0, 0.0, 0.0])
    assert fk_ctx.ok is False

    init = initial_target_pose_from_fk(fk_ctx, 0.30, 0.0, 0.20)
    assert init.from_fk is False
    assert init.fallback_reason == "INVALID_INITIAL_Q"
    assert init.x == pytest.approx(0.30)
    assert init.y == pytest.approx(0.0)
    assert init.z == pytest.approx(0.20)


def test_aligned_initial_target_ik_succeeds_neutral():
    fk_ctx = init_fk_context("", "end_link", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pose, _ = compute_fk_pose(fk_ctx)
    init = initial_target_pose_from_fk(fk_ctx, 0.30, 0.0, 0.20)

    ik_config = IkConfig(
        max_iterations=100,
        tolerance=1e-4,
        max_ik_error=0.005,
        max_joint_delta_rad=0.25,
    )
    q_target, ik_success, ik_reason, _ = call_solve_target_ik(
        fk_ctx,
        pose,
        fk_ctx.q_current,
        target_x=init.x,
        target_y=init.y,
        target_z=init.z,
        ik_config=ik_config,
    )
    assert ik_success is True
    assert ik_reason == ""
    assert ik_reason != "JOINT_DELTA_TOO_LARGE"
    assert len(q_target) == 6


def test_integration_candidate_computes_delta_without_commit_on_ik_failure():
    from conftest import default_workspace, make_cmd

    from rebotarm_cartesian_teleop.jog_core_logic import compute_candidate_target

    fk_ctx = init_fk_context("", "end_link", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    init = initial_target_pose_from_fk(fk_ctx, 0.30, 0.0, 0.20)
    ws = default_workspace()
    cmd = make_cmd(linear_x=0.1, linear_y=0.2, linear_z=-0.1)

    cx, cy, cz, reason = compute_candidate_target(init.x, init.y, init.z, cmd, 0.1, ws)
    assert cx == pytest.approx(init.x + 0.01)
    assert cy == pytest.approx(init.y + 0.02)
    assert cz == pytest.approx(init.z - 0.01)
    assert reason == ""

    unchanged = commit_target_on_ik_success(init.x, init.y, init.z, cx, cy, cz, ik_success=False)
    assert unchanged == pytest.approx((init.x, init.y, init.z))
