"""Tests for commit-on-success target_pose behavior."""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest
from conftest import call_solve_target_ik, default_workspace, make_cmd

from rebotarm_cartesian_teleop.fake_joint_state import fake_joint_positions_to_publish
from rebotarm_cartesian_teleop.fk_kinematics import (
    compute_fk_pose,
    compute_fk_pose_for_q,
    init_fk_context,
)
from rebotarm_cartesian_teleop.ik_kinematics import IkSolveResult
from rebotarm_cartesian_teleop.jog_core_logic import (
    IkConfig,
    build_cartesian_jog_state,
    commit_target_on_ik_success,
    compute_candidate_target,
    compute_state_name,
    integrate_target_pose,
    resync_committed_from_q_sim,
    solve_target_ik,
    update_q_sim_on_ik_success,
)
from rebotarm_cartesian_teleop.sdk_path import ensure_rebot_sdk_in_syspath


def _sdk_available() -> bool:
    try:
        ensure_rebot_sdk_in_syspath()
        return True
    except FileNotFoundError:
        return False


COMMITTED = (0.26, 0.0, 0.192)
CMD = make_cmd(linear_x=0.1, linear_y=0.0, linear_z=0.0)
WS = default_workspace()


def _active_tick(
    committed: tuple[float, float, float],
    *,
    ik_success: bool,
    ik_reason: str = "",
    q_target: list[float] | None = None,
):
    cx, cy, cz = committed
    candidate_x, candidate_y, candidate_z, clamp_reason = compute_candidate_target(
        cx, cy, cz, CMD, 0.1, WS
    )
    new_cx, new_cy, new_cz = commit_target_on_ik_success(
        cx, cy, cz, candidate_x, candidate_y, candidate_z, ik_success
    )
    msg = build_cartesian_jog_state(
        state_name="ACTIVE",
        target_x=new_cx,
        target_y=new_cy,
        target_z=new_cz,
        latest_cmd=CMD,
        clamp_reason=clamp_reason,
        dry_run=True,
        output_mode="dry_run",
        command_age=0.0,
        q_target=q_target if q_target is not None else ([] if not ik_success else [0.0] * 6),
        ik_success=ik_success,
        ik_reason=ik_reason,
    )
    return (new_cx, new_cy, new_cz), (candidate_x, candidate_y, candidate_z), msg


def test_valid_ik_commits_candidate_target():
    committed, candidate, msg = _active_tick(COMMITTED, ik_success=True, q_target=[0.01] * 6)
    expected_x = COMMITTED[0] + 0.01
    assert committed[0] == pytest.approx(expected_x)
    assert committed[1] == pytest.approx(COMMITTED[1])
    assert committed[2] == pytest.approx(COMMITTED[2])
    assert candidate[0] == pytest.approx(expected_x)
    assert float(msg.target_pose.position.x) == pytest.approx(expected_x)
    assert msg.ik_success is True


@pytest.mark.parametrize("ik_reason", ["IK_ERROR_TOO_HIGH", "JOINT_DELTA_TOO_LARGE"])
def test_ik_rejection_does_not_change_committed_target(ik_reason):
    committed, candidate, msg = _active_tick(
        COMMITTED, ik_success=False, ik_reason=ik_reason, q_target=[]
    )
    assert committed == pytest.approx(COMMITTED)
    assert candidate != pytest.approx(COMMITTED)
    assert msg.target_pose.position.x == pytest.approx(COMMITTED[0])
    assert len(msg.q_target) == 0
    assert msg.ik_success is False
    assert msg.rejection_reason == ik_reason


def test_state_message_publishes_committed_not_rejected_candidate():
    _, candidate, msg = _active_tick(COMMITTED, ik_success=False, ik_reason="IK_ERROR_TOO_HIGH")
    assert msg.target_pose.position.x == pytest.approx(COMMITTED[0])
    assert msg.target_pose.position.x != pytest.approx(candidate)


def test_fake_joint_states_freeze_on_rejected_candidate():
    last_valid = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    updated, positions = fake_joint_positions_to_publish(
        enabled=True,
        last_valid_fake_q=last_valid,
        q_current=[0.0] * 6,
        ik_success=False,
        q_target=[],
    )
    assert updated == last_valid
    assert positions == last_valid


@pytest.mark.parametrize(
    ("state_name", "cmd", "command_age"),
    [
        ("DEADMAN_UP", make_cmd(linear_x=1.0, deadman=False), 0.0),
        ("SOFT_STOP", make_cmd(linear_x=1.0, deadman=True, soft_stop=True), 0.0),
        ("TIMEOUT", make_cmd(linear_x=1.0, deadman=True), 0.31),
        ("IDLE", None, math.inf),
    ],
)
def test_non_active_states_do_not_integrate_committed_target(state_name, cmd, command_age):
    x, y, z, reason = integrate_target_pose(*COMMITTED, cmd, 0.1, state_name, WS)
    assert (x, y, z) == pytest.approx(COMMITTED)
    assert reason == ""
    assert compute_state_name(cmd, command_age, 0.3) == state_name


@pytest.mark.skipif(not _sdk_available(), reason="reBotArm_control_py not found")
def test_active_valid_ik_commits_with_real_solver():
    fk_ctx = init_fk_context("", "end_link", [0.0, -0.3, -0.3, 0.0, 0.0, 0.0])
    q_sim = np.asarray(fk_ctx.q_current, dtype=np.float64).copy()
    pose, sim_rot, _ = compute_fk_pose_for_q(fk_ctx, q_sim)
    committed = (float(pose.position.x), float(pose.position.y), float(pose.position.z))
    ik_config = IkConfig(
        max_iterations=100,
        tolerance=1e-4,
        max_ik_error=0.005,
        max_joint_delta_rad=0.25,
    )
    cmd = make_cmd(linear_x=0.01)
    candidate_x, candidate_y, candidate_z, _ = compute_candidate_target(*committed, cmd, 0.1, WS)
    q_target, ik_success, _, _ = solve_target_ik(
        fk_ctx=fk_ctx,
        state_name="ACTIVE",
        target_x=candidate_x,
        target_y=candidate_y,
        target_z=candidate_z,
        target_rotation=sim_rot,
        q_seed=q_sim,
        ik_config=ik_config,
        committed_x=committed[0],
        committed_y=committed[1],
        committed_z=committed[2],
    )
    assert ik_success is True
    assert len(q_target) == 6
    q_sim = update_q_sim_on_ik_success(q_sim, q_target, True)
    new_committed = resync_committed_from_q_sim(fk_ctx, q_sim)[:3]
    assert new_committed[0] >= committed[0]


@pytest.mark.skipif(not _sdk_available(), reason="reBotArm_control_py not found")
def test_active_joint_delta_rejection_keeps_committed():
    fk_ctx = init_fk_context("", "end_link", [0.0, -0.3, -0.3, 0.0, 0.0, 0.0])
    q_sim = np.asarray(fk_ctx.q_current, dtype=np.float64).copy()
    pose, sim_rot, _ = compute_fk_pose_for_q(fk_ctx, q_sim)
    committed = (float(pose.position.x), float(pose.position.y), float(pose.position.z))
    ik_config = IkConfig(
        max_iterations=100,
        tolerance=1e-4,
        max_ik_error=0.005,
        max_joint_delta_rad=0.25,
    )
    candidate_x, candidate_y, candidate_z, _ = compute_candidate_target(
        *committed, make_cmd(linear_x=0.5), 0.1, WS
    )
    q_target, ik_success, ik_reason, _ = solve_target_ik(
        fk_ctx=fk_ctx,
        state_name="ACTIVE",
        target_x=candidate_x,
        target_y=candidate_y,
        target_z=candidate_z,
        target_rotation=sim_rot,
        q_seed=q_sim,
        ik_config=ik_config,
        committed_x=committed[0],
        committed_y=committed[1],
        committed_z=committed[2],
    )
    assert ik_success is False
    assert q_target == []
    assert ik_reason == "JOINT_DELTA_TOO_LARGE"
    unchanged = resync_committed_from_q_sim(fk_ctx, q_sim)[:3]
    assert unchanged == pytest.approx(committed)


def test_active_ik_error_too_high_keeps_committed_with_mock():
    fk_ctx_ok = True
    if not _sdk_available():
        fk_ctx_ok = False

    if fk_ctx_ok:
        fk_ctx = init_fk_context("", "end_link", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        pose, _ = compute_fk_pose(fk_ctx)
    else:
        fk_ctx = None
        pose = None

    committed = COMMITTED
    candidate_x, candidate_y, candidate_z, _ = compute_candidate_target(*committed, CMD, 0.1, WS)
    mock_ik = IkSolveResult(
        success=False,
        q_target=[],
        error=0.006,
        iterations=100,
        reason="IK_ERROR_TOO_HIGH",
    )

    if fk_ctx_ok:
        with patch(
            "rebotarm_cartesian_teleop.jog_core_logic.compute_ik_for_pose",
            return_value=mock_ik,
        ):
            q_target, ik_success, ik_reason, _ = call_solve_target_ik(
                fk_ctx,
                pose,
                fk_ctx.q_current,
                target_x=candidate_x,
                target_y=candidate_y,
                target_z=candidate_z,
                ik_config=IkConfig(100, 0.001, 0.005, 0.25),
                committed_x=committed[0],
                committed_y=committed[1],
                committed_z=committed[2],
            )
    else:
        pytest.skip("SDK unavailable")

    new_committed = commit_target_on_ik_success(
        *committed, candidate_x, candidate_y, candidate_z, ik_success
    )
    assert ik_success is False
    assert q_target == []
    assert ik_reason == "IK_ERROR_TOO_HIGH"
    assert new_committed == pytest.approx(committed)
