"""Tests for JOINT_NEAR_LIMIT soft-limit rejection policy."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from conftest import TELEOP_INITIAL_Q, call_solve_target_ik, default_workspace, make_cmd

from rebotarm_cartesian_teleop.fake_joint_state import fake_joint_positions_to_publish
from rebotarm_cartesian_teleop.fk_kinematics import init_fk_context
from rebotarm_cartesian_teleop.ik_kinematics import IkSolveResult
from rebotarm_cartesian_teleop.ik_quality_diagnostics import (
    joint_limits_from_model,
    joint_names_from_model,
)
from rebotarm_cartesian_teleop.jog_core_logic import (
    IkConfig,
    JointLimitRejectConfig,
    build_cartesian_jog_state,
    build_committed_target_pose,
    compute_candidate_target,
    format_joint_near_limit_log,
    reject_ik_if_near_joint_limit,
    resync_committed_from_q_sim,
    update_q_sim_on_ik_success,
)

DEFAULT_REJECT = JointLimitRejectConfig(reject_margin_rad=0.05)
WS = default_workspace()
CMD = make_cmd(linear_x=0.1)


@pytest.fixture
def fk_ctx():
    ctx = init_fk_context("", "end_link", TELEOP_INITIAL_Q)
    if not ctx.ok:
        pytest.skip("FK init failed")
    return ctx


@pytest.fixture
def joint_limits(fk_ctx):
    names = joint_names_from_model(fk_ctx.model)
    lo, hi = joint_limits_from_model(fk_ctx.model)
    return names, lo, hi


def _committed_from_q(fk_ctx, q_sim: np.ndarray):
    cx, cy, cz, rot, pose, err = resync_committed_from_q_sim(fk_ctx, q_sim)
    assert not err
    assert pose is not None
    assert rot is not None
    return (cx, cy, cz), rot, pose, q_sim


def test_safe_candidate_accepted(fk_ctx, joint_limits):
    names, lo, hi = joint_limits
    safe_q = [float(v) for v in TELEOP_INITIAL_Q]
    q_target, ik_success, ik_reason, info = reject_ik_if_near_joint_limit(
        safe_q,
        names,
        lo,
        hi,
        DEFAULT_REJECT,
    )
    assert ik_success is True
    assert ik_reason == ""
    assert info is None
    assert q_target == safe_q


@pytest.mark.parametrize(
    ("joint_idx", "at_limit_q", "expected_side"),
    [
        (4, 1.57, "upper"),   # joint5 upper
        (1, -3.14, "lower"),  # joint2 lower
        (5, 3.14, "upper"),   # joint6 upper
    ],
    ids=["joint5_upper", "joint2_lower", "joint6_upper"],
)
def test_at_hard_limit_rejected(fk_ctx, joint_limits, joint_idx, at_limit_q, expected_side):
    names, lo, hi = joint_limits
    candidate_q = [float(v) for v in TELEOP_INITIAL_Q]
    candidate_q[joint_idx] = at_limit_q

    q_target, ik_success, ik_reason, info = reject_ik_if_near_joint_limit(
        candidate_q,
        names,
        lo,
        hi,
        DEFAULT_REJECT,
    )
    assert ik_success is False
    assert ik_reason == "JOINT_NEAR_LIMIT"
    assert q_target == []
    assert info is not None
    assert info.joint == names[joint_idx]
    assert info.nearest_margin == pytest.approx(0.0, abs=1e-9)
    assert info.nearest_side == expected_side
    assert "JOINT_NEAR_LIMIT" in format_joint_near_limit_log(info)

    q_sim = np.asarray(TELEOP_INITIAL_Q, dtype=np.float64)
    q_after = update_q_sim_on_ik_success(q_sim, q_target, ik_success)
    assert np.allclose(q_after, q_sim)
    committed_before = resync_committed_from_q_sim(fk_ctx, q_sim)[:3]
    committed_after = resync_committed_from_q_sim(fk_ctx, q_after)[:3]
    assert committed_after == pytest.approx(committed_before)

    _, sim_rot, current_pose, _ = _committed_from_q(fk_ctx, q_sim)
    msg = build_cartesian_jog_state(
        state_name="ACTIVE",
        target_x=committed_before[0],
        target_y=committed_before[1],
        target_z=committed_before[2],
        latest_cmd=CMD,
        clamp_reason="",
        dry_run=True,
        output_mode="dry_run",
        command_age=0.0,
        current_pose=current_pose,
        target_pose=build_committed_target_pose(*committed_before, sim_rot),
        q_current=[float(v) for v in q_sim],
        q_target=q_target,
        ik_success=ik_success,
        ik_reason=ik_reason,
    )
    assert msg.rejection_reason == "JOINT_NEAR_LIMIT"


def test_margin_between_warn_and_reject_accepted(fk_ctx, joint_limits):
    names, lo, hi = joint_limits
    j5_idx = names.index("joint5")
    upper = hi[j5_idx]
    candidate_q = [float(v) for v in TELEOP_INITIAL_Q]
    candidate_q[j5_idx] = upper - 0.10  # margin 0.10: below warn 0.35, above reject 0.05

    q_target, ik_success, ik_reason, info = reject_ik_if_near_joint_limit(
        candidate_q,
        names,
        lo,
        hi,
        DEFAULT_REJECT,
    )
    assert ik_success is True
    assert ik_reason == ""
    assert info is None
    assert q_target == candidate_q


def test_near_limit_freezes_fake_joint_states(fk_ctx, joint_limits):
    names, lo, hi = joint_limits
    candidate_q = [float(v) for v in TELEOP_INITIAL_Q]
    candidate_q[names.index("joint5")] = hi[names.index("joint5")]

    q_target, ik_success, _, _ = reject_ik_if_near_joint_limit(
        candidate_q,
        names,
        lo,
        hi,
        DEFAULT_REJECT,
    )
    last_valid = [float(v) for v in TELEOP_INITIAL_Q]
    updated, positions = fake_joint_positions_to_publish(
        enabled=True,
        last_valid_fake_q=last_valid,
        q_current=last_valid,
        ik_success=ik_success,
        q_target=q_target,
    )
    assert updated == last_valid
    assert positions == last_valid


def test_joint_delta_too_large_unchanged(fk_ctx, joint_limits):
    q_sim = np.asarray(fk_ctx.q_current, dtype=np.float64).copy()
    _, _, current_pose, _ = _committed_from_q(fk_ctx, q_sim)
    candidate_x, candidate_y, candidate_z, _ = compute_candidate_target(
        float(current_pose.position.x),
        float(current_pose.position.y),
        float(current_pose.position.z),
        CMD,
        0.1,
        WS,
    )
    far_q = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    mock_ik = IkSolveResult(
        success=True,
        q_target=far_q,
        error=0.004,
        iterations=100,
        reason="",
    )
    with patch(
        "rebotarm_cartesian_teleop.jog_core_logic.compute_ik_for_pose",
        return_value=mock_ik,
    ):
        q_target, ik_success, ik_reason, diag = call_solve_target_ik(
            fk_ctx,
            current_pose,
            q_sim,
            target_x=candidate_x,
            target_y=candidate_y,
            target_z=candidate_z,
            ik_config=IkConfig(100, 0.001, 0.005, 1e-6),
        )
    assert ik_success is False
    assert ik_reason == "JOINT_DELTA_TOO_LARGE"
    assert diag is not None
    assert diag.rejection_reason == "JOINT_DELTA_TOO_LARGE"


def test_reject_threshold_config(fk_ctx, joint_limits):
    names, lo, hi = joint_limits
    j5_idx = names.index("joint5")
    upper = hi[j5_idx]
    candidate_q = [float(v) for v in TELEOP_INITIAL_Q]
    candidate_q[j5_idx] = upper - 0.03  # margin 0.03

    strict = JointLimitRejectConfig(reject_margin_rad=0.05)
    loose = JointLimitRejectConfig(reject_margin_rad=0.01)

    _, ok_strict, reason_strict, _ = reject_ik_if_near_joint_limit(
        candidate_q, names, lo, hi, strict
    )
    _, ok_loose, reason_loose, _ = reject_ik_if_near_joint_limit(
        candidate_q, names, lo, hi, loose
    )
    assert ok_strict is False
    assert reason_strict == "JOINT_NEAR_LIMIT"
    assert ok_loose is True
    assert reason_loose == ""
