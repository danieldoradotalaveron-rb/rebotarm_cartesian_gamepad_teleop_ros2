"""Tests for IK task mode (full_6d vs position_only)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from conftest import TELEOP_INITIAL_Q, call_solve_target_ik

from rebotarm_cartesian_teleop.fk_kinematics import compute_fk_pose_for_q, init_fk_context
from rebotarm_cartesian_teleop.ik_kinematics import (
    IkSolveResult,
    compute_ik_for_position,
    orientation_drift_rad,
)
from rebotarm_cartesian_teleop.jog_core_logic import (
    IkConfig,
    IkNoEffectConfig,
    JointLimitRejectConfig,
    parse_ik_task_mode,
    reject_ik_if_near_joint_limit,
    reject_ik_if_no_effect,
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
    ctx = init_fk_context("", "end_link", TELEOP_INITIAL_Q)
    if not ctx.ok:
        pytest.skip("FK init failed")
    return ctx


def _ik_config(**overrides) -> IkConfig:
    params = {
        "max_iterations": 100,
        "tolerance": 1e-4,
        "max_ik_error": 0.005,
        "max_joint_delta_rad": 0.25,
        "task_mode": "full_6d",
    }
    params.update(overrides)
    return IkConfig(**params)


def _y_jog_ticks(fk_ctx, ik_config: IkConfig, *, ticks: int, step_m: float):
    q = np.asarray(TELEOP_INITIAL_Q, dtype=np.float64).copy()
    j1_start = float(q[0])
    j1_total = 0.0
    max_first_step = 0.0

    for tick in range(ticks):
        pose, rot, err = compute_fk_pose_for_q(fk_ctx, q)
        assert not err and pose is not None and rot is not None
        cand_x = float(pose.position.x)
        cand_y = float(pose.position.y) + step_m
        cand_z = float(pose.position.z)
        q_target, ok, reason, _ = call_solve_target_ik(
            fk_ctx,
            pose,
            q,
            target_x=cand_x,
            target_y=cand_y,
            target_z=cand_z,
            ik_config=ik_config,
        )
        assert ok, f"tick {tick}: {reason}"
        dq = np.asarray(q_target, dtype=np.float64) - q
        if tick == 0:
            max_first_step = float(np.max(np.abs(dq)))
        j1_total = float(q_target[0]) - j1_start
        q = np.asarray(q_target, dtype=np.float64)

    return j1_total, max_first_step


def test_parse_ik_task_mode_defaults_to_full_6d():
    assert parse_ik_task_mode("") == "full_6d"
    assert parse_ik_task_mode("full_6d") == "full_6d"
    assert parse_ik_task_mode("POSITION_ONLY") == "position_only"


def test_invalid_ik_task_mode_rejected():
    with pytest.raises(ValueError, match="Invalid ik_task_mode"):
        parse_ik_task_mode("weighted")


def test_full_6d_mode_unchanged(fk_ctx):
    pose, rot, _ = compute_fk_pose_for_q(fk_ctx, np.asarray(TELEOP_INITIAL_Q, dtype=np.float64))
    mock_pose = IkSolveResult(
        success=True,
        q_target=list(TELEOP_INITIAL_Q),
        error=0.001,
        iterations=5,
        reason="",
    )
    with (
        patch(
            "rebotarm_cartesian_teleop.jog_core_logic.compute_ik_for_pose",
            return_value=mock_pose,
        ) as mock_pose_fn,
        patch("rebotarm_cartesian_teleop.jog_core_logic.compute_ik_for_position") as mock_pos_fn,
    ):
        q_target, ok, reason, _ = call_solve_target_ik(
            fk_ctx,
            pose,
            TELEOP_INITIAL_Q,
            target_x=float(pose.position.x),
            target_y=float(pose.position.y),
            target_z=float(pose.position.z),
            ik_config=_ik_config(task_mode="full_6d"),
        )
    assert ok is True
    assert reason == ""
    assert q_target == pytest.approx(TELEOP_INITIAL_Q)
    mock_pose_fn.assert_called_once()
    mock_pos_fn.assert_not_called()


def test_position_only_dispatches_to_position_solver(fk_ctx):
    pose, _, _ = compute_fk_pose_for_q(fk_ctx, np.asarray(TELEOP_INITIAL_Q, dtype=np.float64))
    mock_pos = IkSolveResult(
        success=True,
        q_target=list(TELEOP_INITIAL_Q),
        error=0.001,
        iterations=5,
        reason="",
    )
    with (
        patch(
            "rebotarm_cartesian_teleop.jog_core_logic.compute_ik_for_position",
            return_value=mock_pos,
        ) as mock_pos_fn,
        patch("rebotarm_cartesian_teleop.jog_core_logic.compute_ik_for_pose") as mock_pose_fn,
    ):
        call_solve_target_ik(
            fk_ctx,
            pose,
            TELEOP_INITIAL_Q,
            target_x=float(pose.position.x) + 0.01,
            target_y=float(pose.position.y),
            target_z=float(pose.position.z),
            ik_config=_ik_config(task_mode="position_only"),
        )
    mock_pos_fn.assert_called_once()
    mock_pose_fn.assert_not_called()


def test_position_only_ik_reaches_xyz(fk_ctx):
    q_seed = np.asarray(TELEOP_INITIAL_Q, dtype=np.float64)
    pose, rot, _ = compute_fk_pose_for_q(fk_ctx, q_seed)
    offset = np.array([0.01, 0.0, 0.0], dtype=np.float64)
    target_pos = np.array(
        [float(pose.position.x), float(pose.position.y), float(pose.position.z)],
        dtype=np.float64,
    ) + offset

    result = compute_ik_for_position(
        fk_ctx.model,
        fk_ctx.data,
        fk_ctx.end_frame_id,
        target_pos,
        q_seed,
        max_iterations=100,
        tolerance=1e-4,
        max_ik_error=0.005,
    )
    assert result.success is True
    assert result.error < 0.005

    reached_pose, _, err = compute_fk_pose_for_q(
        fk_ctx, np.asarray(result.q_target, dtype=np.float64)
    )
    assert not err and reached_pose is not None
    reached = np.array(
        [reached_pose.position.x, reached_pose.position.y, reached_pose.position.z],
        dtype=np.float64,
    )
    assert np.linalg.norm(reached - target_pos) < 0.005


def test_position_only_allows_orientation_drift(fk_ctx):
    q_seed = np.asarray(TELEOP_INITIAL_Q, dtype=np.float64)
    pose, rot_seed, _ = compute_fk_pose_for_q(fk_ctx, q_seed)
    target_pos = np.array(
        [float(pose.position.x), float(pose.position.y) + 0.02, float(pose.position.z)],
        dtype=np.float64,
    )

    result = compute_ik_for_position(
        fk_ctx.model,
        fk_ctx.data,
        fk_ctx.end_frame_id,
        target_pos,
        q_seed,
        max_iterations=100,
        tolerance=1e-4,
        max_ik_error=0.005,
    )
    assert result.success is True

    _, rot_target, _ = compute_fk_pose_for_q(fk_ctx, np.asarray(result.q_target, dtype=np.float64))
    drift = orientation_drift_rad(rot_seed, rot_target)
    assert drift > 1e-6
    assert result.error < 0.005


def test_position_only_first_step_joint_delta_bounded(fk_ctx):
    _, max_first_step = _y_jog_ticks(
        fk_ctx,
        _ik_config(task_mode="position_only"),
        ticks=1,
        step_m=0.001,
    )
    assert max_first_step < 0.25


def test_position_only_y_jog_joint1_bounded(fk_ctx):
    j1_full, _ = _y_jog_ticks(
        fk_ctx,
        _ik_config(task_mode="full_6d"),
        ticks=15,
        step_m=0.001,
    )
    j1_pos, _ = _y_jog_ticks(
        fk_ctx,
        _ik_config(task_mode="position_only"),
        ticks=15,
        step_m=0.001,
    )
    assert abs(j1_pos) < abs(j1_full)
    assert abs(j1_full) > 0.05


def test_position_only_still_applies_joint_delta_gate(fk_ctx):
    pose, _, _ = compute_fk_pose_for_q(fk_ctx, np.asarray(TELEOP_INITIAL_Q, dtype=np.float64))
    far_q = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    mock_ik = IkSolveResult(
        success=True,
        q_target=far_q,
        error=0.001,
        iterations=10,
        reason="",
    )
    with patch(
        "rebotarm_cartesian_teleop.jog_core_logic.compute_ik_for_position",
        return_value=mock_ik,
    ):
        q_target, ok, reason, diag = call_solve_target_ik(
            fk_ctx,
            pose,
            TELEOP_INITIAL_Q,
            target_x=float(pose.position.x) + 0.01,
            target_y=float(pose.position.y),
            target_z=float(pose.position.z),
            ik_config=_ik_config(task_mode="position_only", max_joint_delta_rad=1e-6),
        )
    assert ok is False
    assert q_target == []
    assert reason == "JOINT_DELTA_TOO_LARGE"
    assert diag is not None
    assert diag.rejection_reason == "JOINT_DELTA_TOO_LARGE"


def test_position_only_still_applies_joint_near_limit_gate(fk_ctx):
    from rebotarm_cartesian_teleop.ik_quality_diagnostics import (
        joint_limits_from_model,
        joint_names_from_model,
    )

    names = joint_names_from_model(fk_ctx.model)
    lo, hi = joint_limits_from_model(fk_ctx.model)
    near_limit_q = list(TELEOP_INITIAL_Q)
    near_limit_q[4] = hi[4] - 0.01

    q_target, ok, reason, info = reject_ik_if_near_joint_limit(
        near_limit_q,
        names,
        lo,
        hi,
        JointLimitRejectConfig(reject_margin_rad=0.05),
    )
    assert ok is False
    assert reason == "JOINT_NEAR_LIMIT"
    assert info is not None
    assert info.joint == "joint5"


def test_position_only_ik_no_effect_gate_still_applies(fk_ctx):
    q_sim = np.asarray(TELEOP_INITIAL_Q, dtype=np.float64)
    pose, _, _ = compute_fk_pose_for_q(fk_ctx, q_sim)
    candidate_x = float(pose.position.x) + 0.01
    candidate_y = float(pose.position.y)
    candidate_z = float(pose.position.z)
    seed_q = [float(v) for v in q_sim]

    mock_ik = IkSolveResult(
        success=True,
        q_target=seed_q,
        error=0.001,
        iterations=10,
        reason="",
    )
    with patch(
        "rebotarm_cartesian_teleop.jog_core_logic.compute_ik_for_position",
        return_value=mock_ik,
    ):
        q_target, ik_success, ik_reason, _ = call_solve_target_ik(
            fk_ctx,
            pose,
            q_sim,
            target_x=candidate_x,
            target_y=candidate_y,
            target_z=candidate_z,
            ik_config=_ik_config(task_mode="position_only"),
        )
    assert ik_success is True

    q_target, ik_success, ik_reason, metrics = reject_ik_if_no_effect(
        fk_ctx,
        q_sim,
        q_target,
        candidate_x,
        candidate_y,
        candidate_z,
        IkNoEffectConfig(
            candidate_step_min_m=0.0005,
            reached_step_min_m=0.0001,
            q_step_min_norm=1e-6,
        ),
    )
    assert ik_success is False
    assert ik_reason == "IK_NO_EFFECT"
    assert metrics is not None
    assert metrics.candidate_step_m > 0.0005
    assert metrics.reached_step_m < 0.0001
    assert metrics.q_step_norm < 1e-6
