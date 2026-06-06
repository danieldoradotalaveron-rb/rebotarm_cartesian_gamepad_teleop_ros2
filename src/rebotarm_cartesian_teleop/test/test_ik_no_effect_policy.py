"""Tests for IK_NO_EFFECT anti phantom-success policy."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from conftest import TELEOP_INITIAL_Q, call_solve_target_ik, default_workspace, make_cmd

from rebotarm_cartesian_teleop.fake_joint_state import fake_joint_positions_to_publish
from rebotarm_cartesian_teleop.fk_kinematics import init_fk_context
from rebotarm_cartesian_teleop.ik_kinematics import IkSolveResult
from rebotarm_cartesian_teleop.jog_core_logic import (
    IkConfig,
    IkNoEffectConfig,
    IkNoEffectMetrics,
    build_cartesian_jog_state,
    build_committed_target_pose,
    compute_candidate_target,
    is_ik_no_effect,
    reject_ik_if_no_effect,
    resync_committed_from_q_sim,
    update_q_sim_on_ik_success,
)
from rebotarm_cartesian_teleop.sdk_path import ensure_rebot_sdk_in_syspath


def _sdk_available() -> bool:
    try:
        ensure_rebot_sdk_in_syspath()
        return True
    except FileNotFoundError:
        return False


DEFAULT_NO_EFFECT = IkNoEffectConfig(
    candidate_step_min_m=0.0005,
    reached_step_min_m=0.0001,
    q_step_min_norm=1e-6,
)

WS = default_workspace()
CMD = make_cmd(linear_x=0.1)


@pytest.fixture
def fk_ctx():
    ctx = init_fk_context("", "end_link", TELEOP_INITIAL_Q)
    if not ctx.ok:
        pytest.skip("FK init failed")
    return ctx


def _committed_from_q(fk_ctx, q_sim: np.ndarray):
    cx, cy, cz, rot, pose, err = resync_committed_from_q_sim(fk_ctx, q_sim)
    assert not err
    assert pose is not None
    assert rot is not None
    return (cx, cy, cz), rot, pose, q_sim


def _phantom_success_flow(fk_ctx, *, no_effect_config: IkNoEffectConfig = DEFAULT_NO_EFFECT):
    q_sim = np.asarray(fk_ctx.q_current, dtype=np.float64).copy()
    committed, sim_rot, current_pose, _ = _committed_from_q(fk_ctx, q_sim)
    fk_before = (
        float(current_pose.position.x),
        float(current_pose.position.y),
        float(current_pose.position.z),
    )
    candidate_x, candidate_y, candidate_z, _ = compute_candidate_target(
        *committed, CMD, 0.1, WS
    )
    seed_q = [float(v) for v in q_sim]
    mock_ik = IkSolveResult(
        success=True,
        q_target=seed_q,
        error=0.001,
        iterations=10,
        reason="",
    )
    with patch(
        "rebotarm_cartesian_teleop.jog_core_logic.compute_ik_for_pose",
        return_value=mock_ik,
    ):
        q_target, ik_success, ik_reason, _ = call_solve_target_ik(
            fk_ctx,
            current_pose,
            q_sim,
            target_x=candidate_x,
            target_y=candidate_y,
            target_z=candidate_z,
            ik_config=IkConfig(100, 0.001, 0.005, 0.25),
        )
    assert ik_success is True
    q_target, ik_success, ik_reason, metrics = reject_ik_if_no_effect(
        fk_ctx,
        q_sim,
        q_target,
        candidate_x,
        candidate_y,
        candidate_z,
        no_effect_config,
        fk_position_before=fk_before,
    )
    return {
        "q_sim_before": q_sim,
        "committed_before": committed,
        "sim_rot": sim_rot,
        "current_pose": current_pose,
        "candidate": (candidate_x, candidate_y, candidate_z),
        "q_target": q_target,
        "ik_success": ik_success,
        "ik_reason": ik_reason,
        "metrics": metrics,
    }


def test_phantom_success_rejected_as_ik_no_effect(fk_ctx):
    result = _phantom_success_flow(fk_ctx)
    assert result["metrics"] is not None
    assert result["metrics"].candidate_step_m > DEFAULT_NO_EFFECT.candidate_step_min_m
    assert result["metrics"].reached_step_m < DEFAULT_NO_EFFECT.reached_step_min_m
    assert result["metrics"].q_step_norm < DEFAULT_NO_EFFECT.q_step_min_norm
    assert result["ik_success"] is False
    assert result["ik_reason"] == "IK_NO_EFFECT"
    assert result["q_target"] == []

    q_sim = result["q_sim_before"]
    committed_before = result["committed_before"]
    q_after = update_q_sim_on_ik_success(q_sim, result["q_target"], result["ik_success"])
    assert np.allclose(q_after, q_sim)
    committed_after = resync_committed_from_q_sim(fk_ctx, q_after)[:3]
    assert committed_after == pytest.approx(committed_before)

    committed_pose = build_committed_target_pose(*committed_before, result["sim_rot"])
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
        current_pose=result["current_pose"],
        target_pose=committed_pose,
        q_current=[float(v) for v in q_sim],
        q_target=result["q_target"],
        ik_success=result["ik_success"],
        ik_reason=result["ik_reason"],
    )
    assert msg.ik_success is False
    assert msg.rejection_reason == "IK_NO_EFFECT"
    assert msg.current_pose.position.x == pytest.approx(committed_before[0])
    assert msg.target_pose.position.x == pytest.approx(committed_before[0])


def test_phantom_success_freezes_fake_joint_states(fk_ctx):
    result = _phantom_success_flow(fk_ctx)
    last_valid = [float(v) for v in result["q_sim_before"]]
    updated, positions = fake_joint_positions_to_publish(
        enabled=True,
        last_valid_fake_q=last_valid,
        q_current=[float(v) for v in result["q_sim_before"]],
        ik_success=result["ik_success"],
        q_target=result["q_target"],
    )
    assert updated == last_valid
    assert positions == last_valid


@pytest.mark.skipif(not _sdk_available(), reason="reBotArm_control_py not found")
def test_small_real_movement_accepted(fk_ctx):
    q_sim = np.asarray(fk_ctx.q_current, dtype=np.float64).copy()
    committed, sim_rot, current_pose, _ = _committed_from_q(fk_ctx, q_sim)
    fk_before = (
        float(current_pose.position.x),
        float(current_pose.position.y),
        float(current_pose.position.z),
    )
    cmd = make_cmd(linear_x=0.01)
    candidate_x, candidate_y, candidate_z, _ = compute_candidate_target(
        *committed, cmd, 0.1, WS
    )
    from rebotarm_cartesian_teleop.jog_core_logic import solve_target_ik

    q_target, ik_success, ik_reason, _ = solve_target_ik(
        fk_ctx=fk_ctx,
        state_name="ACTIVE",
        target_x=candidate_x,
        target_y=candidate_y,
        target_z=candidate_z,
        target_rotation=sim_rot,
        q_seed=q_sim,
        ik_config=IkConfig(100, 1e-4, 0.005, 0.25),
    )
    assert ik_success is True
    q_target, ik_success, ik_reason, metrics = reject_ik_if_no_effect(
        fk_ctx,
        q_sim,
        q_target,
        candidate_x,
        candidate_y,
        candidate_z,
        DEFAULT_NO_EFFECT,
        fk_position_before=fk_before,
    )
    assert ik_success is True
    assert ik_reason == ""
    assert metrics is not None
    assert metrics.reached_step_m >= DEFAULT_NO_EFFECT.reached_step_min_m or (
        metrics.q_step_norm >= DEFAULT_NO_EFFECT.q_step_min_norm
    )

    q_after = update_q_sim_on_ik_success(q_sim, q_target, ik_success)
    committed_after = resync_committed_from_q_sim(fk_ctx, q_after)
    assert committed_after[0] >= committed[0]
    assert not np.allclose(q_after, q_sim)


def test_solver_failure_unchanged(fk_ctx):
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
    mock_ik = IkSolveResult(
        success=False,
        q_target=[],
        error=0.006,
        iterations=100,
        reason="IK_ERROR_TOO_HIGH",
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
            ik_config=IkConfig(100, 0.001, 0.005, 0.25),
        )
    assert ik_success is False
    assert ik_reason == "IK_ERROR_TOO_HIGH"
    assert q_target == []
    assert diag is not None
    assert diag.rejection_reason == "IK_ERROR_TOO_HIGH"


def test_threshold_config_disables_rejection(fk_ctx):
    permissive = IkNoEffectConfig(
        candidate_step_min_m=10.0,
        reached_step_min_m=0.0001,
        q_step_min_norm=1e-6,
    )
    result = _phantom_success_flow(fk_ctx, no_effect_config=permissive)
    assert result["ik_success"] is True
    assert result["ik_reason"] == ""
    assert result["q_target"] == [float(v) for v in result["q_sim_before"]]


def test_is_ik_no_effect_respects_individual_thresholds():
    metrics = IkNoEffectMetrics(
        candidate_step_m=0.001,
        reached_step_m=0.0,
        q_step_norm=0.0,
    )
    assert is_ik_no_effect(metrics, DEFAULT_NO_EFFECT) is True

    metrics_real_reach = IkNoEffectMetrics(
        candidate_step_m=0.001,
        reached_step_m=0.0002,
        q_step_norm=0.0,
    )
    assert is_ik_no_effect(metrics_real_reach, DEFAULT_NO_EFFECT) is False

    metrics_real_q = IkNoEffectMetrics(
        candidate_step_m=0.001,
        reached_step_m=0.0,
        q_step_norm=0.01,
    )
    assert is_ik_no_effect(metrics_real_q, DEFAULT_NO_EFFECT) is False

    metrics_tiny_candidate = IkNoEffectMetrics(
        candidate_step_m=0.0001,
        reached_step_m=0.0,
        q_step_norm=0.0,
    )
    assert is_ik_no_effect(metrics_tiny_candidate, DEFAULT_NO_EFFECT) is False
