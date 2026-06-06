"""Tests for q_sim simulation coherence (microtask 7.2)."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import TELEOP_INITIAL_Q, default_workspace, make_cmd

from rebotarm_cartesian_teleop.fk_kinematics import compute_fk_pose_for_q, init_fk_context
from rebotarm_cartesian_teleop.jog_core_logic import (
    IkConfig,
    compute_candidate_drift_m,
    compute_candidate_target,
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


pytestmark = pytest.mark.skipif(not _sdk_available(), reason="reBotArm_control_py not found")

# Off upper joint limits so repeated +X moves are reachable (matches cartesian_teleop.yaml).
INITIAL_Q = TELEOP_INITIAL_Q


@pytest.fixture
def fk_ctx():
    return init_fk_context("", "end_link", INITIAL_Q)


@pytest.fixture
def ik_config():
    return IkConfig(
        max_iterations=100,
        tolerance=1e-4,
        max_ik_error=0.005,
        max_joint_delta_rad=0.25,
    )


def test_startup_coherence(fk_ctx):
    q_sim = np.asarray(INITIAL_Q, dtype=np.float64)
    cx, cy, cz, rot, pose, err = resync_committed_from_q_sim(fk_ctx, q_sim)
    assert err == ""
    assert pose is not None
    assert rot is not None
    assert np.allclose(q_sim, INITIAL_Q)
    assert cx == pytest.approx(float(pose.position.x))
    assert cy == pytest.approx(float(pose.position.y))
    assert cz == pytest.approx(float(pose.position.z))


def test_ik_success_resyncs_committed_from_fk_not_candidate(fk_ctx, ik_config):
    q_sim = np.asarray(INITIAL_Q, dtype=np.float64)
    pose, sim_rot, _ = compute_fk_pose_for_q(fk_ctx, q_sim)
    sim_x, sim_y, sim_z = float(pose.position.x), float(pose.position.y), float(pose.position.z)
    start_x = sim_x
    cand_x, cand_y, cand_z, _ = compute_candidate_target(
        sim_x, sim_y, sim_z, make_cmd(linear_x=0.03), 0.02, default_workspace()
    )

    q_target, ok, _, _ = solve_target_ik(
        fk_ctx=fk_ctx,
        state_name="ACTIVE",
        target_x=cand_x,
        target_y=cand_y,
        target_z=cand_z,
        target_rotation=sim_rot,
        q_seed=q_sim,
        ik_config=ik_config,
    )
    assert ok is True
    q_sim = update_q_sim_on_ik_success(q_sim, q_target, True)
    cx, cy, cz, _, fk_pose, _ = resync_committed_from_q_sim(fk_ctx, q_sim)

    assert (cx, cy, cz) == pytest.approx(
        (float(fk_pose.position.x), float(fk_pose.position.y), float(fk_pose.position.z))
    )
    assert cx == pytest.approx(float(fk_pose.position.x))
    assert cx >= start_x


def test_resync_always_from_fk_even_when_drift_is_tiny(fk_ctx, ik_config):
    q_sim = np.asarray(INITIAL_Q, dtype=np.float64)
    pose, sim_rot, _ = compute_fk_pose_for_q(fk_ctx, q_sim)
    sim_x = float(pose.position.x)
    q_target, ok, _, _ = solve_target_ik(
        fk_ctx=fk_ctx,
        state_name="ACTIVE",
        target_x=sim_x + 0.0006,
        target_y=float(pose.position.y),
        target_z=float(pose.position.z),
        target_rotation=sim_rot,
        q_seed=q_sim,
        ik_config=ik_config,
    )
    assert ok is True
    drift = compute_candidate_drift_m(
        fk_ctx, sim_x + 0.0006, float(pose.position.y), float(pose.position.z), q_target
    )
    cx, cy, cz, _, fk_pose, _ = resync_committed_from_q_sim(
        fk_ctx, update_q_sim_on_ik_success(q_sim, q_target, True)
    )
    assert drift >= 0.0
    assert cx == pytest.approx(float(fk_pose.position.x))


def test_ik_failure_freezes_sim_and_committed(fk_ctx, ik_config):
    q_sim = np.asarray(INITIAL_Q, dtype=np.float64)
    cx, cy, cz, committed_rot, _, _ = resync_committed_from_q_sim(fk_ctx, q_sim)
    pose, sim_rot, _ = compute_fk_pose_for_q(fk_ctx, q_sim)
    cand_x, cand_y, cand_z, _ = compute_candidate_target(
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
        make_cmd(linear_x=0.5),
        0.1,
        default_workspace(),
    )
    q_target, ok, reason, _ = solve_target_ik(
        fk_ctx=fk_ctx,
        state_name="ACTIVE",
        target_x=cand_x,
        target_y=cand_y,
        target_z=cand_z,
        target_rotation=sim_rot,
        q_seed=q_sim,
        ik_config=ik_config,
    )
    assert ok is False
    assert q_target == []
    assert reason in ("JOINT_DELTA_TOO_LARGE", "IK_ERROR_TOO_HIGH")

    q_after = update_q_sim_on_ik_success(q_sim, q_target, False)
    assert np.allclose(q_after, q_sim)
    cx2, cy2, cz2, rot2, _, _ = resync_committed_from_q_sim(fk_ctx, q_after)
    assert (cx2, cy2, cz2) == pytest.approx((cx, cy, cz))
    assert np.allclose(rot2, committed_rot)


def test_multi_tick_x_moves_no_committed_drift(fk_ctx, ik_config):
    q_sim = np.asarray(INITIAL_Q, dtype=np.float64)
    cx, cy, cz, committed_rot, _, _ = resync_committed_from_q_sim(fk_ctx, q_sim)
    cmd = make_cmd(linear_x=0.03)

    start_x = float(compute_fk_pose_for_q(fk_ctx, np.asarray(INITIAL_Q, dtype=np.float64))[0].position.x)

    for tick in range(50):
        pose, sim_rot, _ = compute_fk_pose_for_q(fk_ctx, q_sim)
        sim_x, sim_y, sim_z = float(pose.position.x), float(pose.position.y), float(pose.position.z)
        cand_x, cand_y, cand_z, _ = compute_candidate_target(
            sim_x, sim_y, sim_z, cmd, 0.02, default_workspace()
        )
        q_target, ok, reason, _ = solve_target_ik(
            fk_ctx=fk_ctx,
            state_name="ACTIVE",
            target_x=cand_x,
            target_y=cand_y,
            target_z=cand_z,
            target_rotation=sim_rot,
            q_seed=q_sim,
            ik_config=ik_config,
        )
        assert ok is True, f"tick {tick}: {reason}"
        q_sim = update_q_sim_on_ik_success(q_sim, q_target, True)
        cx, cy, cz, committed_rot, _, _ = resync_committed_from_q_sim(fk_ctx, q_sim)
        fk_pose, _, _ = compute_fk_pose_for_q(fk_ctx, q_sim)
        assert cx == pytest.approx(float(fk_pose.position.x), abs=1e-6)
        assert cy == pytest.approx(float(fk_pose.position.y), abs=1e-6)
        assert cz == pytest.approx(float(fk_pose.position.z), abs=1e-6)

    assert cx > start_x


def test_orientation_from_q_sim_not_initial_q(fk_ctx, ik_config):
    q_sim = np.array([0.5, -0.3, -0.3, 0.0, 0.0, 0.0], dtype=np.float64)
    _, rot_initial, _ = compute_fk_pose_for_q(fk_ctx, np.zeros(6, dtype=np.float64))
    pose, sim_rot, _ = compute_fk_pose_for_q(fk_ctx, q_sim)

    cand_x, cand_y, cand_z, _ = compute_candidate_target(
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
        make_cmd(linear_z=0.01),
        0.1,
        default_workspace(),
    )
    _, ok_sim_rot, _, _ = solve_target_ik(
        fk_ctx=fk_ctx,
        state_name="ACTIVE",
        target_x=cand_x,
        target_y=cand_y,
        target_z=cand_z,
        target_rotation=sim_rot,
        q_seed=q_sim,
        ik_config=ik_config,
    )
    _, ok_stale_rot, reason_stale, _ = solve_target_ik(
        fk_ctx=fk_ctx,
        state_name="ACTIVE",
        target_x=cand_x,
        target_y=cand_y,
        target_z=cand_z,
        target_rotation=rot_initial,
        q_seed=q_sim,
        ik_config=ik_config,
    )
    assert ok_sim_rot is True
    assert ok_stale_rot is False
    assert reason_stale in ("IK_ERROR_TOO_HIGH", "JOINT_DELTA_TOO_LARGE")
