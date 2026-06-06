"""Tests for simulation-only fake JointState publishing logic."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import call_solve_target_ik

from rebotarm_cartesian_teleop.fake_joint_state import (
    FAKE_JOINT_NAMES,
    build_fake_joint_state,
    fake_joint_positions_to_publish,
    update_last_valid_fake_q,
)
from rebotarm_cartesian_teleop.fk_kinematics import compute_fk_pose, init_fk_context
from rebotarm_cartesian_teleop.jog_core_logic import IkConfig
from rebotarm_cartesian_teleop.sdk_path import ensure_rebot_sdk_in_syspath


def _sdk_available() -> bool:
    try:
        ensure_rebot_sdk_in_syspath()
        return True
    except FileNotFoundError:
        return False


Q_CURRENT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
Q_TARGET_OK = [0.1, 0.2, -0.1, 0.05, -0.05, 0.0]


def test_ik_success_fake_positions_equal_q_target():
    last, positions = fake_joint_positions_to_publish(
        enabled=True,
        last_valid_fake_q=None,
        q_current=Q_CURRENT,
        ik_success=True,
        q_target=Q_TARGET_OK,
    )
    assert last == Q_TARGET_OK
    assert positions == Q_TARGET_OK


def test_ik_failure_freezes_last_valid_fake_q():
    last, positions = fake_joint_positions_to_publish(
        enabled=True,
        last_valid_fake_q=Q_TARGET_OK,
        q_current=Q_CURRENT,
        ik_success=False,
        q_target=[],
    )
    assert last == Q_TARGET_OK
    assert positions == Q_TARGET_OK


def test_before_ik_success_uses_q_current():
    last, positions = fake_joint_positions_to_publish(
        enabled=True,
        last_valid_fake_q=None,
        q_current=Q_CURRENT,
        ik_success=False,
        q_target=[],
    )
    assert last == Q_CURRENT
    assert positions == Q_CURRENT


def test_build_fake_joint_state_joint_names():
    msg = build_fake_joint_state(Q_CURRENT)
    assert list(msg.name) == list(FAKE_JOINT_NAMES)
    assert msg.name == [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ]
    assert list(msg.position) == Q_CURRENT


def test_publish_disabled_returns_none_positions():
    last, positions = fake_joint_positions_to_publish(
        enabled=False,
        last_valid_fake_q=Q_TARGET_OK,
        q_current=Q_CURRENT,
        ik_success=True,
        q_target=[0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
    )
    assert last == Q_TARGET_OK
    assert positions is None


def test_update_last_valid_does_not_mutate_inputs():
    last_valid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    q_current = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    snapshot = list(last_valid)
    result = update_last_valid_fake_q(last_valid, q_current, ik_success=False, q_target=[])
    assert result == snapshot
    assert last_valid == snapshot


@pytest.mark.skipif(not _sdk_available(), reason="reBotArm_control_py not found")
def test_q_current_unchanged_after_fake_update_with_ik():
    fk_ctx = init_fk_context("", "end_link", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q_before = fk_ctx.q_current.copy()
    pose, _ = compute_fk_pose(fk_ctx)
    ik_config = IkConfig(
        max_iterations=100,
        tolerance=1e-4,
        max_ik_error=0.005,
        max_joint_delta_rad=0.25,
    )
    q_target, ik_success, _, _ = call_solve_target_ik(
        fk_ctx,
        pose,
        fk_ctx.q_current,
        target_x=pose.position.x,
        target_y=pose.position.y,
        target_z=pose.position.z,
        ik_config=ik_config,
    )
    q_current_list = [float(v) for v in fk_ctx.q_current]
    fake_joint_positions_to_publish(
        enabled=True,
        last_valid_fake_q=None,
        q_current=q_current_list,
        ik_success=ik_success,
        q_target=q_target,
    )
    assert np.allclose(fk_ctx.q_current, q_before)


@pytest.mark.skipif(not _sdk_available(), reason="reBotArm_control_py not found")
def test_q_target_empty_on_ik_failure_while_fake_freezes():
    fk_ctx = init_fk_context("", "end_link", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pose, _ = compute_fk_pose(fk_ctx)
    ik_config = IkConfig(
        max_iterations=100,
        tolerance=1e-4,
        max_ik_error=0.005,
        max_joint_delta_rad=0.25,
    )
    q_target_ok, ok1, _, _ = call_solve_target_ik(
        fk_ctx,
        pose,
        fk_ctx.q_current,
        target_x=pose.position.x,
        target_y=pose.position.y,
        target_z=pose.position.z,
        ik_config=ik_config,
    )
    assert ok1 is True
    assert len(q_target_ok) == 6

    last_valid, positions = fake_joint_positions_to_publish(
        enabled=True,
        last_valid_fake_q=None,
        q_current=[float(v) for v in fk_ctx.q_current],
        ik_success=ok1,
        q_target=q_target_ok,
    )
    assert positions == q_target_ok

    strict = IkConfig(
        max_iterations=ik_config.max_iterations,
        tolerance=ik_config.tolerance,
        max_ik_error=ik_config.max_ik_error,
        max_joint_delta_rad=1e-6,
    )
    q_target_fail, ok2, reason, _ = call_solve_target_ik(
        fk_ctx,
        pose,
        q_target_ok,
        target_x=0.45,
        target_y=0.25,
        target_z=0.40,
        ik_config=strict,
    )
    assert ok2 is False
    assert q_target_fail == []
    assert reason == "JOINT_DELTA_TOO_LARGE"

    last_valid, positions = fake_joint_positions_to_publish(
        enabled=True,
        last_valid_fake_q=last_valid,
        q_current=[float(v) for v in fk_ctx.q_current],
        ik_success=ok2,
        q_target=q_target_fail,
    )
    assert positions == last_valid
    assert positions != q_target_fail
