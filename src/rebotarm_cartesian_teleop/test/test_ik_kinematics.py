"""Unit tests for pure IK helper (requires vendored SDK)."""

from __future__ import annotations

import numpy as np
import pytest

from rebotarm_cartesian_teleop.ik_kinematics import compute_ik_for_pose
from rebotarm_cartesian_teleop.sdk_path import ensure_rebot_sdk_in_syspath


def _sdk_available() -> bool:
    try:
        ensure_rebot_sdk_in_syspath()
        return True
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(not _sdk_available(), reason="reBotArm_control_py not found")


@pytest.fixture
def ik_setup():
    ensure_rebot_sdk_in_syspath()
    from reBotArm_control_py.kinematics import (
        compute_fk,
        get_end_effector_frame_id,
        load_robot_model,
    )

    model = load_robot_model()
    data = model.createData()
    end_frame_id = get_end_effector_frame_id(model)
    q0 = np.zeros(model.nq, dtype=np.float64)
    position, rotation, _ = compute_fk(model, q0)
    return model, data, end_frame_id, q0, position, rotation


def test_fk_round_trip_ik_succeeds(ik_setup):
    model, data, end_frame_id, q0, position, rotation = ik_setup
    result = compute_ik_for_pose(
        model,
        data,
        end_frame_id,
        position,
        rotation,
        q0,
        max_iterations=100,
        tolerance=1e-4,
        max_ik_error=1e-3,
    )
    assert result.success is True
    assert result.reason == ""
    assert len(result.q_target) == model.nq == 6
    assert max(abs(a - b) for a, b in zip(result.q_target, q0, strict=True)) < 1e-3


def test_ik_error_too_high_when_threshold_tiny(ik_setup):
    model, data, end_frame_id, q0, position, rotation = ik_setup
    result = compute_ik_for_pose(
        model,
        data,
        end_frame_id,
        position,
        rotation,
        q0,
        max_iterations=100,
        tolerance=1e-4,
        max_ik_error=1e-20,
    )
    assert result.success is False
    assert result.reason == "IK_ERROR_TOO_HIGH"
    assert result.q_target == []


def test_invalid_input_does_not_crash(ik_setup):
    model, data, end_frame_id, q0, position, rotation = ik_setup
    bad_rot = np.zeros((2, 2))
    result = compute_ik_for_pose(
        model,
        data,
        end_frame_id,
        position,
        bad_rot,
        q0,
        max_iterations=100,
        tolerance=1e-4,
        max_ik_error=1e-3,
    )
    assert result.success is False
    assert result.reason in ("IK_EXCEPTION", "IK_ERROR_TOO_HIGH")
    assert result.q_target == []


def test_invalid_q_seed_length_returns_failure(ik_setup):
    model, data, end_frame_id, _q0, position, rotation = ik_setup
    result = compute_ik_for_pose(
        model,
        data,
        end_frame_id,
        position,
        rotation,
        np.zeros(3),
        max_iterations=100,
        tolerance=1e-4,
        max_ik_error=1e-3,
    )
    assert result.success is False
    assert result.reason == "IK_EXCEPTION"
    assert result.q_target == []
