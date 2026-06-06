"""Tests for dry-run IK acceptance policy (error threshold vs SDK success flag)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from conftest import call_solve_target_ik

from rebotarm_cartesian_teleop.fk_kinematics import init_fk_context
from rebotarm_cartesian_teleop.ik_kinematics import IkSolveResult, compute_ik_for_pose
from rebotarm_cartesian_teleop.jog_core_logic import IkConfig


def _mock_ik_result(*, success: bool, error: float, q: np.ndarray, iterations: int = 100):
    result = MagicMock()
    result.success = success
    result.error = error
    result.iterations = iterations
    result.q = q
    return result


def _call_compute_ik_with_mock(mock_result, *, max_ik_error: float = 0.005):
    model = MagicMock()
    model.nq = 6
    q_seed = np.zeros(6, dtype=np.float64)
    pos = np.zeros(3, dtype=np.float64)
    rot = np.eye(3, dtype=np.float64)

    with (
        patch("rebotarm_cartesian_teleop.ik_kinematics.ensure_rebot_sdk_in_syspath"),
        patch(
            "reBotArm_control_py.kinematics.inverse_kinematics.solve_ik",
            return_value=mock_result,
        ),
    ):
        return compute_ik_for_pose(
            model,
            MagicMock(),
            0,
            pos,
            rot,
            q_seed,
            max_iterations=100,
            tolerance=0.001,
            max_ik_error=max_ik_error,
        )


def test_accepts_sdk_failure_when_error_within_max():
    q = np.array([0.01, 0.02, -0.01, 0.0, 0.0, 0.0], dtype=np.float64)
    result = _call_compute_ik_with_mock(
        _mock_ik_result(success=False, error=0.004213, q=q),
    )
    assert result.success is True
    assert result.reason == ""
    assert result.q_target == pytest.approx(list(q))


def test_rejects_sdk_failure_when_error_above_max():
    q = np.zeros(6, dtype=np.float64)
    result = _call_compute_ik_with_mock(
        _mock_ik_result(success=False, error=0.006, q=q),
    )
    assert result.success is False
    assert result.reason == "IK_ERROR_TOO_HIGH"
    assert result.q_target == []
    assert result.error == pytest.approx(0.006)


def test_accepts_sdk_success_as_before():
    q = np.zeros(6, dtype=np.float64)
    result = _call_compute_ik_with_mock(
        _mock_ik_result(success=True, error=0.001, q=q),
    )
    assert result.success is True
    assert result.reason == ""
    assert result.q_target == pytest.approx([0.0] * 6)


def test_rejects_invalid_q_length():
    q = np.zeros(3, dtype=np.float64)
    result = _call_compute_ik_with_mock(
        _mock_ik_result(success=False, error=0.001, q=q),
    )
    assert result.success is False
    assert result.reason == "INVALID_IK_RESULT"
    assert result.q_target == []


def test_solve_target_ik_joint_delta_rejects_acceptable_error():
    fk_ctx = init_fk_context("", "end_link", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    if not fk_ctx.ok:
        pytest.skip("FK init failed")

    ik_config = IkConfig(
        max_iterations=100,
        tolerance=0.001,
        max_ik_error=0.005,
        max_joint_delta_rad=1e-6,
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
        from rebotarm_cartesian_teleop.fk_kinematics import compute_fk_pose

        pose, _ = compute_fk_pose(fk_ctx)
        q_target, ik_success, ik_reason, diag = call_solve_target_ik(
            fk_ctx,
            pose,
            fk_ctx.q_current,
            target_x=0.30,
            target_y=0.0,
            target_z=0.20,
            ik_config=ik_config,
        )

    assert ik_success is False
    assert q_target == []
    assert ik_reason == "JOINT_DELTA_TOO_LARGE"
    assert diag is not None
    assert diag.rejection_reason == "JOINT_DELTA_TOO_LARGE"
