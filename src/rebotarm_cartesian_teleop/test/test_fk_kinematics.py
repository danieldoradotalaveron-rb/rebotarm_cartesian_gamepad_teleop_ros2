"""Unit tests for pure FK init/compute (requires vendored SDK)."""

from __future__ import annotations

import pytest

from rebotarm_cartesian_teleop.fk_kinematics import compute_fk_pose, init_fk_context
from rebotarm_cartesian_teleop.sdk_path import ensure_rebot_sdk_in_syspath


def _sdk_available() -> bool:
    try:
        ensure_rebot_sdk_in_syspath()
        return True
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(not _sdk_available(), reason="reBotArm_control_py not found")


def test_init_fk_with_zero_q_succeeds():
    ctx = init_fk_context("", "end_link", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert ctx.ok is True
    assert ctx.error == ""
    assert ctx.model is not None
    assert ctx.q_current is not None
    assert len(ctx.q_current) == 6


def test_invalid_initial_q_length():
    ctx = init_fk_context("", "end_link", [0.0, 0.0, 0.0])
    assert ctx.ok is False
    assert ctx.error == "INVALID_INITIAL_Q"


def test_missing_ee_frame():
    ctx = init_fk_context("", "not_a_real_frame", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert ctx.ok is False
    assert ctx.error == "MISSING_EE_FRAME"


def test_fk_pose_at_zero_configuration():
    ctx = init_fk_context("", "end_link", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pose, err = compute_fk_pose(ctx)
    assert err == ""
    assert pose is not None
    assert pose.position.x == pytest.approx(0.2603, abs=1e-3)
    assert pose.position.y == pytest.approx(0.0, abs=1e-4)
    assert pose.position.z == pytest.approx(0.1917, abs=1e-3)
    assert pose.orientation.w == pytest.approx(1.0, abs=1e-3)


def test_fk_pose_at_teleop_initial_configuration():
    from conftest import TELEOP_INITIAL_Q

    ctx = init_fk_context("", "end_link", TELEOP_INITIAL_Q)
    pose, err = compute_fk_pose(ctx)
    assert err == ""
    assert pose is not None
    assert pose.position.x == pytest.approx(0.2721, abs=1e-3)
    assert pose.position.y == pytest.approx(0.0, abs=1e-4)
    assert pose.position.z == pytest.approx(0.2697, abs=1e-3)
