"""Stage 2b: joint1 anchor/global operational rejection gates."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import TELEOP_INITIAL_Q, call_solve_target_ik

from rebotarm_cartesian_teleop.fk_kinematics import compute_fk_pose_for_q, init_fk_context
from rebotarm_cartesian_teleop.ik_quality_diagnostics import (
    joint_limits_from_model,
    joint_names_from_model,
)
from rebotarm_cartesian_teleop.jog_core_logic import (
    IkConfig,
    IkNoEffectConfig,
    IkNoEffectMetrics,
    Joint1AnchorWindowConfig,
    Joint1GlobalOperationalLimitConfig,
    JointLimitRejectConfig,
    format_joint1_anchor_window_log,
    format_joint1_global_operational_limit_log,
    is_ik_no_effect,
    reject_ik_if_joint1_anchor_window,
    reject_ik_if_joint1_global_operational_limit,
    reject_ik_if_near_joint_limit,
    reject_ik_if_no_effect,
)

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
HARD_RAD = 1.20
GLOBAL_MIN = -1.60
GLOBAL_MAX = 1.60
GATES_OFF_GLOBAL = Joint1GlobalOperationalLimitConfig(enabled=False)
GATES_OFF_ANCHOR = Joint1AnchorWindowConfig(enabled=False)
GATES_ON_GLOBAL = Joint1GlobalOperationalLimitConfig(
    enabled=True, min_rad=GLOBAL_MIN, max_rad=GLOBAL_MAX
)
GATES_ON_ANCHOR = Joint1AnchorWindowConfig(enabled=True, hard_window_rad=HARD_RAD)


def _candidate_q(joint1: float) -> list[float]:
    q = [float(v) for v in TELEOP_INITIAL_Q]
    q[0] = joint1
    return q


def test_gates_disabled_preserve_behavior():
    for joint1 in (1.25, 2.0, -2.0):
        q = _candidate_q(joint1)
        q_out, ok, reason, _ = reject_ik_if_joint1_global_operational_limit(
            q, JOINT_NAMES, GATES_OFF_GLOBAL
        )
        assert ok is True
        assert reason == ""
        assert q_out == q

        q_out, ok, reason, _ = reject_ik_if_joint1_anchor_window(
            q, JOINT_NAMES, base_anchor_q=0.0, config=GATES_OFF_ANCHOR
        )
        assert ok is True
        assert reason == ""
        assert q_out == q


def test_global_cap_rejects_above_max():
    q_out, ok, reason, info = reject_ik_if_joint1_global_operational_limit(
        _candidate_q(1.61), JOINT_NAMES, GATES_ON_GLOBAL
    )
    assert ok is False
    assert reason == "JOINT1_GLOBAL_OPERATIONAL_LIMIT"
    assert q_out == []
    assert info is not None
    assert info.violation_side == "upper"
    assert "JOINT1_GLOBAL_OPERATIONAL_LIMIT" in format_joint1_global_operational_limit_log(info)


def test_global_cap_rejects_below_min():
    q_out, ok, reason, info = reject_ik_if_joint1_global_operational_limit(
        _candidate_q(-1.61), JOINT_NAMES, GATES_ON_GLOBAL
    )
    assert ok is False
    assert reason == "JOINT1_GLOBAL_OPERATIONAL_LIMIT"
    assert info is not None
    assert info.violation_side == "lower"


def test_global_cap_accepts_inside_range():
    for joint1 in (-1.60, 0.0, 1.09, 1.60):
        q_out, ok, reason, info = reject_ik_if_joint1_global_operational_limit(
            _candidate_q(joint1), JOINT_NAMES, GATES_ON_GLOBAL
        )
        assert ok is True, joint1
        assert reason == ""
        assert info is None
        assert q_out == _candidate_q(joint1)


def test_anchor_hard_rejects_local_drift():
    q_out, ok, reason, info = reject_ik_if_joint1_anchor_window(
        _candidate_q(1.25),
        JOINT_NAMES,
        base_anchor_q=0.0,
        config=GATES_ON_ANCHOR,
    )
    assert ok is False
    assert reason == "JOINT1_ANCHOR_WINDOW"
    assert q_out == []
    assert info is not None
    assert info.abs_delta_from_anchor == pytest.approx(1.25)
    assert "JOINT1_ANCHOR_WINDOW" in format_joint1_anchor_window_log(info)


def test_anchor_hard_accepts_known_good_drift():
    q_out, ok, reason, info = reject_ik_if_joint1_anchor_window(
        _candidate_q(1.09),
        JOINT_NAMES,
        base_anchor_q=0.0,
        config=GATES_ON_ANCHOR,
    )
    assert ok is True
    assert reason == ""
    assert info is None
    assert q_out == _candidate_q(1.09)


def test_global_cap_wins_before_anchor_when_both_trigger():
    q = _candidate_q(2.0)
    q_out, ok, reason, _ = reject_ik_if_joint1_global_operational_limit(
        q, JOINT_NAMES, GATES_ON_GLOBAL
    )
    assert ok is False
    assert reason == "JOINT1_GLOBAL_OPERATIONAL_LIMIT"

    q_out, ok, reason, _ = reject_ik_if_joint1_anchor_window(
        q, JOINT_NAMES, base_anchor_q=0.0, config=GATES_ON_ANCHOR
    )
    assert ok is False
    assert reason == "JOINT1_ANCHOR_WINDOW"


def test_global_cap_rejects_bad_pose_even_if_anchor_reset_near_bad_pose():
    q_out, ok, reason, info = reject_ik_if_joint1_global_operational_limit(
        _candidate_q(2.3),
        JOINT_NAMES,
        GATES_ON_GLOBAL,
    )
    assert ok is False
    assert reason == "JOINT1_GLOBAL_OPERATIONAL_LIMIT"
    assert info is not None

    q_out, ok, reason, info = reject_ik_if_joint1_anchor_window(
        _candidate_q(2.3),
        JOINT_NAMES,
        base_anchor_q=2.3,
        config=GATES_ON_ANCHOR,
    )
    assert ok is True
    assert reason == ""
    assert info is None
    assert q_out == _candidate_q(2.3)


@pytest.fixture
def fk_ctx():
    ctx = init_fk_context("", "end_link", TELEOP_INITIAL_Q)
    if not ctx.ok:
        pytest.skip("FK init failed")
    return ctx


def test_existing_joint_delta_too_large_still_works(fk_ctx):
    pose, _, _ = compute_fk_pose_for_q(fk_ctx, np.asarray(TELEOP_INITIAL_Q, dtype=np.float64))
    ik_config = IkConfig(
        max_iterations=100,
        tolerance=1e-4,
        max_ik_error=0.005,
        max_joint_delta_rad=0.01,
        task_mode="position_only",
    )
    _, ok, reason, _ = call_solve_target_ik(
        fk_ctx,
        pose,
        TELEOP_INITIAL_Q,
        target_x=float(pose.position.x) + 0.05,
        target_y=float(pose.position.y),
        target_z=float(pose.position.z),
        ik_config=ik_config,
    )
    assert ok is False
    assert reason == "JOINT_DELTA_TOO_LARGE"


def test_existing_joint_near_limit_still_works(fk_ctx):
    names = joint_names_from_model(fk_ctx.model)
    lo, hi = joint_limits_from_model(fk_ctx.model)
    candidate_q = [float(v) for v in TELEOP_INITIAL_Q]
    candidate_q[4] = 1.57
    q_out, ok, reason, _ = reject_ik_if_near_joint_limit(
        candidate_q,
        names,
        lo,
        hi,
        JointLimitRejectConfig(reject_margin_rad=0.05),
    )
    assert ok is False
    assert reason == "JOINT_NEAR_LIMIT"
    assert q_out == []


def test_existing_ik_no_effect_still_works(fk_ctx):
    cfg = IkNoEffectConfig()
    metrics = IkNoEffectMetrics(
        candidate_step_m=0.01,
        reached_step_m=0.0,
        q_step_norm=0.0,
    )
    assert is_ik_no_effect(metrics, cfg) is True

    q_before = np.asarray(TELEOP_INITIAL_Q, dtype=np.float64)
    pose, _, _ = compute_fk_pose_for_q(fk_ctx, q_before)
    fk_before = (float(pose.position.x), float(pose.position.y), float(pose.position.z))
    q_out, ok, reason, _ = reject_ik_if_no_effect(
        fk_ctx,
        q_before,
        list(q_before),
        fk_before[0] + 0.01,
        fk_before[1],
        fk_before[2],
        cfg,
        fk_position_before=fk_before,
    )
    assert ok is False
    assert reason == "IK_NO_EFFECT"
    assert q_out == []
