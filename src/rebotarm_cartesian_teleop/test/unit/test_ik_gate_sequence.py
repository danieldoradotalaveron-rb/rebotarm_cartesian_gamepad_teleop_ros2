"""Unit tests for post-IK gate sequence (apply_ik_gate_sequence)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from rebotarm_cartesian_teleop.fk_kinematics import compute_fk_pose_for_q, init_fk_context
from rebotarm_cartesian_teleop.jog_core_logic import (
    IkGateSequenceInput,
    IkNoEffectConfig,
    Joint1AnchorWindowConfig,
    Joint1GlobalOperationalLimitConfig,
    JointLimitRejectConfig,
    apply_ik_gate_sequence,
)
from rebotarm_cartesian_teleop.sdk_path import ensure_rebot_sdk_in_syspath

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
TELEOP_INITIAL_Q = [0.0, -0.3, -0.3, 0.0, 0.0, 0.0]
LOWER = [-2.8, -3.14, -3.14, -1.87, -1.57, -3.14]
UPPER = [2.8, 0.0, 0.0, 1.57, 1.57, 3.14]
HARD_RAD = 1.20
GLOBAL_MIN = -1.60
GLOBAL_MAX = 1.60
GATES_ON_GLOBAL = Joint1GlobalOperationalLimitConfig(
    enabled=True, min_rad=GLOBAL_MIN, max_rad=GLOBAL_MAX
)
GATES_ON_ANCHOR = Joint1AnchorWindowConfig(enabled=True, hard_window_rad=HARD_RAD)
GATES_OFF_GLOBAL = Joint1GlobalOperationalLimitConfig(enabled=False)
GATES_OFF_ANCHOR = Joint1AnchorWindowConfig(enabled=False)
JOINT_LIMIT_CFG = JointLimitRejectConfig(reject_margin_rad=0.05)
NO_EFFECT_CFG = IkNoEffectConfig()
FK_BEFORE = (0.26, 0.0, 0.19)


def _sdk_available() -> bool:
    try:
        ensure_rebot_sdk_in_syspath()
        return True
    except FileNotFoundError:
        return False


def _candidate_q(joint1: float, **joint_overrides: float) -> list[float]:
    q = [float(v) for v in TELEOP_INITIAL_Q]
    q[0] = joint1
    for idx, val in joint_overrides.items():
        q[idx] = val
    return q


def _gate_input(
    q_candidate: list[float],
    *,
    fk_ctx=None,
    base_anchor_q: float = 0.0,
    global_config=GATES_ON_GLOBAL,
    anchor_config=GATES_ON_ANCHOR,
    candidate_xyz=FK_BEFORE,
    fk_position_before=FK_BEFORE,
) -> IkGateSequenceInput:
    q_before = np.asarray(TELEOP_INITIAL_Q, dtype=np.float64)
    return IkGateSequenceInput(
        fk_ctx=fk_ctx or MagicMock(),
        q_candidate=list(q_candidate),
        q_before=q_before,
        joint_names=list(JOINT_NAMES),
        lower_limits=list(LOWER),
        upper_limits=list(UPPER),
        base_anchor_q=base_anchor_q,
        candidate_x=candidate_xyz[0],
        candidate_y=candidate_xyz[1],
        candidate_z=candidate_xyz[2],
        fk_position_before=fk_position_before,
        joint1_global_config=global_config,
        joint1_anchor_config=anchor_config,
        joint_limit_config=JOINT_LIMIT_CFG,
        ik_no_effect_config=NO_EFFECT_CFG,
    )


def test_accepts_when_no_gate_fails():
    q = _candidate_q(0.0)
    result = apply_ik_gate_sequence(_gate_input(q))
    assert result.accepted is True
    assert result.rejection_reason == ""
    assert result.gate_name == ""
    assert result.q_candidate == q


def test_global_operational_limit_fails_before_anchor_window():
    q = _candidate_q(2.0)
    result = apply_ik_gate_sequence(_gate_input(q, base_anchor_q=0.0))
    assert result.accepted is False
    assert result.gate_name == "JOINT1_GLOBAL_OPERATIONAL_LIMIT"
    assert result.rejection_source == "JOINT1_GLOBAL_OPERATIONAL_LIMIT"
    assert result.global_cap_info is not None
    assert result.anchor_info is None


def test_anchor_window_fails_before_joint_near_limit():
    q = _candidate_q(1.25, **{4: 1.57})
    result = apply_ik_gate_sequence(_gate_input(q, base_anchor_q=0.0))
    assert result.accepted is False
    assert result.gate_name == "JOINT1_ANCHOR_WINDOW"
    assert result.joint_limit_info is None


def test_joint_near_limit_fails_before_ik_no_effect():
    q = _candidate_q(0.0, **{4: 1.57})
    result = apply_ik_gate_sequence(_gate_input(q))
    assert result.accepted is False
    assert result.gate_name == "JOINT_NEAR_LIMIT"
    assert result.joint_limit_info is not None


@pytest.mark.skipif(not _sdk_available(), reason="reBotArm_control_py not found")
def test_ik_no_effect_fails_only_after_previous_gates_pass():
    fk_ctx = init_fk_context("", "end_link", TELEOP_INITIAL_Q)
    if not fk_ctx.ok:
        pytest.skip("FK init failed")
    q_before = np.asarray(TELEOP_INITIAL_Q, dtype=np.float64)
    pose, _, _ = compute_fk_pose_for_q(fk_ctx, q_before)
    fk_before = (float(pose.position.x), float(pose.position.y), float(pose.position.z))
    candidate = (fk_before[0] + 0.01, fk_before[1], fk_before[2])
    inp = IkGateSequenceInput(
        fk_ctx=fk_ctx,
        q_candidate=[float(v) for v in q_before],
        q_before=q_before,
        joint_names=list(JOINT_NAMES),
        lower_limits=list(LOWER),
        upper_limits=list(UPPER),
        base_anchor_q=float(q_before[0]),
        candidate_x=candidate[0],
        candidate_y=candidate[1],
        candidate_z=candidate[2],
        fk_position_before=fk_before,
        joint1_global_config=GATES_ON_GLOBAL,
        joint1_anchor_config=GATES_ON_ANCHOR,
        joint_limit_config=JOINT_LIMIT_CFG,
        ik_no_effect_config=NO_EFFECT_CFG,
    )
    result = apply_ik_gate_sequence(inp)
    assert result.accepted is False
    assert result.gate_name == "IK_NO_EFFECT"
    assert result.no_effect_metrics is not None


def test_first_failing_gate_wins_when_multiple_would_fail():
    q = _candidate_q(2.0, **{4: 1.57})
    result = apply_ik_gate_sequence(_gate_input(q, base_anchor_q=0.0))
    assert result.gate_name == "JOINT1_GLOBAL_OPERATIONAL_LIMIT"


def test_disabled_joint1_global_cap_does_not_reject():
    q = _candidate_q(2.0)
    result = apply_ik_gate_sequence(
        _gate_input(q, global_config=GATES_OFF_GLOBAL, base_anchor_q=0.0)
    )
    assert result.gate_name == "JOINT1_ANCHOR_WINDOW"


def test_disabled_anchor_hard_gate_does_not_reject():
    q = _candidate_q(1.25)
    result = apply_ik_gate_sequence(
        _gate_input(
            q,
            global_config=GATES_OFF_GLOBAL,
            anchor_config=GATES_OFF_ANCHOR,
            base_anchor_q=0.0,
        )
    )
    assert result.accepted is True
    assert result.q_candidate == q
