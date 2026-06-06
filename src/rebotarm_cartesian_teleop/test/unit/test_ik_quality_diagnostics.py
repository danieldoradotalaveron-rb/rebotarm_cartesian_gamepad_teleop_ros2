"""Unit tests for ik_quality_diagnostics (pure, no ROS node)."""

from __future__ import annotations

import pytest

from rebotarm_cartesian_teleop.ik_quality_diagnostics import (
    IkQualityLogConfig,
    compute_joint_quality_diagnostics,
    format_ik_quality_diagnostics,
    should_log_ik_quality_diagnostics,
    with_log_reasons,
)

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
LOWER = [-2.8, -3.14, -3.14, -1.87, -1.57, -3.14]
UPPER = [2.8, 0.0, 0.0, 1.57, 1.57, 3.14]
INITIAL_Q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
FK_BEFORE = (0.26, 0.0, 0.19)
FK_TARGET = (0.261, 0.0, 0.19)


def _diag(
    q_before,
    q_target,
    *,
    candidate_drift_m=0.0,
    ik_error=0.0,
    candidate_step_m=0.001,
    fk_before=FK_BEFORE,
    fk_target=FK_TARGET,
    joint_limit_near_rad=0.35,
):
    return compute_joint_quality_diagnostics(
        JOINT_NAMES,
        q_before,
        q_target,
        LOWER,
        UPPER,
        INITIAL_Q,
        fk_position_before=fk_before,
        fk_position_target=fk_target,
        candidate_drift_m=candidate_drift_m,
        ik_error=ik_error,
        candidate_step_m=candidate_step_m,
        joint_limit_near_rad=joint_limit_near_rad,
    )


def test_joint_margin_computation():
    q_before = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    q_target = [0.0, -0.5, -0.2, -0.4, 0.8, 0.0]
    diag = _diag(q_before, q_target)

    j2 = diag.joints[1]
    assert j2.margin_to_lower == pytest.approx(-0.5 - (-3.14))
    assert j2.margin_to_upper == pytest.approx(0.0 - (-0.5))
    assert j2.nearest_margin == pytest.approx(0.5)
    assert j2.nearest_side == "upper"

    j5 = diag.joints[4]
    assert j5.margin_to_lower == pytest.approx(0.8 - (-1.57))
    assert j5.margin_to_upper == pytest.approx(1.57 - 0.8)
    assert j5.nearest_margin == pytest.approx(0.77)
    assert j5.nearest_side == "upper"


def test_nearest_limit_detection():
    q_target = [0.0, -2.9, -1.0, -0.4, 0.0, 0.0]
    diag = _diag([0.0] * 6, q_target)
    assert diag.nearest_limit_joint == "joint2"
    assert diag.nearest_limit_margin == pytest.approx(0.24)
    assert diag.nearest_limit_side == "lower"


def test_q_delta_diagnostics():
    q_before = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    q_target = [0.0, -0.2, 0.0, -0.16, 0.0, 0.0]
    diag = _diag(q_before, q_target)

    assert diag.joints[1].q_delta == pytest.approx(-0.2)
    assert diag.joints[1].abs_q_delta == pytest.approx(0.2)
    assert diag.max_abs_q_delta == pytest.approx(0.2)
    assert diag.max_abs_q_delta_joint == "joint2"
    assert diag.q_step_norm == pytest.approx((0.2**2 + 0.16**2) ** 0.5)


def test_all_six_joints_present():
    diag = _diag([0.0] * 6, [0.01, -0.01, -0.01, 0.01, 0.01, 0.01])
    assert len(diag.joints) == 6
    assert [j.name for j in diag.joints] == JOINT_NAMES


def test_joint4_joint5_highlighted():
    diag = _diag([0.0] * 6, [0.0, -0.3, -0.3, -1.0, 1.1, 0.0])
    assert diag.joint4.name == "joint4"
    assert diag.joint5.name == "joint5"
    assert diag.joint4.q_target == pytest.approx(-1.0)
    assert diag.joint5.q_target == pytest.approx(1.1)
    assert diag.joints[3].q_target == diag.joint4.q_target
    assert diag.joints[4].q_target == diag.joint5.q_target


def test_reached_step_and_posture_distance():
    diag = _diag(
        [0.0] * 6,
        [0.1, -0.1, -0.1, 0.0, 0.0, 0.0],
        fk_before=(0.0, 0.0, 0.0),
        fk_target=(0.01, 0.0, 0.0),
        candidate_step_m=0.01,
    )
    assert diag.reached_step_m == pytest.approx(0.01)
    assert diag.candidate_step_m == pytest.approx(0.01)
    assert diag.posture_distance_from_initial_q > 0.0


@pytest.mark.parametrize(
    ("q_before", "q_target", "kwargs", "expected"),
    [
        ([0.0] * 6, [0.0, -2.9, 0.0, 0.0, 0.0, 0.0], {}, True),
        ([0.0] * 6, [0.0, 0.0, 0.0, 0.0, 1.1, 0.0], {}, True),
        ([0.0] * 6, [0.0, 0.0, 0.0, -1.1, 0.0, 0.0], {}, True),
        ([0.0] * 6, [0.0, -0.2, 0.0, 0.0, 0.0, 0.0], {}, True),
        ([0.0] * 6, [0.0] * 6, {"candidate_drift_m": 0.004}, True),
        (
            [0.0] * 6,
            [0.0] * 6,
            {"candidate_step_m": 0.001, "fk_before": FK_BEFORE, "fk_target": FK_BEFORE},
            True,
        ),
        (
            [0.0, -0.5, -0.5, 0.0, 0.0, 0.0],
            [0.0, -0.5, -0.5, 0.0, 0.0, 0.0],
            {"candidate_step_m": 0.0},
            False,
        ),
        ([0.0] * 6, [0.0, -0.3, -0.3, -0.2, 0.0, 0.0], {}, True),
    ],
)
def test_should_log_conditions(q_before, q_target, kwargs, expected):
    cfg = IkQualityLogConfig()
    diag = _diag(q_before, q_target, **kwargs)
    assert should_log_ik_quality_diagnostics(diag, cfg) is expected


def test_should_log_true_on_ik_failure():
    cfg = IkQualityLogConfig()
    diag = _diag(
        [0.0, -0.5, -0.5, 0.0, 0.0, 0.0],
        [0.0, -0.5, -0.5, 0.0, 0.0, 0.0],
        candidate_step_m=0.0,
    )
    assert should_log_ik_quality_diagnostics(diag, cfg) is False
    assert should_log_ik_quality_diagnostics(diag, cfg, ik_failure=True) is True


def test_format_includes_all_joints_and_highlights():
    diag = with_log_reasons(
        _diag([0.0] * 6, [0.0, -0.3, -0.3, -1.0, 1.1, 0.0]),
        IkQualityLogConfig(),
    )
    text = format_ik_quality_diagnostics(diag)
    for name in JOINT_NAMES:
        assert name in text
    assert "highlight joint4" in text
    assert "highlight joint5" in text
    assert "joint2:" in text


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_joint_quality_diagnostics(
            JOINT_NAMES,
            [0.0] * 5,
            [0.0] * 6,
            LOWER,
            UPPER,
            INITIAL_Q,
            fk_position_before=FK_BEFORE,
            fk_position_target=FK_TARGET,
        )
