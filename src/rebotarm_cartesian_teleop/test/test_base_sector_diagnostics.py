"""Stage 1/2a diagnostics: joint1/base sector (no control behavior changes)."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import TELEOP_INITIAL_Q, call_solve_target_ik, default_workspace, make_cmd

from rebotarm_cartesian_teleop.fk_kinematics import compute_fk_pose_for_q, init_fk_context
from rebotarm_cartesian_teleop.ik_quality_diagnostics import (
    IkQualityLogConfig,
    compute_base_sector_diagnostics,
    format_base_sector_diagnostics,
    format_ik_quality_diagnostics,
    resolve_joint1_warning_window_rad,
    should_log_ik_quality_diagnostics,
    with_log_reasons,
)
from rebotarm_cartesian_teleop.jog_core_logic import (
    IkConfig,
    compute_candidate_target,
    update_base_anchor_q_on_deadman_rising,
    update_q_sim_on_ik_success,
)
from rebotarm_cartesian_teleop.sdk_path import ensure_rebot_sdk_in_syspath

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
LOWER = [-2.8, -3.14, -3.14, -1.87, -1.57, -3.14]
UPPER = [2.8, 0.0, 0.0, 1.57, 1.57, 3.14]
WARNING = 0.25
HARD = 1.20
GLOBAL_MIN = -1.60
GLOBAL_MAX = 1.60


def _sdk_available() -> bool:
    try:
        ensure_rebot_sdk_in_syspath()
        return True
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(not _sdk_available(), reason="reBotArm_control_py not found")


def _base_sector(
    *,
    q_before,
    q_target,
    base_anchor_q=0.0,
    q_candidate_from_ik=None,
    ik_accepted=False,
    rejection_source="",
    warning_window_rad=WARNING,
    hard_window_rad=HARD,
    global_cap_min_rad=GLOBAL_MIN,
    global_cap_max_rad=GLOBAL_MAX,
    enable_joint1_anchor_hard_gate=False,
    enable_joint1_global_operational_cap=False,
):
    from rebotarm_cartesian_teleop.ik_quality_diagnostics import compute_joint_quality_diagnostics

    diag = compute_joint_quality_diagnostics(
        JOINT_NAMES,
        q_before,
        q_target,
        LOWER,
        UPPER,
        TELEOP_INITIAL_Q,
        fk_position_before=(0.3, 0.0, 0.2),
        fk_position_target=(0.3, 0.0, 0.2),
    )
    return compute_base_sector_diagnostics(
        joint_names=JOINT_NAMES,
        lower_limits=LOWER,
        upper_limits=UPPER,
        q_before=q_before,
        q_target_or_current=q_target,
        base_anchor_q=base_anchor_q,
        warning_window_rad=warning_window_rad,
        hard_window_rad=hard_window_rad,
        global_cap_min_rad=global_cap_min_rad,
        global_cap_max_rad=global_cap_max_rad,
        enable_joint1_anchor_hard_gate=enable_joint1_anchor_hard_gate,
        enable_joint1_global_operational_cap=enable_joint1_global_operational_cap,
        command_frame="base_link",
        workspace_frame="base_link",
        cartesian_command_linear_x=0.01,
        cartesian_command_linear_y=0.0,
        cartesian_command_linear_z=0.0,
        raw_candidate_position=(0.31, 0.0, 0.2),
        clamped_candidate_position=(0.31, 0.0, 0.2),
        workspace_clamp_active=False,
        workspace_clamped_axes=(),
        ik_accepted=ik_accepted,
        rejection_source=rejection_source,
        nearest_limit_joint=diag.nearest_limit_joint,
        nearest_limit_margin=diag.nearest_limit_margin,
        posture_distance_from_initial_q=diag.posture_distance_from_initial_q,
        q_candidate_from_ik=q_candidate_from_ik,
        accepted_fk_position=(0.31, 0.0, 0.2) if ik_accepted else None,
    )


def test_legacy_window_rad_alias():
    assert resolve_joint1_warning_window_rad(warning_rad=0.25, legacy_window_rad=0.30) == 0.30
    assert resolve_joint1_warning_window_rad(warning_rad=0.30, legacy_window_rad=0.25) == 0.30


def test_anchor_resets_on_deadman_rising_edge_only():
    anchor = 0.0
    prev = False
    for deadman in [False, False, True, True, False, True]:
        anchor, prev = update_base_anchor_q_on_deadman_rising(
            deadman_pressed=deadman,
            prev_deadman_pressed=prev,
            joint1_q=-0.4,
            base_anchor_q=anchor,
        )
    assert anchor == pytest.approx(-0.4)


def test_warning_at_exactly_025_does_not_violate():
    sector = _base_sector(
        q_before=[0.0, -0.3, -0.3, 0.0, 0.0, 0.0],
        q_target=[0.0, -0.3, -0.3, 0.0, 0.0, 0.0],
        base_anchor_q=0.0,
        q_candidate_from_ik=[0.25, -0.3, -0.3, 0.0, 0.0, 0.0],
    )
    assert sector is not None
    assert sector.joint1_would_violate_warning_window is False
    assert sector.joint1_warning_window_error_rad == pytest.approx(0.0)


def test_warning_violation_diagnostic_only():
    sector = _base_sector(
        q_before=[0.0, -0.3, -0.3, 0.0, 0.0, 0.0],
        q_target=[0.0, -0.3, -0.3, 0.0, 0.0, 0.0],
        base_anchor_q=0.0,
        q_candidate_from_ik=[0.26, -0.3, -0.3, 0.0, 0.0, 0.0],
        enable_joint1_anchor_hard_gate=False,
    )
    assert sector is not None
    assert sector.joint1_would_violate_warning_window is True
    assert sector.enable_joint1_anchor_hard_gate is False


def test_delta_109_does_not_violate_hard_window():
    sector = _base_sector(
        q_before=[0.0, -0.3, -0.3, 0.0, 0.0, 0.0],
        q_target=[0.0, -0.3, -0.3, 0.0, 0.0, 0.0],
        base_anchor_q=0.0,
        q_candidate_from_ik=[1.09, -0.3, -0.3, 0.0, 0.0, 0.0],
    )
    assert sector is not None
    assert sector.joint1_would_violate_hard_window is False
    assert sector.joint1_hard_window_error_rad == pytest.approx(0.0)


def test_delta_above_hard_violates_hard_diagnostic_only():
    sector = _base_sector(
        q_before=[0.0, -0.3, -0.3, 0.0, 0.0, 0.0],
        q_target=[0.0, -0.3, -0.3, 0.0, 0.0, 0.0],
        base_anchor_q=0.0,
        q_candidate_from_ik=[1.25, -0.3, -0.3, 0.0, 0.0, 0.0],
        enable_joint1_anchor_hard_gate=False,
    )
    assert sector is not None
    assert sector.joint1_would_violate_hard_window is True
    assert sector.joint1_hard_window_error_rad == pytest.approx(0.05)
    assert sector.enable_joint1_anchor_hard_gate is False


def test_joint1_minus_20_violates_global_cap_diagnostic_only():
    sector = _base_sector(
        q_before=[0.0, -0.3, -0.3, 0.0, 0.0, 0.0],
        q_target=[0.0, -0.3, -0.3, 0.0, 0.0, 0.0],
        q_candidate_from_ik=[-2.0, -0.3, -0.3, 0.0, 0.0, 0.0],
        enable_joint1_global_operational_cap=False,
    )
    assert sector is not None
    assert sector.joint1_would_violate_global_cap is True
    assert sector.joint1_global_cap_error_rad == pytest.approx(0.4)
    assert sector.enable_joint1_global_operational_cap is False


def test_global_cap_within_range_is_clean():
    sector = _base_sector(
        q_before=[0.0, -0.3, -0.3, 0.0, 0.0, 0.0],
        q_target=[0.0, -0.3, -0.3, 0.0, 0.0, 0.0],
        q_candidate_from_ik=[-0.7, -0.3, -0.3, 0.0, 0.0, 0.0],
    )
    assert sector is not None
    assert sector.joint1_would_violate_global_cap is False
    assert sector.joint1_global_cap_error_rad == pytest.approx(0.0)


def test_diagnostics_disabled_do_not_add_window_log_triggers():
    from rebotarm_cartesian_teleop.ik_quality_diagnostics import (
        _collect_base_sector_log_reasons,
        compute_joint_quality_diagnostics,
    )

    q_safe = list(TELEOP_INITIAL_Q)
    diag = compute_joint_quality_diagnostics(
        JOINT_NAMES,
        q_safe,
        q_safe,
        LOWER,
        UPPER,
        TELEOP_INITIAL_Q,
        fk_position_before=(0.3, 0.0, 0.2),
        fk_position_target=(0.3, 0.0, 0.2),
    )
    sector = _base_sector(
        q_before=q_safe,
        q_target=q_safe,
        q_candidate_from_ik=[-2.0, -0.3, -0.3, 0.0, 0.0, 0.0],
    )
    cfg_off = IkQualityLogConfig(enable_cartesian_joint1_window_diagnostics=False)
    cfg_on = IkQualityLogConfig(enable_cartesian_joint1_window_diagnostics=True)
    assert _collect_base_sector_log_reasons(sector, cfg_off) == ()
    assert any("warning_window" in r for r in _collect_base_sector_log_reasons(sector, cfg_on))
    assert any("global_cap" in r for r in _collect_base_sector_log_reasons(sector, cfg_on))
    assert should_log_ik_quality_diagnostics(diag, cfg_on, base_sector=sector) is True


def test_format_includes_dual_threshold_fields():
    sector = _base_sector(
        q_before=[0.0, -0.3, -0.3, 0.0, 0.0, 0.0],
        q_target=[0.1, -0.3, -0.3, 0.0, 0.0, 0.0],
        q_candidate_from_ik=[0.1, -0.3, -0.3, 0.0, 0.0, 0.0],
        ik_accepted=True,
    )
    from rebotarm_cartesian_teleop.ik_quality_diagnostics import compute_joint_quality_diagnostics

    diag = with_log_reasons(
        compute_joint_quality_diagnostics(
            JOINT_NAMES,
            [0.0, -0.3, -0.3, 0.0, 0.0, 0.0],
            [0.1, -0.3, -0.3, 0.0, 0.0, 0.0],
            LOWER,
            UPPER,
            TELEOP_INITIAL_Q,
            fk_position_before=(0.3, 0.0, 0.2),
            fk_position_target=(0.31, 0.0, 0.2),
        ),
        IkQualityLogConfig(),
        base_sector=sector,
    )
    text = format_ik_quality_diagnostics(diag, base_sector=sector)
    assert "joint1_warning_window:" in text
    assert "joint1_hard_window:" in text
    assert "joint1_global_cap:" in text
    assert "gate_enabled=False" in text
    assert format_base_sector_diagnostics(sector) in text


@pytest.fixture
def fk_ctx():
    ctx = init_fk_context("", "end_link", TELEOP_INITIAL_Q)
    if not ctx.ok:
        pytest.skip("FK init failed")
    return ctx


def test_ik_decisions_unchanged_with_diagnostics_config(fk_ctx):
    """Diagnostics params must not alter IK accept/reject outcomes."""
    q_sim = np.asarray(TELEOP_INITIAL_Q, dtype=np.float64)
    pose, sim_rot, _ = compute_fk_pose_for_q(fk_ctx, q_sim)
    sim_x, sim_y, sim_z = (
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
    )
    cand_x, cand_y, cand_z, _ = compute_candidate_target(
        sim_x,
        sim_y,
        sim_z,
        make_cmd(linear_x=0.03),
        0.02,
        default_workspace(),
    )
    ik_config = IkConfig(
        max_iterations=100,
        tolerance=1e-4,
        max_ik_error=0.005,
        max_joint_delta_rad=0.25,
        task_mode="position_only",
    )
    q_target, ok, reason, _ = call_solve_target_ik(
        fk_ctx,
        pose,
        q_sim,
        target_x=cand_x,
        target_y=cand_y,
        target_z=cand_z,
        ik_config=ik_config,
    )
    assert ok is True
    assert reason == ""
    assert q_target
    q_after = update_q_sim_on_ik_success(q_sim, q_target, True)
    assert not np.allclose(q_after, q_sim)
    assert np.allclose(q_after, q_target)
