import math

import numpy as np
import rclpy
from rclpy.node import Node
from rebotarm_msgs.msg import CartesianJogCmd, CartesianJogState
from sensor_msgs.msg import JointState

from .fake_joint_state import build_fake_joint_state
from .fk_kinematics import (
    FkContext,
    compute_fk_pose_for_q,
    init_fk_context,
)
from .ik_kinematics import compute_ik_for_pose, compute_ik_for_position
from .ik_quality_diagnostics import (
    IkQualityLogConfig,
    LocalWindowDiagnostics,
    LocalWindowLimitsView,
    _global_cap_error_rad,
    _window_error_rad,
    candidate_step_m,
    compute_base_sector_diagnostics,
    compute_joint_quality_diagnostics,
    format_ik_quality_diagnostics,
    joint1_index,
    joint_limits_from_model,
    joint_names_from_model,
    pos3_from_pose,
    resolve_joint1_warning_window_rad,
    should_log_ik_quality_diagnostics,
    with_log_reasons,
)
from .jog_core_logic import (
    COMMAND_FRAME_LOCAL_WINDOW,
    BaseJogResult,
    IkConfig,
    IkGateSequenceInput,
    IkGateSequenceResult,
    IkNoEffectConfig,
    Joint1AnchorWindowConfig,
    Joint1GlobalOperationalLimitConfig,
    JointLimitRejectConfig,
    LocalWindowLimits,
    LocalWindowState,
    WorkspaceLimits,
    apply_base_joint1_jog,
    apply_ik_gate_sequence,
    build_cartesian_jog_state,
    build_committed_target_pose,
    compute_candidate_drift_m,
    compute_candidate_target,
    compute_local_axes,
    compute_state_name,
    format_ik_failure_log,
    format_joint1_anchor_window_log,
    format_joint1_global_operational_limit_log,
    format_joint_near_limit_log,
    integrate_local_target_offset,
    integrate_local_window_candidate,
    parse_ik_task_mode,
    reanchor_local_window_from_fk,
    resync_committed_from_q_sim,
    solve_target_ik,
    update_q_sim_on_ik_success,
)


class CartesianJogCore(Node):
    def __init__(self):
        super().__init__("cartesian_jog_core")

        self.declare_parameter("cartesian_jog_cmd_topic", "/rebotarm/cartesian_jog_cmd")
        self.declare_parameter("cartesian_jog_state_topic", "/rebotarm/cartesian_jog_state")
        self.declare_parameter("output_mode", "dry_run")
        self.declare_parameter("dry_run", True)
        self.declare_parameter("command_timeout_s", 0.3)
        self.declare_parameter("servo_hz", 50.0)

        self.declare_parameter("initial_x", 0.30)
        self.declare_parameter("initial_y", 0.00)
        self.declare_parameter("initial_z", 0.20)

        self.declare_parameter("workspace_x_min", 0.150)
        self.declare_parameter("workspace_x_max", 0.450)
        self.declare_parameter("workspace_y_min", -0.250)
        self.declare_parameter("workspace_y_max", 0.250)
        self.declare_parameter("workspace_z_min", 0.020)
        self.declare_parameter("workspace_z_max", 0.450)
        self.declare_parameter("enable_local_teleop_window", True)
        self.declare_parameter("enable_base_joint_jog", True)
        self.declare_parameter("local_window_x_min_m", -0.12)
        self.declare_parameter("local_window_x_max_m", 0.18)
        self.declare_parameter("local_window_y_min_m", -0.25)
        self.declare_parameter("local_window_y_max_m", 0.25)
        self.declare_parameter("local_window_z_min_m", -0.25)
        self.declare_parameter("local_window_z_max_m", 0.18)
        self.declare_parameter("global_z_min_m", 0.020)
        self.declare_parameter("global_z_max_m", 0.450)
        self.declare_parameter("base_joint_jog_speed_rad_s", 0.5)

        self.declare_parameter("urdf_path", "")
        self.declare_parameter("ee_frame", "end_link")
        self.declare_parameter("initial_q", [0.0, -0.3, -0.3, 0.0, 0.0, 0.0])

        self.declare_parameter("ik_max_iterations", 100)
        self.declare_parameter("ik_tolerance", 0.001)
        self.declare_parameter("ik_task_mode", "position_only")
        self.declare_parameter("max_ik_error", 0.005)
        self.declare_parameter("max_joint_delta_rad", 0.25)
        self.declare_parameter("ik_failure_log_interval_s", 1.0)
        self.declare_parameter("candidate_drift_log_threshold_m", 0.001)
        self.declare_parameter("joint_limit_warn_margin_rad", 0.35)
        self.declare_parameter("joint_limit_reject_margin_rad", 0.05)
        self.declare_parameter("joint5_warn_abs_rad", 1.0)
        self.declare_parameter("joint4_warn_abs_rad", 1.0)
        self.declare_parameter("q_delta_warn_rad", 0.15)
        self.declare_parameter("candidate_drift_warn_m", 0.003)
        self.declare_parameter("reached_step_warn_min_m", 0.0001)
        self.declare_parameter("ik_quality_log_interval_s", 1.0)
        self.declare_parameter("enable_cartesian_joint1_window_diagnostics", True)
        self.declare_parameter("cartesian_joint1_window_warning_rad", 0.25)
        self.declare_parameter("cartesian_joint1_window_hard_rad", 1.20)
        self.declare_parameter("cartesian_joint1_window_rad", 0.25)
        self.declare_parameter("enable_joint1_anchor_hard_gate", True)
        self.declare_parameter("enable_joint1_global_operational_cap", True)
        self.declare_parameter("joint1_global_operational_min_rad", -1.60)
        self.declare_parameter("joint1_global_operational_max_rad", 1.60)
        self.declare_parameter("joint1_large_delta_from_anchor_rad", 0.15)
        self.declare_parameter("ik_no_effect_candidate_step_min_m", 0.0005)
        self.declare_parameter("ik_no_effect_reached_step_min_m", 0.0001)
        self.declare_parameter("ik_no_effect_q_step_min_norm", 1.0e-6)
        self.declare_parameter("publish_fake_joint_states", True)
        self.declare_parameter("fake_joint_states_topic", "/rebotarm/fake_joint_states")
        self.declare_parameter("fake_joint_state_hz", 50.0)

        cmd_topic = self.get_parameter("cartesian_jog_cmd_topic").value
        state_topic = self.get_parameter("cartesian_jog_state_topic").value

        self.output_mode = self.get_parameter("output_mode").value
        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.command_timeout_s = float(self.get_parameter("command_timeout_s").value)
        self.servo_hz = float(self.get_parameter("servo_hz").value)

        self._workspace = WorkspaceLimits(
            x_min=float(self.get_parameter("workspace_x_min").value),
            x_max=float(self.get_parameter("workspace_x_max").value),
            y_min=float(self.get_parameter("workspace_y_min").value),
            y_max=float(self.get_parameter("workspace_y_max").value),
            z_min=float(self.get_parameter("workspace_z_min").value),
            z_max=float(self.get_parameter("workspace_z_max").value),
        )
        self._enable_local_teleop_window = bool(
            self.get_parameter("enable_local_teleop_window").value
        )
        self._enable_base_joint_jog = bool(self.get_parameter("enable_base_joint_jog").value)
        self._base_joint_jog_speed_rad_s = float(
            self.get_parameter("base_joint_jog_speed_rad_s").value
        )
        self._local_window_limits = LocalWindowLimits(
            x_min=float(self.get_parameter("local_window_x_min_m").value),
            x_max=float(self.get_parameter("local_window_x_max_m").value),
            y_min=float(self.get_parameter("local_window_y_min_m").value),
            y_max=float(self.get_parameter("local_window_y_max_m").value),
            z_min=float(self.get_parameter("local_window_z_min_m").value),
            z_max=float(self.get_parameter("local_window_z_max_m").value),
            global_z_min=float(self.get_parameter("global_z_min_m").value),
            global_z_max=float(self.get_parameter("global_z_max_m").value),
        )
        self._teleop_mode = "cartesian"
        self._last_base_jog_result: BaseJogResult | None = None

        urdf_path = str(self.get_parameter("urdf_path").value)
        ee_frame = str(self.get_parameter("ee_frame").value)
        initial_q = [float(v) for v in self.get_parameter("initial_q").value]
        self._initial_q = np.asarray(initial_q, dtype=np.float64)

        self._fk: FkContext = init_fk_context(urdf_path, ee_frame, initial_q)
        if self._fk.ok:
            self.get_logger().info(
                f"FK model loaded (nq={self._fk.model.nq}, frame={self._fk.ee_frame})"
            )
            if self._fk.q_current is not None:
                q_str = ", ".join(f"{v:.4f}" for v in self._fk.q_current)
                self.get_logger().info(f"initial_q: [{q_str}]")
        else:
            self.get_logger().error(f"FK init failed: {self._fk.error}")

        if self._fk.ok and self._fk.q_current is not None:
            self._q_sim = np.asarray(self._fk.q_current, dtype=np.float64).copy()
        else:
            self._q_sim = np.zeros(len(initial_q), dtype=np.float64)

        (
            self.committed_target_x,
            self.committed_target_y,
            self.committed_target_z,
            self._committed_rotation,
            _,
            fk_init_err,
        ) = resync_committed_from_q_sim(self._fk, self._q_sim)
        if fk_init_err:
            fallback_x = float(self.get_parameter("initial_x").value)
            fallback_y = float(self.get_parameter("initial_y").value)
            fallback_z = float(self.get_parameter("initial_z").value)
            self.committed_target_x = fallback_x
            self.committed_target_y = fallback_y
            self.committed_target_z = fallback_z
            self.get_logger().warn(
                f"FK(q_sim) init failed ({fk_init_err}); using YAML fallback target"
            )
        else:
            self.get_logger().info(
                "Initial committed target from FK(q_sim): "
                f"x={self.committed_target_x:.3f}, y={self.committed_target_y:.3f}, "
                f"z={self.committed_target_z:.3f}"
            )

        self.latest_cmd = None
        self.latest_cmd_time_ns = None
        self.last_tick_time_ns = self.get_clock().now().nanoseconds
        self.last_clamp_reason = ""
        self._fk_tick_error = ""
        self._ik_failure_log_interval_s = float(
            self.get_parameter("ik_failure_log_interval_s").value
        )
        self._candidate_drift_log_threshold_m = float(
            self.get_parameter("candidate_drift_log_threshold_m").value
        )
        self._ik_quality_log_config = IkQualityLogConfig(
            joint_limit_warn_margin_rad=float(
                self.get_parameter("joint_limit_warn_margin_rad").value
            ),
            joint5_warn_abs_rad=float(self.get_parameter("joint5_warn_abs_rad").value),
            joint4_warn_abs_rad=float(self.get_parameter("joint4_warn_abs_rad").value),
            q_delta_warn_rad=float(self.get_parameter("q_delta_warn_rad").value),
            candidate_drift_warn_m=float(self.get_parameter("candidate_drift_warn_m").value),
            reached_step_warn_min_m=float(self.get_parameter("reached_step_warn_min_m").value),
            enable_cartesian_joint1_window_diagnostics=bool(
                self.get_parameter("enable_cartesian_joint1_window_diagnostics").value
            ),
            cartesian_joint1_window_warning_rad=resolve_joint1_warning_window_rad(
                warning_rad=float(
                    self.get_parameter("cartesian_joint1_window_warning_rad").value
                ),
                legacy_window_rad=float(
                    self.get_parameter("cartesian_joint1_window_rad").value
                ),
            ),
            cartesian_joint1_window_hard_rad=float(
                self.get_parameter("cartesian_joint1_window_hard_rad").value
            ),
            enable_joint1_anchor_hard_gate=bool(
                self.get_parameter("enable_joint1_anchor_hard_gate").value
            ),
            enable_joint1_global_operational_cap=bool(
                self.get_parameter("enable_joint1_global_operational_cap").value
            ),
            joint1_global_operational_min_rad=float(
                self.get_parameter("joint1_global_operational_min_rad").value
            ),
            joint1_global_operational_max_rad=float(
                self.get_parameter("joint1_global_operational_max_rad").value
            ),
            joint1_large_delta_from_anchor_rad=float(
                self.get_parameter("joint1_large_delta_from_anchor_rad").value
            ),
        )
        self._ik_quality_log_interval_s = float(
            self.get_parameter("ik_quality_log_interval_s").value
        )
        self._ik_no_effect_config = IkNoEffectConfig(
            candidate_step_min_m=float(
                self.get_parameter("ik_no_effect_candidate_step_min_m").value
            ),
            reached_step_min_m=float(
                self.get_parameter("ik_no_effect_reached_step_min_m").value
            ),
            q_step_min_norm=float(self.get_parameter("ik_no_effect_q_step_min_norm").value),
        )
        self._joint_limit_reject_config = JointLimitRejectConfig(
            reject_margin_rad=float(self.get_parameter("joint_limit_reject_margin_rad").value),
        )
        self._joint1_global_cap_config = Joint1GlobalOperationalLimitConfig(
            enabled=self._ik_quality_log_config.enable_joint1_global_operational_cap,
            min_rad=self._ik_quality_log_config.joint1_global_operational_min_rad,
            max_rad=self._ik_quality_log_config.joint1_global_operational_max_rad,
        )
        self._joint1_anchor_window_config = Joint1AnchorWindowConfig(
            enabled=self._ik_quality_log_config.enable_joint1_anchor_hard_gate,
            hard_window_rad=self._ik_quality_log_config.cartesian_joint1_window_hard_rad,
        )
        self._last_ik_failure_log_ns = 0
        self._last_ik_quality_log_ns = 0
        self._prev_deadman_pressed = False

        self._joint_names: list[str] = []
        self._joint_lower_limits: list[float] = []
        self._joint_upper_limits: list[float] = []
        if self._fk.ok and self._fk.model is not None:
            self._joint_names = joint_names_from_model(self._fk.model)
            lo, hi = joint_limits_from_model(self._fk.model)
            self._joint_lower_limits = lo
            self._joint_upper_limits = hi

        self._joint1_idx = joint1_index(self._joint_names)
        self._base_anchor_q = (
            float(self._q_sim[self._joint1_idx]) if self._joint1_idx is not None else 0.0
        )
        self._local_window_state = LocalWindowState(
            base_anchor_q=self._base_anchor_q,
            anchor_position=(
                self.committed_target_x,
                self.committed_target_y,
                self.committed_target_z,
            ),
            local_offset=(0.0, 0.0, 0.0),
        )
        self._reanchor_local_window_from_q_sim()

        self._publish_fake_joint_states = bool(
            self.get_parameter("publish_fake_joint_states").value
        )
        fake_joint_states_topic = str(self.get_parameter("fake_joint_states_topic").value)
        self._fake_joint_state_hz = float(self.get_parameter("fake_joint_state_hz").value)
        self._last_valid_fake_q: list[float] = [float(v) for v in self._q_sim]

        ik_task_mode = parse_ik_task_mode(str(self.get_parameter("ik_task_mode").value))
        self._ik_config = IkConfig(
            max_iterations=int(self.get_parameter("ik_max_iterations").value),
            tolerance=float(self.get_parameter("ik_tolerance").value),
            max_ik_error=float(self.get_parameter("max_ik_error").value),
            max_joint_delta_rad=float(self.get_parameter("max_joint_delta_rad").value),
            task_mode=ik_task_mode,
        )

        self.subscription = self.create_subscription(
            CartesianJogCmd,
            cmd_topic,
            self.on_cmd,
            10,
        )

        self.publisher = self.create_publisher(
            CartesianJogState,
            state_topic,
            10,
        )

        self._fake_joint_states_publisher = None
        self._fake_joint_states_timer = None
        if self._publish_fake_joint_states:
            self._fake_joint_states_publisher = self.create_publisher(
                JointState,
                fake_joint_states_topic,
                10,
            )
            self._fake_joint_states_timer = self.create_timer(
                1.0 / self._fake_joint_state_hz,
                self._publish_fake_joint_state,
            )

        self.timer = self.create_timer(1.0 / self.servo_hz, self.tick)

        self.get_logger().info("cartesian_jog_core started")
        self.get_logger().info(f"Listening to: {cmd_topic}")
        self.get_logger().info(f"Publishing to: {state_topic}")
        self.get_logger().info(f"Output mode: {self.output_mode}")
        self.get_logger().info(f"Dry run: {self.dry_run}")
        self.get_logger().info(f"IK task mode: {self._ik_config.task_mode}")
        if self._publish_fake_joint_states:
            self.get_logger().info(
                f"Publishing fake joint states to: {fake_joint_states_topic} "
                f"at {self._fake_joint_state_hz:.1f} Hz"
            )

    def on_cmd(self, msg: CartesianJogCmd):
        self.latest_cmd = msg
        self.latest_cmd_time_ns = self.get_clock().now().nanoseconds

    def get_command_age(self) -> float:
        if self.latest_cmd_time_ns is None:
            return math.inf

        now_ns = self.get_clock().now().nanoseconds
        return (now_ns - self.latest_cmd_time_ns) / 1e9

    def _fk_error_reason(self) -> str:
        if not self._fk.ok:
            return self._fk.error
        return self._fk_tick_error

    def _pose_from_q_sim(self):
        fk_error = self._fk_error_reason()
        if fk_error:
            return None, None, fk_error

        pose, rotation, tick_error = compute_fk_pose_for_q(self._fk, self._q_sim)
        if tick_error:
            self._fk_tick_error = tick_error
            return None, None, tick_error
        self._fk_tick_error = ""
        return pose, rotation, ""

    def _maybe_log_ik_failure(self, diag, now_ns: int) -> None:
        if diag is None:
            return
        interval_ns = int(self._ik_failure_log_interval_s * 1e9)
        if now_ns - self._last_ik_failure_log_ns < interval_ns:
            return
        self._last_ik_failure_log_ns = now_ns
        self.get_logger().warn(format_ik_failure_log(diag))

    def _maybe_log_ik_rejection(self, message: str, now_ns: int) -> None:
        interval_ns = int(self._ik_failure_log_interval_s * 1e9)
        if now_ns - self._last_ik_failure_log_ns < interval_ns:
            return
        self._last_ik_failure_log_ns = now_ns
        self.get_logger().warn(message)

    def _maybe_log_joint_near_limit(self, info, now_ns: int) -> None:
        if info is None:
            return
        self._maybe_log_ik_rejection(format_joint_near_limit_log(info), now_ns)

    def _reanchor_local_window_from_q_sim(self) -> None:
        pose, rotation, err = self._pose_from_q_sim()
        if err or pose is None or rotation is None:
            return
        j1_q = (
            float(self._q_sim[self._joint1_idx])
            if self._joint1_idx is not None
            else 0.0
        )
        pos = pos3_from_pose(pose)
        self._local_window_state = reanchor_local_window_from_fk(
            fk_position=pos,
            joint1_q=j1_q,
        )
        self._base_anchor_q = self._local_window_state.base_anchor_q
        (
            self.committed_target_x,
            self.committed_target_y,
            self.committed_target_z,
            self._committed_rotation,
            _,
            fk_err,
        ) = resync_committed_from_q_sim(self._fk, self._q_sim)
        if fk_err:
            self._fk_tick_error = fk_err

    def _maybe_reanchor_on_deadman_rising(self, deadman_pressed: bool) -> None:
        if deadman_pressed and not self._prev_deadman_pressed:
            self._reanchor_local_window_from_q_sim()
        self._prev_deadman_pressed = deadman_pressed

    def _local_window_limits_view(self) -> LocalWindowLimitsView:
        lim = self._local_window_limits
        return LocalWindowLimitsView(
            x_min=lim.x_min,
            x_max=lim.x_max,
            y_min=lim.y_min,
            y_max=lim.y_max,
            z_min=lim.z_min,
            z_max=lim.z_max,
            global_z_min=lim.global_z_min,
            global_z_max=lim.global_z_max,
        )

    def _build_local_window_diagnostics(
        self,
        *,
        target_before_ik: tuple[float, float, float],
        clamp_active: bool,
        clamped_axes: tuple[str, ...],
        joint1_candidate_q: float | None,
        cartesian_skipped: bool,
        base_jog_result: BaseJogResult | None,
    ) -> LocalWindowDiagnostics:
        cmd = self.latest_cmd
        j1_cur = (
            float(self._q_sim[self._joint1_idx])
            if self._joint1_idx is not None
            else 0.0
        )
        j1_cand = float(joint1_candidate_q) if joint1_candidate_q is not None else j1_cur
        anchor = self._local_window_state.anchor_position
        offset = self._local_window_state.local_offset
        target = target_before_ik
        frame_kind = (
            cmd.command_frame_kind
            if cmd is not None and cmd.command_frame_kind
            else COMMAND_FRAME_LOCAL_WINDOW
        )
        hard_rad = self._ik_quality_log_config.cartesian_joint1_window_hard_rad
        cap_min = self._ik_quality_log_config.joint1_global_operational_min_rad
        cap_max = self._ik_quality_log_config.joint1_global_operational_max_rad
        delta = j1_cand - self._base_anchor_q
        return LocalWindowDiagnostics(
            teleop_mode=self._teleop_mode,
            command_frame_kind=frame_kind,
            base_jog_active=bool(cmd.base_jog_active) if cmd is not None else False,
            base_jog_delta_rad=(
                base_jog_result.delta_rad if base_jog_result is not None else 0.0
            ),
            base_jog_before_q=(
                base_jog_result.before_q if base_jog_result is not None else j1_cur
            ),
            base_jog_after_q=(
                base_jog_result.after_q if base_jog_result is not None else j1_cur
            ),
            cartesian_mode_skipped_due_to_base_jog=cartesian_skipped,
            base_anchor_q=self._base_anchor_q,
            local_window_anchor_position=anchor,
            local_target_offset=offset,
            local_window_limits=self._local_window_limits_view(),
            local_window_clamp_active=clamp_active,
            local_window_clamped_axes=clamped_axes,
            target_base_link_from_local_window=target,
            target_base_link_before_ik=target_before_ik,
            joint1_current_q=j1_cur,
            joint1_candidate_q=j1_cand,
            joint1_delta_from_anchor=delta,
            joint1_would_violate_global_cap=(
                j1_cand < cap_min or j1_cand > cap_max
            ),
            joint1_global_cap_error_rad=_global_cap_error_rad(
                j1_cand, min_rad=cap_min, max_rad=cap_max
            ),
            joint1_would_violate_hard_window=abs(delta) > hard_rad,
            joint1_hard_window_error_rad=_window_error_rad(abs(delta), hard_rad),
        )

    def _build_base_sector_diagnostics(
        self,
        *,
        diag,
        q_before: np.ndarray,
        q_target: np.ndarray,
        q_candidate_from_ik: list[float] | None,
        ik_accepted: bool,
        rejection_source: str,
        raw_candidate_position: tuple[float, float, float],
        clamped_candidate_position: tuple[float, float, float],
        workspace_clamp_active: bool,
        workspace_clamped_axes: tuple[str, ...],
        accepted_fk_position: tuple[float, float, float] | None,
    ):
        if not self._ik_quality_log_config.enable_cartesian_joint1_window_diagnostics:
            return None
        cmd = self.latest_cmd
        return compute_base_sector_diagnostics(
            joint_names=self._joint_names,
            lower_limits=self._joint_lower_limits,
            upper_limits=self._joint_upper_limits,
            q_before=q_before,
            q_target_or_current=q_target,
            base_anchor_q=self._base_anchor_q,
            warning_window_rad=self._ik_quality_log_config.cartesian_joint1_window_warning_rad,
            hard_window_rad=self._ik_quality_log_config.cartesian_joint1_window_hard_rad,
            global_cap_min_rad=self._ik_quality_log_config.joint1_global_operational_min_rad,
            global_cap_max_rad=self._ik_quality_log_config.joint1_global_operational_max_rad,
            enable_joint1_anchor_hard_gate=(
                self._ik_quality_log_config.enable_joint1_anchor_hard_gate
            ),
            enable_joint1_global_operational_cap=(
                self._ik_quality_log_config.enable_joint1_global_operational_cap
            ),
            command_frame="base_link",
            workspace_frame="base_link",
            cartesian_command_linear_x=float(cmd.linear.x) if cmd is not None else 0.0,
            cartesian_command_linear_y=float(cmd.linear.y) if cmd is not None else 0.0,
            cartesian_command_linear_z=float(cmd.linear.z) if cmd is not None else 0.0,
            raw_candidate_position=raw_candidate_position,
            clamped_candidate_position=clamped_candidate_position,
            workspace_clamp_active=workspace_clamp_active,
            workspace_clamped_axes=workspace_clamped_axes,
            ik_accepted=ik_accepted,
            rejection_source=rejection_source,
            nearest_limit_joint=diag.nearest_limit_joint,
            nearest_limit_margin=diag.nearest_limit_margin,
            posture_distance_from_initial_q=diag.posture_distance_from_initial_q,
            q_candidate_from_ik=q_candidate_from_ik,
            accepted_fk_position=accepted_fk_position,
            large_delta_threshold_rad=(
                self._ik_quality_log_config.joint1_large_delta_from_anchor_rad
            ),
        )

    def _maybe_log_ik_quality(
        self,
        *,
        q_before: np.ndarray,
        q_target: np.ndarray,
        fk_position_before: tuple[float, float, float],
        fk_position_target: tuple[float, float, float],
        candidate_x: float,
        candidate_y: float,
        candidate_z: float,
        candidate_drift_m: float,
        now_ns: int,
        ik_failure: bool = False,
        resolve_ik_error=None,
        q_candidate_from_ik: list[float] | None = None,
        rejection_source: str = "",
        ik_accepted: bool = False,
        raw_candidate_position: tuple[float, float, float] | None = None,
        workspace_clamp_active: bool = False,
        workspace_clamped_axes: tuple[str, ...] = (),
        accepted_fk_position: tuple[float, float, float] | None = None,
        local_window: LocalWindowDiagnostics | None = None,
    ) -> None:
        if not self._joint_names:
            return

        clamped_candidate_position = (candidate_x, candidate_y, candidate_z)
        raw_pos = raw_candidate_position or clamped_candidate_position

        cand_step = candidate_step_m(
            fk_position_before,
            clamped_candidate_position,
        )
        diag = compute_joint_quality_diagnostics(
            self._joint_names,
            q_before,
            q_target,
            self._joint_lower_limits,
            self._joint_upper_limits,
            self._initial_q,
            fk_position_before=fk_position_before,
            fk_position_target=fk_position_target,
            candidate_drift_m=candidate_drift_m,
            ik_error=0.0,
            candidate_step_m=cand_step,
            joint_limit_near_rad=self._ik_quality_log_config.joint_limit_warn_margin_rad,
        )
        base_sector = self._build_base_sector_diagnostics(
            diag=diag,
            q_before=q_before,
            q_target=q_target,
            q_candidate_from_ik=q_candidate_from_ik,
            ik_accepted=ik_accepted,
            rejection_source=rejection_source,
            raw_candidate_position=raw_pos,
            clamped_candidate_position=clamped_candidate_position,
            workspace_clamp_active=workspace_clamp_active,
            workspace_clamped_axes=workspace_clamped_axes,
            accepted_fk_position=accepted_fk_position,
        )
        if not should_log_ik_quality_diagnostics(
            diag,
            self._ik_quality_log_config,
            ik_failure=ik_failure,
            base_sector=base_sector,
        ):
            return

        interval_ns = int(self._ik_quality_log_interval_s * 1e9)
        if now_ns - self._last_ik_quality_log_ns < interval_ns:
            return
        self._last_ik_quality_log_ns = now_ns

        ik_error = 0.0
        if resolve_ik_error is not None:
            ik_error = float(resolve_ik_error())
        diag = compute_joint_quality_diagnostics(
            self._joint_names,
            q_before,
            q_target,
            self._joint_lower_limits,
            self._joint_upper_limits,
            self._initial_q,
            fk_position_before=fk_position_before,
            fk_position_target=fk_position_target,
            candidate_drift_m=candidate_drift_m,
            ik_error=ik_error,
            candidate_step_m=cand_step,
            joint_limit_near_rad=self._ik_quality_log_config.joint_limit_warn_margin_rad,
        )
        base_sector = self._build_base_sector_diagnostics(
            diag=diag,
            q_before=q_before,
            q_target=q_target,
            q_candidate_from_ik=q_candidate_from_ik,
            ik_accepted=ik_accepted,
            rejection_source=rejection_source,
            raw_candidate_position=raw_pos,
            clamped_candidate_position=clamped_candidate_position,
            workspace_clamp_active=workspace_clamp_active,
            workspace_clamped_axes=workspace_clamped_axes,
            accepted_fk_position=accepted_fk_position,
        )
        diag = with_log_reasons(
            diag,
            self._ik_quality_log_config,
            ik_failure=ik_failure,
            base_sector=base_sector,
        )
        self.get_logger().warn(
            format_ik_quality_diagnostics(
                diag,
                base_sector=base_sector,
                local_window=local_window,
            )
        )

    def _resolve_ik_error_for_log(
        self,
        *,
        ik_failure: bool,
        ik_failure_diag,
        candidate_x: float,
        candidate_y: float,
        candidate_z: float,
        sim_rotation: np.ndarray | None,
        q_seed: np.ndarray,
    ) -> float:
        if ik_failure and ik_failure_diag is not None and ik_failure_diag.ik_error is not None:
            return float(ik_failure_diag.ik_error)
        if (
            not self._fk.ok
            or self._fk.model is None
            or self._fk.data is None
            or self._fk.end_frame_id is None
            or sim_rotation is None
        ):
            return 0.0
        target_pos = np.array([candidate_x, candidate_y, candidate_z], dtype=np.float64)
        if self._ik_config.task_mode == "position_only":
            ik_result = compute_ik_for_position(
                self._fk.model,
                self._fk.data,
                self._fk.end_frame_id,
                target_pos,
                q_seed,
                self._ik_config.max_iterations,
                self._ik_config.tolerance,
                self._ik_config.max_ik_error,
            )
        else:
            ik_result = compute_ik_for_pose(
                self._fk.model,
                self._fk.data,
                self._fk.end_frame_id,
                target_pos,
                sim_rotation,
                q_seed,
                self._ik_config.max_iterations,
                self._ik_config.tolerance,
                self._ik_config.max_ik_error,
            )
        return float(ik_result.error)

    def _log_post_ik_gate_rejection(
        self,
        gate_result: IkGateSequenceResult,
        *,
        q_before_ik: np.ndarray,
        fk_position_before: tuple[float, float, float],
        candidate_x: float,
        candidate_y: float,
        candidate_z: float,
        raw_candidate_x: float,
        raw_candidate_y: float,
        raw_candidate_z: float,
        now_ns: int,
        sim_rotation: np.ndarray | None,
        workspace_clamp_active: bool,
        workspace_clamped_axes: tuple[str, ...],
    ) -> None:
        if gate_result.global_cap_info is not None:
            self._maybe_log_ik_rejection(
                format_joint1_global_operational_limit_log(gate_result.global_cap_info),
                now_ns,
            )
        elif gate_result.anchor_info is not None:
            self._maybe_log_ik_rejection(
                format_joint1_anchor_window_log(gate_result.anchor_info),
                now_ns,
            )
        elif gate_result.joint_limit_info is not None:
            self._maybe_log_joint_near_limit(gate_result.joint_limit_info, now_ns)

        self._maybe_log_ik_quality(
            q_before=q_before_ik,
            q_target=np.asarray(q_before_ik, dtype=np.float64),
            fk_position_before=fk_position_before,
            fk_position_target=fk_position_before,
            candidate_x=candidate_x,
            candidate_y=candidate_y,
            candidate_z=candidate_z,
            candidate_drift_m=0.0,
            now_ns=now_ns,
            ik_failure=True,
            q_candidate_from_ik=gate_result.q_from_ik,
            rejection_source=gate_result.rejection_source,
            ik_accepted=False,
            raw_candidate_position=(
                raw_candidate_x,
                raw_candidate_y,
                raw_candidate_z,
            ),
            workspace_clamp_active=workspace_clamp_active,
            workspace_clamped_axes=workspace_clamped_axes,
            resolve_ik_error=lambda: self._resolve_ik_error_for_log(
                ik_failure=True,
                ik_failure_diag=None,
                candidate_x=candidate_x,
                candidate_y=candidate_y,
                candidate_z=candidate_z,
                sim_rotation=sim_rotation,
                q_seed=q_before_ik,
            ),
        )

    def _publish_fake_joint_state(self) -> None:
        if self._fake_joint_states_publisher is None:
            return
        msg = build_fake_joint_state(self._last_valid_fake_q)
        msg.header.stamp = self.get_clock().now().to_msg()
        self._fake_joint_states_publisher.publish(msg)

    def tick(self):
        now_ns = self.get_clock().now().nanoseconds
        dt = (now_ns - self.last_tick_time_ns) / 1e9
        self.last_tick_time_ns = now_ns

        command_age = self.get_command_age()
        state_name = compute_state_name(
            self.latest_cmd,
            command_age,
            self.command_timeout_s,
        )

        self.last_clamp_reason = ""

        deadman_pressed = (
            self.latest_cmd is not None
            and self.latest_cmd.deadman
            and not self.latest_cmd.soft_stop
            and state_name == "ACTIVE"
        )
        self._maybe_reanchor_on_deadman_rising(deadman_pressed)

        current_pose, sim_rotation, fk_error = self._pose_from_q_sim()
        q_current_list = [float(v) for v in self._q_sim]
        q_before_ik = np.asarray(self._q_sim, dtype=np.float64).copy()

        sim_x = self.committed_target_x
        sim_y = self.committed_target_y
        sim_z = self.committed_target_z
        if current_pose is not None:
            sim_x = float(current_pose.position.x)
            sim_y = float(current_pose.position.y)
            sim_z = float(current_pose.position.z)

        fk_position_before = (
            pos3_from_pose(current_pose) if current_pose is not None else (sim_x, sim_y, sim_z)
        )

        candidate_x = sim_x
        candidate_y = sim_y
        candidate_z = sim_z
        raw_candidate_x = sim_x
        raw_candidate_y = sim_y
        raw_candidate_z = sim_z
        workspace_clamp_active = False
        workspace_clamped_axes: tuple[str, ...] = ()
        local_clamp_active = False
        local_clamped_axes: tuple[str, ...] = ()
        base_jog_result: BaseJogResult | None = None
        cartesian_skipped = False
        run_ik = False

        base_jog_active = (
            state_name == "ACTIVE"
            and self.latest_cmd is not None
            and self._enable_base_joint_jog
            and bool(self.latest_cmd.base_jog_active)
            and self._joint1_idx is not None
        )

        if base_jog_active:
            self._teleop_mode = "base_jog"
            cartesian_skipped = True
            base_jog_result = apply_base_joint1_jog(
                self._q_sim,
                joint1_index=self._joint1_idx,
                velocity_rad_s=float(self.latest_cmd.joint1_jog_velocity_rad_s),
                dt=dt,
                min_rad=self._joint1_global_cap_config.min_rad,
                max_rad=self._joint1_global_cap_config.max_rad,
            )
            self._last_base_jog_result = base_jog_result
            self._q_sim = np.asarray(base_jog_result.q_after, dtype=np.float64)
            self._reanchor_local_window_from_q_sim()
            self._last_valid_fake_q = [float(v) for v in self._q_sim]
            q_current_list = [float(v) for v in self._q_sim]
            current_pose, sim_rotation, fk_err2 = self._pose_from_q_sim()
            if fk_err2:
                self._fk_tick_error = fk_err2
            if current_pose is not None:
                sim_x = float(current_pose.position.x)
                sim_y = float(current_pose.position.y)
                sim_z = float(current_pose.position.z)
                fk_position_before = pos3_from_pose(current_pose)
            candidate_x = sim_x
            candidate_y = sim_y
            candidate_z = sim_z
            raw_candidate_x = sim_x
            raw_candidate_y = sim_y
            raw_candidate_z = sim_z
            self.last_clamp_reason = ""

        elif (
            state_name == "ACTIVE"
            and deadman_pressed
            and self.latest_cmd is not None
            and self._enable_local_teleop_window
        ):
            self._teleop_mode = "cartesian"
            v_local = (
                float(self.latest_cmd.linear.x),
                float(self.latest_cmd.linear.y),
                float(self.latest_cmd.linear.z),
            )
            uncapped_offset = integrate_local_target_offset(
                self._local_window_state.local_offset,
                v_local,
                dt,
            )
            anchor = self._local_window_state.anchor_position
            base_q = self._local_window_state.base_anchor_q
            forward, lateral = compute_local_axes(base_q)
            raw_candidate_x = (
                anchor[0] + forward[0] * uncapped_offset[0] + lateral[0] * uncapped_offset[1]
            )
            raw_candidate_y = (
                anchor[1] + forward[1] * uncapped_offset[0] + lateral[1] * uncapped_offset[1]
            )
            raw_candidate_z = anchor[2] + uncapped_offset[2]

            self._local_window_state, clamp_result = integrate_local_window_candidate(
                self._local_window_state,
                v_local,
                dt,
                self._local_window_limits,
            )
            candidate_x, candidate_y, candidate_z = clamp_result.target_base_link
            local_clamp_active = clamp_result.clamp_active
            local_clamped_axes = clamp_result.clamped_axes
            self.last_clamp_reason = ",".join(local_clamped_axes) if local_clamp_active else ""
            run_ik = current_pose is not None and sim_rotation is not None

        elif state_name == "ACTIVE" and self.latest_cmd is not None:
            self._teleop_mode = "cartesian"
            raw_candidate_x = sim_x + float(self.latest_cmd.linear.x) * dt
            raw_candidate_y = sim_y + float(self.latest_cmd.linear.y) * dt
            raw_candidate_z = sim_z + float(self.latest_cmd.linear.z) * dt
            candidate_x, candidate_y, candidate_z, self.last_clamp_reason = (
                compute_candidate_target(
                    sim_x,
                    sim_y,
                    sim_z,
                    self.latest_cmd,
                    dt,
                    self._workspace,
                )
            )
            workspace_clamp_active = bool(self.last_clamp_reason)
            workspace_clamped_axes = tuple(
                part for part in self.last_clamp_reason.split(",") if part
            )
            run_ik = current_pose is not None and sim_rotation is not None
        else:
            self._teleop_mode = "hold"
            self.last_clamp_reason = ""

        q_target: list[float] = []
        ik_success = False
        ik_reason = ""
        ik_failure_diag = None

        if run_ik and not base_jog_active:
            q_target, ik_success, ik_reason, ik_failure_diag = solve_target_ik(
                fk_ctx=self._fk,
                state_name=state_name,
                target_x=candidate_x,
                target_y=candidate_y,
                target_z=candidate_z,
                target_rotation=sim_rotation,
                q_seed=self._q_sim,
                ik_config=self._ik_config,
                clamp_reason=self.last_clamp_reason,
                committed_x=self.committed_target_x,
                committed_y=self.committed_target_y,
                committed_z=self.committed_target_z,
            )
            self._maybe_log_ik_failure(ik_failure_diag, now_ns)

            if not ik_success:
                self._maybe_log_ik_quality(
                    q_before=q_before_ik,
                    q_target=q_before_ik,
                    fk_position_before=fk_position_before,
                    fk_position_target=fk_position_before,
                    candidate_x=candidate_x,
                    candidate_y=candidate_y,
                    candidate_z=candidate_z,
                    candidate_drift_m=0.0,
                    now_ns=now_ns,
                    ik_failure=True,
                    q_candidate_from_ik=None,
                    rejection_source=ik_reason or "IK_FAILURE",
                    ik_accepted=False,
                    raw_candidate_position=(
                        raw_candidate_x,
                        raw_candidate_y,
                        raw_candidate_z,
                    ),
                    workspace_clamp_active=workspace_clamp_active,
                    workspace_clamped_axes=workspace_clamped_axes,
                    resolve_ik_error=lambda: self._resolve_ik_error_for_log(
                        ik_failure=True,
                        ik_failure_diag=ik_failure_diag,
                        candidate_x=candidate_x,
                        candidate_y=candidate_y,
                        candidate_z=candidate_z,
                        sim_rotation=sim_rotation,
                        q_seed=q_before_ik,
                    ),
                )

        if ik_success:
            gate_result = apply_ik_gate_sequence(
                IkGateSequenceInput(
                    fk_ctx=self._fk,
                    q_candidate=list(q_target),
                    q_before=q_before_ik,
                    joint_names=self._joint_names,
                    lower_limits=self._joint_lower_limits,
                    upper_limits=self._joint_upper_limits,
                    base_anchor_q=self._base_anchor_q,
                    candidate_x=candidate_x,
                    candidate_y=candidate_y,
                    candidate_z=candidate_z,
                    fk_position_before=fk_position_before,
                    joint1_global_config=self._joint1_global_cap_config,
                    joint1_anchor_config=self._joint1_anchor_window_config,
                    joint_limit_config=self._joint_limit_reject_config,
                    ik_no_effect_config=self._ik_no_effect_config,
                )
            )
            if gate_result.accepted:
                q_target = gate_result.q_candidate
            else:
                ik_success = False
                ik_reason = gate_result.rejection_reason
                q_target = gate_result.q_candidate
                self._log_post_ik_gate_rejection(
                    gate_result,
                    q_before_ik=q_before_ik,
                    fk_position_before=fk_position_before,
                    candidate_x=candidate_x,
                    candidate_y=candidate_y,
                    candidate_z=candidate_z,
                    raw_candidate_x=raw_candidate_x,
                    raw_candidate_y=raw_candidate_y,
                    raw_candidate_z=raw_candidate_z,
                    now_ns=now_ns,
                    sim_rotation=sim_rotation,
                    workspace_clamp_active=workspace_clamp_active,
                    workspace_clamped_axes=workspace_clamped_axes,
                )

        if ik_success:
            drift_m = compute_candidate_drift_m(
                self._fk,
                candidate_x,
                candidate_y,
                candidate_z,
                q_target,
            )
            q_target_arr = np.asarray(q_target, dtype=np.float64)
            fk_target_pose, _, fk_target_err = compute_fk_pose_for_q(self._fk, q_target_arr)
            fk_position_target = (
                pos3_from_pose(fk_target_pose)
                if fk_target_pose is not None and not fk_target_err
                else fk_position_before
            )
            local_diag = None
            if self._enable_local_teleop_window:
                local_diag = self._build_local_window_diagnostics(
                    target_before_ik=(candidate_x, candidate_y, candidate_z),
                    clamp_active=local_clamp_active,
                    clamped_axes=local_clamped_axes,
                    joint1_candidate_q=(
                        float(q_target_arr[self._joint1_idx])
                        if self._joint1_idx is not None
                        else None
                    ),
                    cartesian_skipped=cartesian_skipped,
                    base_jog_result=base_jog_result,
                )
            self._maybe_log_ik_quality(
                q_before=q_before_ik,
                q_target=q_target_arr,
                fk_position_before=fk_position_before,
                fk_position_target=fk_position_target,
                candidate_x=candidate_x,
                candidate_y=candidate_y,
                candidate_z=candidate_z,
                candidate_drift_m=drift_m,
                now_ns=now_ns,
                ik_failure=False,
                q_candidate_from_ik=list(q_target),
                rejection_source="",
                ik_accepted=True,
                raw_candidate_position=(
                    raw_candidate_x,
                    raw_candidate_y,
                    raw_candidate_z,
                ),
                workspace_clamp_active=workspace_clamp_active,
                workspace_clamped_axes=workspace_clamped_axes,
                accepted_fk_position=fk_position_target,
                local_window=local_diag,
                resolve_ik_error=lambda: self._resolve_ik_error_for_log(
                    ik_failure=False,
                    ik_failure_diag=None,
                    candidate_x=candidate_x,
                    candidate_y=candidate_y,
                    candidate_z=candidate_z,
                    sim_rotation=sim_rotation,
                    q_seed=q_before_ik,
                ),
            )
            self._q_sim = update_q_sim_on_ik_success(self._q_sim, q_target, True)
            (
                self.committed_target_x,
                self.committed_target_y,
                self.committed_target_z,
                self._committed_rotation,
                current_pose,
                fk_resync_err,
            ) = resync_committed_from_q_sim(self._fk, self._q_sim)
            if fk_resync_err:
                self._fk_tick_error = fk_resync_err
            else:
                self._fk_tick_error = ""
            if drift_m > self._candidate_drift_log_threshold_m:
                self.get_logger().debug(
                    f"IK candidate drift (log only): {drift_m:.6f} m "
                    f"(threshold {self._candidate_drift_log_threshold_m:.6f} m)"
                )
            q_current_list = [float(v) for v in self._q_sim]
            self._last_valid_fake_q = q_current_list
            q_target = q_current_list

        if base_jog_active:
            q_target = q_current_list
            ik_success = False
            ik_reason = "BASE_JOG"

        committed_target_pose = build_committed_target_pose(
            self.committed_target_x,
            self.committed_target_y,
            self.committed_target_z,
            self._committed_rotation,
        )

        msg = build_cartesian_jog_state(
            state_name=state_name,
            target_x=self.committed_target_x,
            target_y=self.committed_target_y,
            target_z=self.committed_target_z,
            latest_cmd=self.latest_cmd,
            clamp_reason=self.last_clamp_reason,
            dry_run=self.dry_run,
            output_mode=self.output_mode,
            command_age=command_age,
            current_pose=current_pose,
            target_pose=committed_target_pose,
            q_current=q_current_list,
            q_target=q_target,
            ik_success=ik_success,
            fk_error=fk_error,
            ik_reason=ik_reason,
        )
        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CartesianJogCore()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
