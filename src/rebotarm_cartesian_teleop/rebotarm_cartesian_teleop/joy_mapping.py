"""Pure Joy -> CartesianJogCmd mapping (no ROS node dependencies)."""

from __future__ import annotations

from dataclasses import dataclass

from rebotarm_msgs.msg import CartesianJogCmd
from sensor_msgs.msg import Joy

from .jog_core_logic import COMMAND_FRAME_LOCAL_WINDOW


@dataclass(frozen=True)
class JoyMapperConfig:
    axis_x: int = 1
    axis_y: int = 0
    axis_z: int = 5
    invert_x: bool = False
    invert_y: bool = False
    invert_z: bool = False
    deadzone: float = 0.15
    joy_timeout_s: float = 0.3
    max_linear_velocity: float = 0.03
    deadman_button: int = 4
    soft_stop_button: int = 2
    speed_boost_button: int = 5
    speed_scale_default: float = 1.0
    speed_scale_boost: float = 1.5
    enable_velocity_smoothing: bool = False
    max_linear_accel_m_s2: float = 0.25
    velocity_smoothing_reset_on_deadman_release: bool = True
    velocity_smoothing_reset_on_soft_stop: bool = True
    enable_base_joint_jog: bool = True
    enable_local_teleop_window: bool = True
    base_jog_input_type: str = "axis"
    base_jog_axis_index: int = 6
    base_jog_axis_deadzone: float = 0.5
    base_jog_axis_to_joint_sign: float = -1.0
    base_jog_left_button: int = 13
    base_jog_right_button: int = 14
    base_joint_jog_speed_rad_s: float = 0.5


@dataclass
class VelocitySmoothingState:
    prev_linear_x: float = 0.0
    prev_linear_y: float = 0.0
    prev_linear_z: float = 0.0
    last_publish_time_ns: int | None = None
    prev_base_jog_active: bool = False

    def reset(self) -> None:
        self.prev_linear_x = 0.0
        self.prev_linear_y = 0.0
        self.prev_linear_z = 0.0


def button_pressed(joy: Joy | None, button_index: int) -> bool:
    if joy is None:
        return False
    if button_index < 0 or button_index >= len(joy.buttons):
        return False
    return joy.buttons[button_index] == 1


def raw_axis_value(joy: Joy | None, axis_index: int) -> float:
    if joy is None:
        return 0.0
    if axis_index < 0 or axis_index >= len(joy.axes):
        return 0.0
    return float(joy.axes[axis_index])


def axis_value(
    joy: Joy | None,
    axis_index: int,
    invert: bool,
    deadzone: float,
) -> float:
    if joy is None:
        return 0.0
    if axis_index < 0 or axis_index >= len(joy.axes):
        return 0.0

    value = float(joy.axes[axis_index])
    if abs(value) < deadzone:
        return 0.0
    if invert:
        value = -value
    return value


def is_joy_fresh(
    latest_joy_time_ns: int | None,
    now_ns: int,
    joy_timeout_s: float,
) -> bool:
    if latest_joy_time_ns is None:
        return False
    age_s = (now_ns - latest_joy_time_ns) / 1e9
    return age_s <= joy_timeout_s


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def resolve_smoothing_dt_s(
    now_ns: int,
    last_publish_time_ns: int | None,
    publish_hz: float,
    *,
    max_dt_s: float = 0.5,
) -> float:
    """Return dt for acceleration limiting; fall back to 1/publish_hz when invalid."""
    fallback_dt = 1.0 / publish_hz if publish_hz > 0.0 else 1.0 / 30.0
    if last_publish_time_ns is None:
        return fallback_dt
    dt_s = (now_ns - last_publish_time_ns) / 1e9
    if dt_s <= 0.0 or dt_s > max_dt_s:
        return fallback_dt
    return dt_s


def smooth_linear_axis(
    raw: float,
    previous: float,
    max_delta_v: float,
) -> float:
    delta = _clamp(raw - previous, -max_delta_v, max_delta_v)
    return previous + delta


def apply_linear_velocity_smoothing(
    raw_linear_x: float,
    raw_linear_y: float,
    raw_linear_z: float,
    *,
    deadman: bool,
    soft_stop: bool,
    cfg: JoyMapperConfig,
    state: VelocitySmoothingState,
    dt_s: float,
) -> tuple[float, float, float]:
    """Acceleration-limit linear velocity toward raw targets."""
    if soft_stop:
        if cfg.velocity_smoothing_reset_on_soft_stop:
            state.reset()
        return 0.0, 0.0, 0.0

    if not deadman:
        if cfg.velocity_smoothing_reset_on_deadman_release:
            state.reset()
        return 0.0, 0.0, 0.0

    max_delta_v = cfg.max_linear_accel_m_s2 * dt_s
    smoothed_x = smooth_linear_axis(raw_linear_x, state.prev_linear_x, max_delta_v)
    smoothed_y = smooth_linear_axis(raw_linear_y, state.prev_linear_y, max_delta_v)
    smoothed_z = smooth_linear_axis(raw_linear_z, state.prev_linear_z, max_delta_v)
    state.prev_linear_x = smoothed_x
    state.prev_linear_y = smoothed_y
    state.prev_linear_z = smoothed_z
    return smoothed_x, smoothed_y, smoothed_z


def resolve_base_jog_from_joy(
    joy: Joy | None,
    cfg: JoyMapperConfig,
) -> tuple[bool, float]:
    """Map D-pad / base-jog input to joint1 jog velocity (rad/s)."""
    if not cfg.enable_base_joint_jog:
        return False, 0.0

    raw_axis = 0.0
    input_type = (cfg.base_jog_input_type or "axis").strip().lower()
    if input_type == "button":
        left = button_pressed(joy, cfg.base_jog_left_button)
        right = button_pressed(joy, cfg.base_jog_right_button)
        if left and not right:
            raw_axis = 1.0
        elif right and not left:
            raw_axis = -1.0
        else:
            return False, 0.0
    else:
        raw_axis = raw_axis_value(joy, cfg.base_jog_axis_index)
        if abs(raw_axis) < cfg.base_jog_axis_deadzone:
            return False, 0.0

    velocity = (
        raw_axis * float(cfg.base_jog_axis_to_joint_sign) * float(cfg.base_joint_jog_speed_rad_s)
    )
    return True, velocity


def map_joy_to_cmd(
    joy: Joy | None,
    cfg: JoyMapperConfig,
    *,
    latest_joy_time_ns: int | None,
    now_ns: int,
    smoothing_state: VelocitySmoothingState | None = None,
    publish_hz: float = 30.0,
) -> CartesianJogCmd | None:
    """Map Joy to CartesianJogCmd, or None if Joy is stale (do not publish)."""
    if not is_joy_fresh(latest_joy_time_ns, now_ns, cfg.joy_timeout_s):
        return None

    msg = CartesianJogCmd()
    msg.header.frame_id = "base_link"

    deadman = button_pressed(joy, cfg.deadman_button)
    soft_stop = button_pressed(joy, cfg.soft_stop_button)
    speed_boost = button_pressed(joy, cfg.speed_boost_button)

    speed_scale = cfg.speed_scale_boost if speed_boost else cfg.speed_scale_default

    base_jog_active, joint1_jog_velocity = resolve_base_jog_from_joy(joy, cfg)

    if smoothing_state is not None and smoothing_state.prev_base_jog_active and not base_jog_active:
        smoothing_state.reset()

    x = axis_value(joy, cfg.axis_x, cfg.invert_x, cfg.deadzone)
    y = axis_value(joy, cfg.axis_y, cfg.invert_y, cfg.deadzone)
    z = axis_value(joy, cfg.axis_z, cfg.invert_z, cfg.deadzone)

    scale = cfg.max_linear_velocity * speed_scale
    raw_linear_x = 0.0 if base_jog_active else x * scale
    raw_linear_y = 0.0 if base_jog_active else y * scale
    raw_linear_z = 0.0 if base_jog_active else z * scale

    if cfg.enable_velocity_smoothing and smoothing_state is not None:
        dt_s = resolve_smoothing_dt_s(
            now_ns,
            smoothing_state.last_publish_time_ns,
            publish_hz,
        )
        smoothing_state.last_publish_time_ns = now_ns
        linear_x, linear_y, linear_z = apply_linear_velocity_smoothing(
            raw_linear_x,
            raw_linear_y,
            raw_linear_z,
            deadman=deadman,
            soft_stop=soft_stop,
            cfg=cfg,
            state=smoothing_state,
            dt_s=dt_s,
        )
    else:
        if not deadman or soft_stop:
            linear_x = 0.0
            linear_y = 0.0
            linear_z = 0.0
        else:
            linear_x = raw_linear_x
            linear_y = raw_linear_y
            linear_z = raw_linear_z

    msg.linear.x = linear_x
    msg.linear.y = linear_y
    msg.linear.z = linear_z

    msg.angular.x = 0.0
    msg.angular.y = 0.0
    msg.angular.z = 0.0

    msg.deadman = deadman
    msg.soft_stop = soft_stop
    msg.speed_scale = speed_scale
    msg.enable_orientation = False
    msg.base_jog_active = base_jog_active
    msg.joint1_jog_velocity_rad_s = float(joint1_jog_velocity)
    msg.command_frame_kind = (
        COMMAND_FRAME_LOCAL_WINDOW
        if cfg.enable_local_teleop_window
        else "base_link"
    )

    if smoothing_state is not None:
        smoothing_state.prev_base_jog_active = base_jog_active

    return msg
