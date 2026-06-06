import rclpy
from rclpy.node import Node
from rebotarm_msgs.msg import CartesianJogCmd
from sensor_msgs.msg import Joy

from .joy_mapping import JoyMapperConfig, VelocitySmoothingState, map_joy_to_cmd


class JoyCartesianMapper(Node):
    _PUBLISH_HZ = 30.0

    def __init__(self):
        super().__init__("joy_cartesian_mapper")

        self.declare_parameter("joy_topic", "/joy")
        self.declare_parameter("cartesian_jog_cmd_topic", "/rebotarm/cartesian_jog_cmd")

        self.declare_parameter("axis_x", 1)
        self.declare_parameter("axis_y", 0)
        self.declare_parameter("axis_z", 5)

        self.declare_parameter("invert_x", False)
        self.declare_parameter("invert_y", False)
        self.declare_parameter("invert_z", False)

        self.declare_parameter("deadzone", 0.15)
        self.declare_parameter("joy_timeout_s", 0.3)
        self.declare_parameter("max_linear_velocity_m_s", 0.03)

        self.declare_parameter("deadman_button", 4)
        self.declare_parameter("soft_stop_button", 2)
        self.declare_parameter("speed_boost_button", 5)

        self.declare_parameter("speed_scale_default", 1.0)
        self.declare_parameter("speed_scale_boost", 1.5)
        self.declare_parameter("enable_velocity_smoothing", True)
        self.declare_parameter("max_linear_accel_m_s2", 0.25)
        self.declare_parameter("velocity_smoothing_reset_on_deadman_release", True)
        self.declare_parameter("velocity_smoothing_reset_on_soft_stop", True)
        self.declare_parameter("enable_base_joint_jog", True)
        self.declare_parameter("enable_local_teleop_window", True)
        self.declare_parameter("base_jog_input_type", "axis")
        self.declare_parameter("base_jog_axis_index", 6)
        self.declare_parameter("base_jog_axis_deadzone", 0.5)
        self.declare_parameter("base_jog_axis_to_joint_sign", -1.0)
        self.declare_parameter("base_jog_left_button", 13)
        self.declare_parameter("base_jog_right_button", 14)
        self.declare_parameter("base_joint_jog_speed_rad_s", 0.5)

        joy_topic = self.get_parameter("joy_topic").value
        cmd_topic = self.get_parameter("cartesian_jog_cmd_topic").value

        self._config = JoyMapperConfig(
            axis_x=int(self.get_parameter("axis_x").value),
            axis_y=int(self.get_parameter("axis_y").value),
            axis_z=int(self.get_parameter("axis_z").value),
            invert_x=bool(self.get_parameter("invert_x").value),
            invert_y=bool(self.get_parameter("invert_y").value),
            invert_z=bool(self.get_parameter("invert_z").value),
            deadzone=float(self.get_parameter("deadzone").value),
            joy_timeout_s=float(self.get_parameter("joy_timeout_s").value),
            max_linear_velocity=float(self.get_parameter("max_linear_velocity_m_s").value),
            deadman_button=int(self.get_parameter("deadman_button").value),
            soft_stop_button=int(self.get_parameter("soft_stop_button").value),
            speed_boost_button=int(self.get_parameter("speed_boost_button").value),
            speed_scale_default=float(self.get_parameter("speed_scale_default").value),
            speed_scale_boost=float(self.get_parameter("speed_scale_boost").value),
            enable_velocity_smoothing=bool(
                self.get_parameter("enable_velocity_smoothing").value
            ),
            max_linear_accel_m_s2=float(self.get_parameter("max_linear_accel_m_s2").value),
            velocity_smoothing_reset_on_deadman_release=bool(
                self.get_parameter("velocity_smoothing_reset_on_deadman_release").value
            ),
            velocity_smoothing_reset_on_soft_stop=bool(
                self.get_parameter("velocity_smoothing_reset_on_soft_stop").value
            ),
            enable_base_joint_jog=bool(self.get_parameter("enable_base_joint_jog").value),
            enable_local_teleop_window=bool(
                self.get_parameter("enable_local_teleop_window").value
            ),
            base_jog_input_type=str(self.get_parameter("base_jog_input_type").value),
            base_jog_axis_index=int(self.get_parameter("base_jog_axis_index").value),
            base_jog_axis_deadzone=float(self.get_parameter("base_jog_axis_deadzone").value),
            base_jog_axis_to_joint_sign=float(
                self.get_parameter("base_jog_axis_to_joint_sign").value
            ),
            base_jog_left_button=int(self.get_parameter("base_jog_left_button").value),
            base_jog_right_button=int(self.get_parameter("base_jog_right_button").value),
            base_joint_jog_speed_rad_s=float(
                self.get_parameter("base_joint_jog_speed_rad_s").value
            ),
        )

        self._smoothing_state = VelocitySmoothingState()

        self.latest_joy = None
        self.latest_joy_time_ns = None

        self.subscription = self.create_subscription(
            Joy,
            joy_topic,
            self.on_joy,
            10,
        )

        self.publisher = self.create_publisher(
            CartesianJogCmd,
            cmd_topic,
            10,
        )

        self.timer = self.create_timer(1.0 / 30.0, self.publish_cmd)

        self.get_logger().info("joy_cartesian_mapper started")
        self.get_logger().info(f"Listening to: {joy_topic}")
        self.get_logger().info(f"Publishing to: {cmd_topic}")
        self.get_logger().info(
            f"Axes x/y/z: {self._config.axis_x}/{self._config.axis_y}/{self._config.axis_z}"
        )
        self.get_logger().info(f"Deadman button: {self._config.deadman_button}")
        self.get_logger().info(f"Soft stop button: {self._config.soft_stop_button}")
        self.get_logger().info(f"Speed boost button: {self._config.speed_boost_button}")
        self.get_logger().info(f"Joy timeout: {self._config.joy_timeout_s}s")

    def on_joy(self, msg: Joy):
        self.latest_joy = msg
        self.latest_joy_time_ns = self.get_clock().now().nanoseconds
        self.get_logger().info(
            f"Joy received: axes={len(msg.axes)} buttons={len(msg.buttons)}",
            throttle_duration_sec=2.0,
        )

    def publish_cmd(self):
        now_ns = self.get_clock().now().nanoseconds
        msg = map_joy_to_cmd(
            self.latest_joy,
            self._config,
            latest_joy_time_ns=self.latest_joy_time_ns,
            now_ns=now_ns,
            smoothing_state=self._smoothing_state,
            publish_hz=self._PUBLISH_HZ,
        )
        if msg is None:
            self.get_logger().warn(
                "Joy input timeout; not publishing CartesianJogCmd",
                throttle_duration_sec=2.0,
            )
            return

        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = JoyCartesianMapper()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
