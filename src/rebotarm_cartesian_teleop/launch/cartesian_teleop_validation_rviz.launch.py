"""RViz validation stack for Cartesian teleop (base_link-aligned view + TCP markers).

Launch together with (separate terminals):
  just run-joy
  just run-joy-mapper
  just run-cartesian-core

Validation workflow:
  1. In RViz Views panel, select saved view ``TeleopBaseValidation`` (or ``TeleopTopDownZ``).
  2. Do not orbit the camera while testing axis motion.
  3. Jog one axis at a time (+X, -X, +Y, -Y, +Z, -Z in base_link).
  4. Judge TCP motion via the small green TCP sphere + blue trail vs base_link grid/axes.
  5. Guide the TCP into blue validation targets; they turn grey on contact.
"""

from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from rebotarm_bringup.robot_description_launch import (
    d405_launch_arguments,
    robot_description_parameter,
)

from launch import LaunchDescription


def generate_launch_description():
    teleop_share = FindPackageShare("rebotarm_cartesian_teleop")

    fake_joint_states_topic = LaunchConfiguration("fake_joint_states_topic")

    rviz_config = PathJoinSubstitution(
        [teleop_share, "rviz", "cartesian_teleop_validation.rviz"]
    )
    teleop_params = PathJoinSubstitution(
        [teleop_share, "config", "cartesian_teleop.yaml"]
    )
    robot_description = robot_description_parameter()

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "fake_joint_states_topic",
                default_value="/rebotarm/fake_joint_states",
            ),
            *d405_launch_arguments(),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[{"robot_description": robot_description}],
                remappings=[("/joint_states", fake_joint_states_topic)],
            ),
            Node(
                package="rebotarm_cartesian_teleop",
                executable="teleop_viz_markers",
                name="teleop_viz_markers",
                output="screen",
                parameters=[teleop_params],
            ),
            Node(
                package="rebotarm_cartesian_teleop",
                executable="teleop_validation_targets",
                name="teleop_validation_targets",
                output="screen",
                parameters=[teleop_params],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_config],
            ),
        ]
    )
