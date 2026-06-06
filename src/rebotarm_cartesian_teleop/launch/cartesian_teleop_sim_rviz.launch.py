"""Simulation-only RViz for Cartesian teleop (fake_joint_states -> robot_state_publisher)."""

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
    bringup_share = FindPackageShare("rebotarm_bringup")
    fake_joint_states_topic = LaunchConfiguration("fake_joint_states_topic")

    rviz_config = PathJoinSubstitution([bringup_share, "rviz", "rebotarm.rviz"])
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
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_config],
            ),
        ]
    )
