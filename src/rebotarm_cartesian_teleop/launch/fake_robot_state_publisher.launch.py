"""Simulation-only robot_state_publisher fed by /rebotarm/fake_joint_states."""

from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from rebotarm_bringup.robot_description_launch import (
    d405_launch_arguments,
    robot_description_parameter,
)

from launch import LaunchDescription


def generate_launch_description():
    fake_joint_states_topic = LaunchConfiguration("fake_joint_states_topic")
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
        ]
    )
