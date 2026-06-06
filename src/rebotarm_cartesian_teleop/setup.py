import os
import sys
from glob import glob

from setuptools import find_packages, setup
from setuptools.command.test import test as TestCommand


class PytestCommand(TestCommand):
    user_options = [("pytest-args=", "a", "Arguments forwarded to pytest")]

    def initialize_options(self):
        super().initialize_options()
        self.pytest_args = ""

    def finalize_options(self):
        super().finalize_options()

    def run_tests(self):
        import shlex

        import pytest

        sys.exit(pytest.main(shlex.split(self.pytest_args) + ["test"]))


package_name = "rebotarm_cartesian_teleop"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "docs"), glob("docs/*.md")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="danieldorado",
    maintainer_email="danieldoradotalaveron@gmail.com",
    description="Cartesian gamepad teleop for reBot Arm (simulation-first).",
    license="Apache-2.0",
    cmdclass={"test": PytestCommand},
    # Dev tools (pytest, ruff): uv sync in this package dir — see pyproject.toml
    entry_points={
        "console_scripts": [
            "joy_cartesian_mapper = rebotarm_cartesian_teleop.joy_cartesian_mapper:main",
            "cartesian_jog_core = rebotarm_cartesian_teleop.cartesian_jog_core:main",
            "teleop_viz_markers = rebotarm_cartesian_teleop.teleop_viz_markers:main",
            "teleop_validation_targets = rebotarm_cartesian_teleop.teleop_validation_targets:main",
        ],
    },
)
