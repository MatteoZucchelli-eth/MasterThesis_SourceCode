# MIT License
"""The dataset recorder's state stream is wired into the control loop.

The recorder builds every episode from ControlStep, and nothing else subscribes
to it, so a missing publisher only shows up as empty episodes.
ControllerManagerNode cannot be constructed in a unit test, so the wiring is
checked on the source.
"""

import importlib.util
from pathlib import Path

SOURCE = Path(
    importlib.util.find_spec("g1_control.nodes.controller_manager_node").origin
).read_text()


def test_the_controller_builds_the_recorder_step_publisher():
    assert "ControlStepPublisher(self, ctx)" in SOURCE


def test_the_control_loop_publishes_one_step_per_tick():
    loop = SOURCE.split("def _control_loop(self)")[1].split("\n    def ")[0]
    assert "self._step_pub.publish(" in loop
