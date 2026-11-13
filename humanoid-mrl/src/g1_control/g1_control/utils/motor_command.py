# MIT License
"""Helpers for assembling Unitree ``LowCmd`` messages."""

from __future__ import annotations

import math

import numpy as np
from g1_definitions import definitions
from unitree_hg.msg import LowCmd, MotorCmd


def build_low_cmd(
    mode_machine: int,
    q: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    tau: np.ndarray | None = None,
    dq: np.ndarray | None = None,
) -> LowCmd:
    """Build a PR-mode ``LowCmd`` for the G1 base motors.

    Each of ``q``, ``kp``, ``kd`` and (optionally) ``tau``, ``dq`` is a
    per-motor array of length :data:`definitions.NUM_MOTOR_G1_BASE`; ``tau``
    defaults to zero feedforward torque and ``dq`` to zero.

    ``dq`` is the velocity target: zero makes Kd brake all motion, so pass the
    reference velocity.
    """
    n = definitions.NUM_MOTOR_G1_BASE
    if tau is None:
        tau = np.zeros(n)
    if dq is None:
        dq = np.zeros(n)

    cmd = LowCmd()
    cmd.mode_machine = mode_machine
    cmd.mode_pr = definitions.G1JointControlMode.PR_Mode.value
    cmd.reserve = [0, 0, 0, 0]

    for i in range(n):
        mc = MotorCmd()
        mc.mode = 1
        mc.q = float(q[i])
        mc.dq = float(dq[i])
        mc.tau = float(tau[i])
        mc.kp = float(kp[i])
        mc.kd = float(kd[i])
        cmd.motor_cmd[i] = mc

    return cmd


def nonfinite_motors(cmd: LowCmd) -> list[int]:
    """Indices of base motors whose command carries a non-finite number.

    Nothing upstream rejects NaN, so this is the last check before publishing;
    empty means safe.
    """
    n = definitions.NUM_MOTOR_G1_BASE
    return [
        i
        for i, mc in enumerate(cmd.motor_cmd[:n])
        if not (
            math.isfinite(mc.q)
            and math.isfinite(mc.dq)
            and math.isfinite(mc.tau)
            and math.isfinite(mc.kp)
            and math.isfinite(mc.kd)
        )
    ]
