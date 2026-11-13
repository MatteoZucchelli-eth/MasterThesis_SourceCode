#!/usr/bin/env python3
"""
Compute PD damping (Kd) for the G1 arms using the mass matrix from pinocchio.

Kp and current Kd are read from gear_wbc.yaml; only a new Kd is proposed. The
right arm carries the drill payload (from the URDF), the left is bare. Per joint:
  1. Sample joint space for worst-case I_eff = M(q)[j, j] and |g(q)[j]|.
  2. Report the current damping ratio  ζ = Kd / (2·√(Kp·I_eff)).
  3. Propose Kd = 2·ζ·√(Kp·I_eff_max) for a target ζ (default 0.7).
  4. Report worst-case torque τ = g(q) + Kp·Δq_max + Kd·ω_max against the
     effort limit (a conservative saturation-margin warning).

Legs and waist are not tuned: their gains belong to the RL locomotion policy.

Usage:
    python3 scripts/tune_arm_gains.py
    python3 scripts/tune_arm_gains.py --zeta 1.0        # critical damping
    python3 scripts/tune_arm_gains.py --samples 20000   # finer worst-case search
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml

# ── Paths ────────────────────────────────────────────────────────────────────
_INSTALL_ROOT = Path(__file__).resolve().parents[3] / "install"
URDF_PATH = str(
    _INSTALL_ROOT
    / "g1_state_publisher/share/g1_state_publisher/urdf"
    / "g1_29_dof_with_hand_drill.urdf.xml"
)


def _arm_joints(side: str):
    return [
        f"{side}_shoulder_pitch_joint",
        f"{side}_shoulder_roll_joint",
        f"{side}_shoulder_yaw_joint",
        f"{side}_elbow_joint",
        f"{side}_wrist_roll_joint",
        f"{side}_wrist_pitch_joint",
        f"{side}_wrist_yaw_joint",
    ]


# motor_kps / motor_kds in gear_wbc.yaml are 29-vectors in PR order:
#   [left_leg(6), right_leg(6), waist(3), left_arm(7), right_arm(7)]
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "gear_wbc.yaml"
_ARM_SLICE = {"left": slice(15, 22), "right": slice(22, 29)}


def load_gains(config_path: Path):
    """Return (kp_by_side, kd_by_side, kd_full) read from gear_wbc.yaml.

    kp_by_side / kd_by_side map "left"/"right" to the arm 7-vector; kd_full is
    the whole 29-vector.
    """
    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    kps = np.asarray(cfg["motor_kps"], dtype=float)
    kds = np.asarray(cfg["motor_kds"], dtype=float)
    if kps.size != 29 or kds.size != 29:
        sys.exit(f"Expected 29 motor_kps/motor_kds, got {kps.size}/{kds.size}.")
    kp_by_side = {s: kps[sl] for s, sl in _ARM_SLICE.items()}
    kd_by_side = {s: kds[sl] for s, sl in _ARM_SLICE.items()}
    return kp_by_side, kd_by_side, kds


# Effort limits from the URDF <limit effort="..."/> (same for both arms).
EFFORT_LIMITS = np.array([25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0])  # Nm

# Practical operating bound on the position tracking error (NOT mechanical max).
POS_ERR_MAX = 0.087  # rad (~5°)

# Joint speed for the torque-headroom check: upper_body_max_joint_speed from the config.
VEL_MAX_FALLBACK = 0.25  # rad/s, only if the config does not say


def load_vel_max(config_path: Path) -> float:
    """The interpolator's joint-speed limit, from the config being tuned."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return float(cfg.get("upper_body_max_joint_speed", VEL_MAX_FALLBACK))


# Position-limit fallback for any joint that reports a non-finite range.
_FALLBACK_LIMIT = np.pi


def build_model() -> "tuple[pin.Model, pin.Data]":
    if not Path(URDF_PATH).exists():
        sys.exit(
            f"URDF not found: {URDF_PATH}\n"
            "Run 'colcon build' first, or fix URDF_PATH in this script."
        )
    model = pin.buildModelFromUrdf(URDF_PATH)
    return model, model.createData()


def _arm_indices(model: pin.Model, joints):
    """Return (idx_q, idx_v, lower, upper) arrays for the given joints.

    idx_q indexes the configuration vector (and the position limits); idx_v
    indexes the mass matrix and gravity vector, which live in velocity space.
    """
    idx_q, idx_v, lo, hi = [], [], [], []
    for name in joints:
        jm = model.joints[model.getJointId(name)]
        idx_q.append(jm.idx_q)
        idx_v.append(jm.idx_v)
        low = model.lowerPositionLimit[jm.idx_q]
        high = model.upperPositionLimit[jm.idx_q]
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low, high = -_FALLBACK_LIMIT, _FALLBACK_LIMIT
        lo.append(low)
        hi.append(high)
    return (np.array(idx_q), np.array(idx_v), np.array(lo), np.array(hi))


def worst_case_inertia_gravity(model, data, joints, samples: int, seed: int = 0):
    """Sample the arm's joint space; return per-joint worst-case I_eff, |g|.

    Other joints stay neutral; the first sample is the neutral pose.
    """
    rng = np.random.default_rng(seed)
    idx_q, idx_v, lo, hi = _arm_indices(model, joints)

    n = len(joints)
    i_eff_max = np.zeros(n)
    g_max = np.zeros(n)

    q = pin.neutral(model)
    for s in range(samples):
        if s > 0:
            q[idx_q] = rng.uniform(lo, hi)
        M = pin.crba(model, data, q)
        g = pin.computeGeneralizedGravity(model, data, q)
        i_eff_max = np.maximum(i_eff_max, M[idx_v, idx_v])
        g_max = np.maximum(g_max, np.abs(g[idx_v]))

    return i_eff_max, g_max


def analyze_arm(model, data, side, kp_arm, kd_cur, target_zeta, samples, vel_max):
    """Print the damping + torque tables for one arm and return the proposed Kd.

    Returns (kd_new, over_joints) where over_joints lists (name, tau_tot, ratio)
    for every joint whose worst-case torque exceeds its effort limit.
    """
    joints = _arm_joints(side)
    missing = [n for n in joints if not model.existJointName(n)]
    if missing:
        sys.exit(f"Joints not found in URDF: {missing}")

    i_eff, g_max = worst_case_inertia_gravity(model, data, joints, samples)

    kd_new = np.round(2.0 * target_zeta * np.sqrt(kp_arm * i_eff), 1)
    zeta_current = kd_cur / (2.0 * np.sqrt(kp_arm * i_eff))

    payload = "with drill" if side == "right" else "bare"
    hdr = (
        f"{'Joint':<28} {'I_eff_max':>10} {'omega_n':>9} {'zeta_cur':>9}"
        f" {'zeta_tgt':>9} {'Kd_cur':>7} {'Kd_new':>7}"
    )
    print("\n" + "=" * len(hdr))
    print(
        f"  {side.upper()} ARM ({payload}) — target zeta = {target_zeta:.2f},"
        f" Kp from config, {samples} samples"
    )
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for i, name in enumerate(joints):
        omega_n = np.sqrt(kp_arm[i] / i_eff[i])
        print(
            f"{name:<28} {i_eff[i]:>10.4f} {omega_n:>9.2f} {zeta_current[i]:>9.3f}"
            f" {target_zeta:>9.2f} {kd_cur[i]:>7.1f} {kd_new[i]:>7.1f}"
        )

    tau_grav = g_max
    tau_kp = kp_arm * POS_ERR_MAX
    tau_kd = kd_new * vel_max
    tau_total = tau_grav + tau_kp + tau_kd
    ratio = tau_total / EFFORT_LIMITS

    hdr2 = (
        f"{'Joint':<28} {'tau_grav':>9} {'tau_Kp':>8} {'tau_Kd':>8}"
        f" {'tau_tot':>8} {'limit':>6} {'ratio':>6} {'OK?':>5}"
    )
    print("\n" + "-" * len(hdr2))
    print(
        f"  TORQUE HEADROOM  (worst-case: |g|_max + Kp*{np.degrees(POS_ERR_MAX):.0f}deg"
        f" + Kd*{vel_max:g}rad/s)"
    )
    print("-" * len(hdr2))
    print(hdr2)
    print("-" * len(hdr2))
    over_joints = []
    for i, name in enumerate(joints):
        over = ratio[i] > 1.0
        if over:
            over_joints.append((side, name, tau_total[i], EFFORT_LIMITS[i], ratio[i]))
        print(
            f"{name:<28} {tau_grav[i]:>9.2f} {tau_kp[i]:>8.2f} {tau_kd[i]:>8.2f}"
            f" {tau_total[i]:>8.2f} {EFFORT_LIMITS[i]:>6.1f} {ratio[i]:>6.2f}"
            f" {'OVER' if over else 'OK':>5}"
        )
    return kd_new, over_joints


def analyze(
    target_zeta: float = 0.7, samples: int = 10000, config_path: Path = CONFIG_PATH
) -> None:
    model, data = build_model()
    kp_by_side, kd_by_side, kd_full = load_gains(config_path)
    vel_max = load_vel_max(config_path)
    print(f"Loaded gains from {config_path} (joint speed limit {vel_max:g} rad/s)")

    kd_left, over_l = analyze_arm(
        model,
        data,
        "left",
        kp_by_side["left"],
        kd_by_side["left"],
        target_zeta,
        samples,
        vel_max,
    )
    kd_right, over_r = analyze_arm(
        model,
        data,
        "right",
        kp_by_side["right"],
        kd_by_side["right"],
        target_zeta,
        samples,
        vel_max,
    )

    # ── YAML snippet: keep legs/waist from config, swap in the new arm Kds ─────
    kd_full = kd_full.copy()
    kd_full[_ARM_SLICE["left"]] = kd_left
    kd_full[_ARM_SLICE["right"]] = kd_right

    def _row(vals):
        return ", ".join(f"{v:.1f}" for v in vals)

    print("\n" + "=" * 60)
    print("  PROPOSED gear_wbc.yaml  motor_kds  (motor_kps + legs/waist unchanged)")
    print("=" * 60)
    print(f"motor_kds: [{_row(kd_full[0:6])}, # left_leg")
    print(f"  {_row(kd_full[6:12])}, # right_leg")
    print(f"  {_row(kd_full[12:15])}, # waist")
    print(f"  {_row(kd_full[15:22])}, # left_arm")
    print(f"  {_row(kd_full[22:29])}] # right_arm")

    # ── Torque-limit log: warn if any joint exceeds its effort limit ──────────
    over = over_l + over_r
    if over:
        print("\n" + "!" * 60)
        print(f"  WARNING: {len(over)} joint(s) exceed the torque limit at the")
        print("  worst case (max gravity pose + max tracking error + max speed):")
        for side, name, tau_tot, limit, ratio in over:
            print(
                f"    [{side:>5}] {name:<26} tau={tau_tot:6.2f} Nm"
                f"  limit={limit:4.1f} Nm  ratio={ratio:4.2f}"
            )
        print("!" * 60)
        print(
            "\nNote: this is the SIMULTANEOUS worst case (max gravity pose + max\n"
            "tracking error + max speed), which rarely coincides in practice, and\n"
            "gravity is applied as feed-forward by the controller. Treat a ratio\n"
            "slightly above 1.0 as a saturation-margin warning; a large ratio\n"
            "(especially on the 5 Nm wrist motors) means back off that joint's Kp."
        )
    else:
        print("\nAll joints within torque limits at the worst case (ratio <= 1.0).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zeta",
        type=float,
        default=0.7,
        help="Target damping ratio (default 0.7). Use 1.0 for critical damping.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Joint-space samples for the worst-case inertia/gravity search.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help=f"gear_wbc.yaml to read motor_kps/motor_kds from (default: {CONFIG_PATH}).",
    )
    args = parser.parse_args()
    analyze(target_zeta=args.zeta, samples=args.samples, config_path=args.config)
