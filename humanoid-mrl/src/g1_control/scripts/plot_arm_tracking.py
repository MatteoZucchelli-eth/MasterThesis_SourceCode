#!/usr/bin/env python3
"""Plot commanded-vs-measured right-arm tracking from a /lowcmd + /lowstate bag.

Separates command-side from servo-side oscillation.

    ros2 bag record /lowcmd /lowstate      # then:
    python3 scripts/plot_arm_tracking.py <bag_dir>

Reading the output, per joint:

* ``q_cmd`` jagged                    -> command side (IK or interpolator).
* ``q_cmd`` smooth, ``q_meas`` jagged -> servo side; the error spectrum says which:
    - peak near sqrt(Kp/I_eff)/2pi (omega_n in tune_arm_gains.py) -> Kd too low.
    - broadband up to the command Nyquist, torque noise tracking ``kd * dq_cmd``
      -> reference-velocity path.
    - sawtooth in q_err, amplitude ~= tau_stiction/Kp -> stick-slip, Kp too low.

``tau_est`` is backend-dependent (Isaac: joint force projected on the DOF;
MuJoCo: actuator force). State (500 Hz) is linearly resampled onto command
(50 Hz) bag receive stamps.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ── Joint layout ─────────────────────────────────────────────────────────────
try:
    from g1_definitions import definitions
except ImportError:  # pragma: no cover - depends on the workspace being sourced
    sys.exit(
        "Cannot import g1_definitions. Source the workspace first:\n"
        "  source install/setup.bash"
    )

SHORT_NAMES = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)


def read_bag(bag_dir: Path, cmd_topic: str, state_topic: str):
    """Return (cmd, state) dicts of stacked per-motor arrays, PR order.

    Each dict holds ``t`` (seconds from the first message of either stream) plus
    (N, 29) arrays; the command dict also carries the live ``kp``/``kd``.
    """
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    # storage_id "" lets rosbag2 sniff sqlite3 vs mcap from the metadata.
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=""),
        rosbag2_py.ConverterOptions("", ""),
    )
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    for topic in (cmd_topic, state_topic):
        if topic not in types:
            sys.exit(
                f"Topic {topic} not in bag. Present: {sorted(types)}\n"
                "Record with: ros2 bag record /lowcmd /lowstate"
            )
    # Filter in the storage layer to skip unrelated (e.g. camera) topics.
    reader.set_filter(rosbag2_py.StorageFilter(topics=[cmd_topic, state_topic]))
    cmd_cls = get_message(types[cmd_topic])
    state_cls = get_message(types[state_topic])

    n = definitions.NUM_MOTOR_G1_BASE
    raw = {cmd_topic: [], state_topic: []}
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        msg = deserialize_message(data, cmd_cls if topic == cmd_topic else state_cls)
        motors = (msg.motor_cmd if topic == cmd_topic else msg.motor_state)[:n]
        row = {
            "t": stamp * 1e-9,
            "q": [m.q for m in motors],
            "dq": [m.dq for m in motors],
            "tau": [m.tau for m in motors] if topic == cmd_topic
            else [m.tau_est for m in motors],
        }
        if topic == cmd_topic:
            row["kp"] = [m.kp for m in motors]
            row["kd"] = [m.kd for m in motors]
        raw[topic].append(row)

    if not raw[cmd_topic] or not raw[state_topic]:
        sys.exit(
            f"Bag has {len(raw[cmd_topic])} {cmd_topic} and "
            f"{len(raw[state_topic])} {state_topic} messages -- need both."
        )

    t0 = min(raw[cmd_topic][0]["t"], raw[state_topic][0]["t"])

    def stack(rows):
        out = {"t": np.array([r["t"] for r in rows]) - t0}
        for key in rows[0]:
            if key != "t":
                out[key] = np.array([r[key] for r in rows], dtype=float)
        return out

    return stack(raw[cmd_topic]), stack(raw[state_topic])


#: Frequencies below this [Hz] are trajectory trend, not oscillation.
_TREND_HZ = 0.2


def highpass(t: np.ndarray, x: np.ndarray, f_cut: float = _TREND_HZ) -> np.ndarray:
    """``x`` minus its centred moving average over ``1/f_cut`` seconds.

    Boxcar (no ringing); edges use a shrinking, normalised window.
    """
    dt = float(np.median(np.diff(t)))
    width = max(3, int(round(1.0 / (f_cut * dt))) | 1)  # odd, >= 3
    kernel = np.ones(width)
    norm = np.convolve(np.ones_like(x), kernel, mode="same")
    return x - np.convolve(x, kernel, mode="same") / norm


def uniform(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """``x`` on a uniform grid at the median tick, plus that tick.

    Command stamps jitter; resampling keeps the FFT peak sharp.
    """
    dt = float(np.median(np.diff(t)))
    return np.interp(np.arange(t[0], t[-1], dt), t, x), dt


def dominant_freq(t: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """(frequency, amplitude) of the largest component of ``x`` above _TREND_HZ.

    Detrended (see :func:`highpass`) and band-limited.
    """
    if len(t) < 16 or np.ptp(x) < 1e-6:
        # Parked joint: no meaningful peak.
        return float("nan"), float("nan")
    xi, dt = uniform(t, highpass(t, x))
    spec = np.abs(np.fft.rfft(xi * np.hanning(len(xi))))
    freqs = np.fft.rfftfreq(len(xi), dt)
    band = freqs >= _TREND_HZ
    if band.sum() < 1:
        return float("nan"), float("nan")
    k = int(np.argmax(spec[band]))
    # Hann window halves coherent gain; x4/N recovers a single-tone amplitude.
    return float(freqs[band][k]), float(spec[band][k] * 4.0 / len(xi))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("bag", type=Path, help="rosbag2 directory (or .mcap file)")
    ap.add_argument(
        "--arm", choices=("right", "left"), default="right", help="which arm to plot"
    )
    ap.add_argument("--cmd-topic", default=definitions.TOPIC_LOWCMD)
    ap.add_argument("--state-topic", default=definitions.TOPIC_LOWSTATE)
    ap.add_argument(
        "--start", type=float, default=0.0, help="crop: seconds from bag start"
    )
    ap.add_argument("--end", type=float, default=np.inf, help="crop: seconds")
    ap.add_argument("--out", type=Path, help="save PNG(s) here instead of showing")
    args = ap.parse_args()

    if not args.bag.exists():
        sys.exit(f"No such bag: {args.bag}")

    import matplotlib

    if args.out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmd, state = read_bag(args.bag, args.cmd_topic, args.state_topic)
    lo, hi = definitions.JOINT_PR_SLICES[f"{args.arm}_arm"]

    keep = (cmd["t"] >= args.start) & (cmd["t"] <= args.end)
    if keep.sum() < 2:
        sys.exit(f"Crop [{args.start}, {args.end}] keeps {keep.sum()} command samples.")
    t = cmd["t"][keep]

    # State onto command timestamps; np.interp needs sorted x.
    order = np.argsort(state["t"])
    ts = state["t"][order]

    def on_cmd_grid(arr):
        return np.column_stack(
            [np.interp(t, ts, arr[order, j]) for j in range(lo, hi)]
        )

    q_cmd, dq_cmd, tau_ff = (cmd[k][keep, lo:hi] for k in ("q", "dq", "tau"))
    kp, kd = (cmd[k][keep, lo:hi] for k in ("kp", "kd"))
    q_meas, dq_meas, tau_est = (on_cmd_grid(state[k]) for k in ("q", "dq", "tau"))

    q_err = q_cmd - q_meas
    dq_err = dq_cmd - dq_meas
    # Compare the computed command with the backend-dependent tau_est signal.
    tau_total = tau_ff + kp * q_err + kd * dq_err

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{args.bag}  |  {args.arm} arm  |  {len(t)} cmd ticks over "
          f"{t[-1] - t[0]:.1f} s "
          f"(cmd {1.0 / np.median(np.diff(t)):.0f} Hz, "
          f"state {1.0 / np.median(np.diff(ts)):.0f} Hz)")
    print(
        f"\n{'joint':<16}{'Kp':>6}{'Kd':>6}"
        f"{'err mean':>10}{'err p2p':>9}{'osc rms':>9}"
        f"{'f_err':>8}{'amp':>8}{'tau_ff':>8}{'tau_est':>8}{'ff gap':>8}"
    )
    print(f"{'':<16}{'':>6}{'':>6}{'[mrad]':>10}{'[mrad]':>9}{'[mrad]':>9}"
          f"{'[Hz]':>8}{'[mrad]':>8}{'[Nm]':>8}{'[Nm]':>8}{'[Nm]':>8}")
    print("-" * 106)
    for j, name in enumerate(SHORT_NAMES):
        f_hz, amp = dominant_freq(t, q_err[:, j])
        # Detrended: oscillation only.
        osc = float(np.std(highpass(t, q_err[:, j])))
        print(
            f"{name:<16}{kp[0, j]:>6.1f}{kd[0, j]:>6.2f}"
            f"{q_err[:, j].mean() * 1e3:>10.1f}"
            f"{np.ptp(q_err[:, j]) * 1e3:>9.1f}{osc * 1e3:>9.1f}"
            f"{f_hz:>8.2f}{amp * 1e3:>8.1f}"
            f"{tau_ff[:, j].mean():>8.2f}{tau_est[:, j].mean():>8.2f}"
            f"{(tau_est[:, j] - tau_ff[:, j]).mean():>8.2f}"
        )
    print(
        f"\nerr mean  -> steady droop. Times Kp, this IS the torque the FF failed to supply\n"
        f"err p2p   -> worst excursion over the window\n"
        f"osc rms   -> the oscillation alone, trajectory trend removed (>{_TREND_HZ} Hz)\n"
        f"f_err     -> compare to omega_n/2pi from tune_arm_gains.py: a match means Kd,\n"
        f"             broadband/near-Nyquist means the dq path, sawtooth means Kp\n"
        f"ff gap    -> tau_est - tau_ff; meaning depends on the active backend.\n"
        f"             Only meaningful at a settled hold -- crop with --start/--end."
    )

    # ── Figure 1: time series ─────────────────────────────────────────────────
    fig, axes = plt.subplots(
        len(SHORT_NAMES), 4, figsize=(19, 2.1 * len(SHORT_NAMES)), sharex=True
    )
    fig.suptitle(
        f"{args.arm} arm tracking -- {args.bag.name}"
        f"  (Kp {np.array2string(kp[0], precision=0, separator=',')},"
        f" Kd {np.array2string(kd[0], precision=2, separator=',')})",
        fontsize=11,
    )
    for j, name in enumerate(SHORT_NAMES):
        ax = axes[j, 0]
        ax.plot(t, q_cmd[:, j], lw=1.1, label="cmd")
        ax.plot(t, q_meas[:, j], lw=0.9, label="meas")
        ax.set_ylabel(f"{name}\n[rad]", fontsize=8)

        ax = axes[j, 1]
        ax.plot(t, q_err[:, j] * 1e3, lw=0.9, color="tab:red", label="cmd - meas")
        ax.axhline(0.0, lw=0.5, color="k", alpha=0.4)
        ax.set_ylabel("[mrad]", fontsize=8)

        ax = axes[j, 2]
        ax.plot(t, dq_cmd[:, j], lw=1.1, label="dq cmd (ref)")
        ax.plot(t, dq_meas[:, j], lw=0.7, alpha=0.8, label="dq meas")
        ax.set_ylabel("[rad/s]", fontsize=8)

        ax = axes[j, 3]
        ax.plot(t, tau_ff[:, j], lw=1.1, label="tau_ff (gravity)")
        ax.plot(t, tau_total[:, j], lw=0.9, label="tau_ff + PD")
        ax.plot(t, tau_est[:, j], lw=0.7, alpha=0.8, label="tau_est")
        ax.set_ylabel("[Nm]", fontsize=8)

        for col in range(4):
            axes[j, col].grid(alpha=0.25, lw=0.4)
            axes[j, col].tick_params(labelsize=7)
    for col, title in enumerate(
        ("position", "position error", "velocity", "torque")
    ):
        axes[0, col].set_title(title, fontsize=9)
        axes[0, col].legend(fontsize=6, loc="best")
        axes[-1, col].set_xlabel("t [s]", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    # ── Figure 2: error spectra ───────────────────────────────────────────────
    # Log-log so a second-order peak is distinguishable from a broadband floor.
    fig2, axes2 = plt.subplots(
        len(SHORT_NAMES), 1, figsize=(8, 1.7 * len(SHORT_NAMES)), sharex=True
    )
    fig2.suptitle(
        f"{args.arm} arm position-error spectrum (detrended >{_TREND_HZ} Hz)",
        fontsize=11,
    )
    dt = float(np.median(np.diff(t)))
    grid = np.arange(t[0], t[-1], dt)
    for j, name in enumerate(SHORT_NAMES):
        xi = np.interp(grid, t, highpass(t, q_err[:, j]))
        xi = (xi - xi.mean()) * np.hanning(len(xi))
        spec = np.abs(np.fft.rfft(xi)) * 4.0 / len(xi)
        freqs = np.fft.rfftfreq(len(xi), dt)
        ax = axes2[j]
        # Mask non-positive bins (a parked joint's spectrum is all zero).
        good = spec[1:] > 0.0
        if good.any():
            ax.loglog(freqs[1:][good], spec[1:][good] * 1e3, lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(f"{name}\n[mrad]", fontsize=8)
        ax.grid(alpha=0.25, which="both", lw=0.4)
        ax.tick_params(labelsize=7)
    axes2[-1].set_xlabel("frequency [Hz]  (Nyquist = cmd rate / 2)", fontsize=8)
    fig2.tight_layout(rect=(0, 0, 1, 0.98))

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for fig_obj, tag in ((fig, "tracking"), (fig2, "spectrum")):
            path = args.out / f"{args.bag.name}_{args.arm}_{tag}.png"
            fig_obj.savefig(path, dpi=120)
            print(f"wrote {path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
