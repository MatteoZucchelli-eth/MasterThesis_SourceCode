"""
Minimal UDP receiver — prints incoming joystick/pose packets from the VR headset.
No robot SDK required.

Run:
    python udp_receiver.py            # listens on 0.0.0.0:9870
    python udp_receiver.py --port 9871
"""

import argparse
import socket
import struct

import numpy as np

MAGIC = b"\xab\xcd"
BASE_FMT = "<2sBB16f16f16f8fH"
BASE_SIZE = struct.calcsize(BASE_FMT)

BTN_NAMES = [
    "L-trigger",
    "L-squeeze",
    "L-A/X",
    "L-B/Y",
    "L-stick",
    "R-trigger",
    "R-squeeze",
    "R-A",
    "R-B",
    "R-stick",
]


def decode(data):
    if len(data) < BASE_SIZE:
        print(f"  [!] packet too short: {len(data)} B")
        return
    f = struct.unpack_from(BASE_FMT, data)
    if f[0] != MAGIC:
        print(f"  [!] bad magic: {f[0].hex()}")
        return
    head = np.array(f[3:19], dtype=np.float32).reshape(4, 4)
    left = np.array(f[19:35], dtype=np.float32).reshape(4, 4)
    right = np.array(f[35:51], dtype=np.float32).reshape(4, 4)
    l_trig, r_trig = f[51], f[52]
    l_sqz, r_sqz = f[53], f[54]
    l_sx, l_sy = f[55], f[56]
    r_sx, r_sy = f[57], f[58]
    btns = int(f[59])

    pressed = [BTN_NAMES[i] for i in range(10) if btns & (1 << i)]

    print(
        f"  head pos: ({head[0,3]:.3f}, {head[1,3]:.3f}, {head[2,3]:.3f})"
        f"  |  L palm: ({left[0,3]:.3f}, {left[1,3]:.3f}, {left[2,3]:.3f})"
        f"  |  R palm: ({right[0,3]:.3f}, {right[1,3]:.3f}, {right[2,3]:.3f})"
        f"\n  L trigger={l_trig:.1f}  squeeze={l_sqz:.2f}  stick=({l_sx:.2f},{l_sy:.2f})"
        f"  |  R trigger={r_trig:.1f}  squeeze={r_sqz:.2f}  stick=({r_sx:.2f},{r_sy:.2f})"
        + (f"\n  buttons: {', '.join(pressed)}" if pressed else "")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9870)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    print(f"Listening on {args.host}:{args.port}  (Ctrl-C to stop)")

    pkt_count = 0
    try:
        while True:
            data, addr = sock.recvfrom(2048)
            pkt_count += 1
            print(f"[#{pkt_count}] from {addr}  ({len(data)} B)")
            decode(data)
    except KeyboardInterrupt:
        print(f"\nDone. Received {pkt_count} packets.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
