import struct

import numpy as np


def parse_buttons(key_state_buffer: bytes) -> dict:
    """
    Transform RC byte array into the corresponding button presses.

    The RC maps the buttons to two bytes in the key state buffer. Each bit in the byte
    corresponds to a specific button. The order is as follows:

    [F3, F1, L2, R2, SELECT, START, L1, R1]

    [DLEFT, DDOWN, DRIGHT, DUP, Y, X, B, A]

    where the left most bit is the most significant bit.

    Args:
        key_state_buffer: bytearray read from the RC
    Returns:
        Dict where keys correspond to the respective RC buttons and the values to whether
        they are pressed.
    """

    # Button groups (2 bytes)
    button_group1 = key_state_buffer[2]
    button_group2 = key_state_buffer[3]

    # Decode button bits back to binary (assuming base-2 packed string)
    button_bits_1 = f"{button_group1:08b}"
    button_bits_2 = f"{button_group2:08b}"

    buttons_1 = {
        "F3": int(button_bits_1[0]),
        "F1": int(button_bits_1[1]),
        "L2": int(button_bits_1[2]),
        "R2": int(button_bits_1[3]),
        "SELECT": int(button_bits_1[4]),
        "START": int(button_bits_1[5]),
        "L1": int(button_bits_1[6]),
        "R1": int(button_bits_1[7]),
    }

    buttons_2 = {
        "DLEFT": int(button_bits_2[0]),
        "DDOWN": int(button_bits_2[1]),
        "DRIGHT": int(button_bits_2[2]),
        "DUP": int(button_bits_2[3]),
        "Y": int(button_bits_2[4]),
        "X": int(button_bits_2[5]),
        "B": int(button_bits_2[6]),
        "A": int(button_bits_2[7]),
    }
    return {**buttons_1, **buttons_2}


def parse_axes(key_state_buffer: bytes) -> dict:
    """
    Extract RC axes from the RC byte array.
    """

    # Axes (floats)
    lx = struct.unpack("f", key_state_buffer[4:8])[0]
    rx = struct.unpack("f", key_state_buffer[8:12])[0]
    ry = struct.unpack("f", key_state_buffer[12:16])[0]
    ly = struct.unpack("f", key_state_buffer[20:24])[0]
    return {"LX": lx, "RX": rx, "RY": ry, "LY": ly}


def get_velocity_commands(key_state_buffer: bytes, gains) -> np.ndarray:
    """Extract velocity commands from RC axis."""

    input = parse_axes(key_state_buffer)
    vy = -1 * input["LX"] * gains["vy"]
    vx = input["LY"] * gains["vx"]
    vyaw = -1 * input["RX"] * gains["vyaw"]
    vz = input["RY"] * gains["vz"]
    return np.array([vx, vy, vyaw]), np.array([vz])  # (vx, vy, vyaw), (vz)
