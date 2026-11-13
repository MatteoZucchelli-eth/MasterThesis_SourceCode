import struct
from abc import ABC, abstractmethod
from os import path
from time import sleep

import pygame


class InputDeviceBase(ABC):
    """
    Abstract input device: packs button/axis state into the low-level wireless_remote
    buffer.
    """

    @abstractmethod
    def read_state(self) -> list[int]:
        """
        Read the current state of the input device and return it as a byte buffer.

        Returns:
            list: A list of bytes representing the input state.
        """
        pass


class Joystick(InputDeviceBase):
    """
    Interface for handling joystick inputs via Pygame.

    Original Source:
    https://github.com/unitreerobotics/unitree_sdk2_python/blob/a035adeaa6f8ea171bef9a43e8477abb87a0b35e/unitree_sdk2py/utils/joystick.py#L65-L182

    Args:
        js_type (str): The type of joystick mapping to use. Currently only supports "logi".
        device_id (int): The system device index of the joystick to open (default 0).
    """

    def __init__(self, js_type="logi", device_id=0):
        pygame.init()
        pygame.joystick.init()
        self.joystick = None
        self._setup(device_id, js_type)

    def _setup(self, device_id, js_type):
        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            print("No joystick detected.")
            return
        self.joystick = pygame.joystick.Joystick(device_id)
        self.joystick.init()

        if js_type == "logi":
            self.axis_id = {
                "LX": 0,  # Left stick axis x
                "LY": 1,  # Left stick axis y
                "L2": 2,  # L2 (aka LB), this is a button which is interpreted as an axis
                "RX": 3,  # Right stick axis x
                "RY": 4,  # Right stick axis y
                "R2": 5,  # R2 (aka RB), this is a button which is interpreted as an axis
            }
            self.button_id = {
                "A": 0,
                "B": 1,
                "X": 2,
                "Y": 3,
                "L1": 4,
                "R1": 5,
                "SELECT": 6,
                "START": 7,
                "LThumb": 9,
                "RThumb": 10,
            }
        else:
            raise ValueError("Unsupported joystick type")

    def read_state(self) -> list[int]:
        if self.joystick is None:
            return 40 * [0]
        key_state_buffer = 40 * [0]
        if self.joystick is not None:
            pygame.event.get()
            key_state_buffer[2] = int(
                "".join(
                    [
                        f"{key}"
                        for key in [
                            0,
                            0,
                            int(self.joystick.get_axis(self.axis_id["L2"]) > 0),
                            int(self.joystick.get_axis(self.axis_id["R2"]) > 0),
                            int(self.joystick.get_button(self.button_id["SELECT"])),
                            int(self.joystick.get_button(self.button_id["START"])),
                            int(self.joystick.get_button(self.button_id["L1"])),
                            int(self.joystick.get_button(self.button_id["R1"])),
                        ]  # most significant bit first
                    ]
                ),
                2,
            )
            key_state_buffer[3] = int(
                "".join(
                    [
                        f"{key}"
                        for key in [
                            int(self.joystick.get_hat(0)[0] < 0),  # left
                            int(self.joystick.get_hat(0)[1] < 0),  # down
                            int(self.joystick.get_hat(0)[0] > 0),  # right
                            int(self.joystick.get_hat(0)[1] > 0),  # up
                            int(self.joystick.get_button(self.button_id["Y"])),  # Y
                            int(self.joystick.get_button(self.button_id["X"])),  # X
                            int(self.joystick.get_button(self.button_id["B"])),  # B
                            int(self.joystick.get_button(self.button_id["A"])),  # A
                        ]  # most significant bit first
                    ]
                ),
                2,
            )

        # Axes
        sticks = [
            self.joystick.get_axis(self.axis_id["LX"]),
            self.joystick.get_axis(self.axis_id["RX"]),
            -self.joystick.get_axis(self.axis_id["RY"]),
            -self.joystick.get_axis(self.axis_id["LY"]),
        ]
        packs = list(map(lambda x: struct.pack("f", x), sticks))
        key_state_buffer[4:8] = packs[0]
        key_state_buffer[8:12] = packs[1]
        key_state_buffer[12:16] = packs[2]
        key_state_buffer[20:24] = packs[3]

        return key_state_buffer


class NullInputDevice(InputDeviceBase):
    """No-op input device for headless runs: every button and axis reads zero."""

    def read_state(self) -> list[int]:
        return 40 * [0]


class Keyboard(InputDeviceBase):
    """
    Interface for handling keyboard inputs via pynput.
    """

    def __init__(self, logger=None):
        # Lazy import: pynput needs an X display.
        from pynput import keyboard

        self._keyboard = keyboard
        self.button_map = {
            # BUTTONS
            "A": "1",
            "B": "2",
            "X": "3",
            "Y": "4",
            "L1": "e",
            "R1": "u",
            "L2": "q",
            "R2": "o",
            "SELECT": keyboard.Key.shift_l,
            "START": keyboard.Key.shift_r,
            "F1": keyboard.Key.f1,
            "F3": keyboard.Key.f3,
            # AXIS
            "L_left": "a",
            "L_right": "d",
            "L_up": "w",
            "L_down": "s",
            "R_left": "j",
            "R_right": "l",
            "R_up": "i",
            "R_down": "k",
            "DUP": keyboard.Key.up,
            "DDOWN": keyboard.Key.down,
            "DLEFT": keyboard.Key.left,
            "DRIGHT": keyboard.Key.right,
        }

        self.log = logger
        self.reverse_button_map = {v: k for k, v in self.button_map.items()}
        self.key_state = {}
        for keycode in self.button_map.keys():
            self.key_state[keycode] = False

        self.device = keyboard.Listener(
            on_press=lambda key: self._button_event(key, is_press=True),
            on_release=lambda key: self._button_event(key, is_press=False),
        )
        self.device.start()

    def _button_event(self, key, is_press):
        if isinstance(key, self._keyboard.KeyCode):
            key = key.char

        if key not in self.reverse_button_map:
            return

        button_name = self.reverse_button_map[key]
        self.key_state[button_name] = is_press

    def read_state(self) -> list[int]:
        key_state_buffer = 40 * [0]

        # --- Button ---
        group1 = [
            int(self.key_state.get("F3", False)),
            int(self.key_state.get("F1", False)),
            int(self.key_state.get("L2", False)),
            int(self.key_state.get("R2", False)),
            int(self.key_state.get("SELECT", False)),
            int(self.key_state.get("START", False)),
            int(self.key_state.get("L1", False)),
            int(self.key_state.get("R1", False)),
        ]  # most significant bit first
        key_state_buffer[2] = int("".join(map(str, group1)), 2)

        group2 = [
            int(self.key_state.get("DLEFT", False)),
            int(self.key_state.get("DDOWN", False)),
            int(self.key_state.get("DRIGHT", False)),
            int(self.key_state.get("DUP", False)),
            int(self.key_state.get("Y", False)),
            int(self.key_state.get("X", False)),
            int(self.key_state.get("B", False)),
            int(self.key_state.get("A", False)),
        ]  # most significant bit first
        key_state_buffer[3] = int("".join(map(str, group2)), 2)

        # --- Axes ---
        def get_axis(neg_key, pos_key):
            return float(self.key_state.get(pos_key, False)) - float(
                self.key_state.get(neg_key, False)
            )

        sticks = [
            get_axis("L_left", "L_right"),  # LX
            get_axis("R_left", "R_right"),  # RX
            get_axis("R_down", "R_up"),  # RY
            get_axis("L_down", "L_up"),  # LY
        ]

        packs = list(map(lambda x: struct.pack("f", x), sticks))
        key_state_buffer[4:8] = packs[0]
        key_state_buffer[8:12] = packs[1]
        key_state_buffer[12:16] = packs[2]
        key_state_buffer[20:24] = packs[3]

        return key_state_buffer

    def print_mapping(self, logger=None):
        log_lines = ["\n\n######## Keyboard ########"]

        log_lines.append("\n--- Button Mapping ---")
        for key, value in self.button_map.items():
            if key == "L_left":
                log_lines.append("\n--- Axes Mapping ---")
            clean_val = value
            log_lines.append(f"  {key:<{10}} : {clean_val}")

        log_lines.append("\n##########################\n")
        output = "\n".join(log_lines)
        if logger:
            logger.info(output)
        else:
            print(output)


def setup_input_device(logger) -> InputDeviceBase:
    """
    Set up the input device for the simulation.

    Prefers a connected joystick, then the keyboard, then
    :class:`NullInputDevice` (headless, no X display).
    """

    if path.exists("/dev/input/js0"):
        joystick = Joystick(js_type="logi")
        sleep(0.1)
        joystick.read_state()
        return joystick

    try:
        keyboard = Keyboard()
    except Exception as e:  # noqa: BLE001 — pynput failure modes vary by backend
        if logger:
            logger.warning(
                f"Keyboard input unavailable ({e}); using a null input device. "
                "FSM transitions must be sent over the fsm_trigger topic."
            )
        return NullInputDevice()
    sleep(0.1)
    keyboard.print_mapping(logger)
    keyboard.read_state()
    return keyboard
