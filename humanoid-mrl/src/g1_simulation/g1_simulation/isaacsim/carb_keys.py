# MIT License
"""Helper for resolving configurable keyboard keys to carb input codes."""
from __future__ import annotations

import logging


def key_target(key_name: str):
    """Resolve a key name ("R", "7", ...) to a carb.input.KeyboardInput, or None."""
    import carb.input

    name = key_name if key_name.isalpha() else f"KEY_{key_name}"
    target = getattr(carb.input.KeyboardInput, name, None)
    if target is None:
        logging.warning(f"Could not find keyboard binding for: {name}")
    return target


def is_key_press(e, key_name: str) -> bool:
    """True when the carb keyboard event ``e`` is a press of ``key_name``."""
    from carb.input import KeyboardEventType

    return e.type == KeyboardEventType.KEY_PRESS and e.input == key_target(key_name)
