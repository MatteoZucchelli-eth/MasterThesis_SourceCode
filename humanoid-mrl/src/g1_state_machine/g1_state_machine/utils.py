def get_fsm_state_from_buttons(buttons, config):
    """
    Given current button states and trigger config, determine FSM state.

    Args:
        buttons (dict): current button states
        config (dict): state name -> button combo (from YAML)

    Returns:
        str or None: FSM state name if a trigger matches, else None
    """
    pressed = {btn for btn, state in buttons.items() if state}
    for key, value in config.items():
        combo = set(value)
        if combo.issubset(pressed):
            return key
    return None


def print_transition_info(config, logger):
    """Print the transition to RC mappings."""
    lines = ["\n\n######## Transition to RC Mappings ########\n"]

    for state, buttons in config.items():
        button_str = " + ".join(buttons)
        lines.append(f"  {state:<10}: {button_str}")

    lines.append("\n###########################################\n")
    output = "\n".join(lines)
    logger.info(output)
