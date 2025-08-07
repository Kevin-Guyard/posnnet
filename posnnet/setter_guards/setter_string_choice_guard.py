from functools import wraps
from typing import get_type_hints, Callable, Self, List


def setter_string_choice_guard(allowed_choices: List[str]) -> Callable:

    """
    Add a guard to the setter that checks that the provided parameter value is a string from a list of allowed choices.

    Args:
        - allowed_choices (List[str]): The list of allowed choices.

    Returns:
        - setter_string_choice_guarded (Callable): The setter function with a value guard.

    Raises:
        - ValueError: If the input is a string not in the allowed choices list.
    """

    def setter_string_choice_guard_low_level(setter_function: Callable) -> Callable:

        @wraps(setter_function)
        def setter_string_choice_guarded(*args: Self, **kwargs: str) -> None:

            # Get the type hints to get the param_name
            type_hints = get_type_hints(setter_function)
            param_name = list(type_hints.keys())[1]
    
            # Get the param value
            param_value = args[1]

            # Check that the value is one of the allowed choices.
            if not param_value in allowed_choices:
                raise ValueError(f"The parameter '{param_name:s}' has to be one of the available choices ({allowed_choices}) but you provided the value {param_value:s}.")

            # Call the setter function.
            setter_function(*args, **kwargs)

        return setter_string_choice_guarded

    return setter_string_choice_guard_low_level