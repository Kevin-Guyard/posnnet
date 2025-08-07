from functools import wraps
from typing import get_type_hints, Callable, Self, List


def setter_string_choice_or_strictly_positive_integer_guard(allowed_choices: List[str]) -> Callable:

    """
    Add a guard to the setter that checks that the provided parameter value is either a string from a list of allowed choices or a strictly positive integer.

    Args:
        - allowed_choices (List[str]): The list of allowed choices if the input is a string.

    Returns:
        - setter_string_choice_or_strictly_positive_integer_guard_low_level (Callable): The setter function with a value guard.

    Raises:
        - ValueError: If the input is a string not in the allowed choices list or a non strictly positive integer.
    """

    def setter_string_choice_or_strictly_positive_integer_guard_low_level(setter_function: Callable) -> Callable:

        @wraps(setter_function)
        def setter_string_choice_or_strictly_positive_integer_guarded(*args: Self, **kwargs: str) -> None:

            # Get the type hints to get the param_name
            type_hints = get_type_hints(setter_function)
            param_name = list(type_hints.keys())[1]
    
            # Get the param value
            param_value = args[1]

            # Check that the value is one of the allowed choices or a strictly positive integer.
            if not (isinstance(param_value, int) and param_value > 0) and not (isinstance(param_value, str) and param_value in allowed_choices):
                raise ValueError(f"The parameter '{param_name:s}' has to be one of the available choices ({allowed_choices}) or a strictly positive integer (0 excluded) but you provided the value {param_value}.")

            # Call the setter function.
            setter_function(*args, **kwargs)

        return setter_string_choice_or_strictly_positive_integer_guarded

    return setter_string_choice_or_strictly_positive_integer_guard_low_level