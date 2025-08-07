from functools import wraps
from typing import get_type_hints, Callable, Self


def setter_no_empty_string_or_none_guard(setter_function: Callable) -> Callable:

    """
    Add a guard to the setter that checks that the provided parameter value is either None or a none empty string.

    Args:
        - setter_function (Callable): The setter function.

    Returns:
        - setter_no_empty_string_or_none_guarded (Callable): The setter function with a value guard.

    Raises:
        - ValueError: If the input is not a None or an empty string
    """

    @wraps(setter_function)
    def setter_no_empty_string_or_none_guarded(*args: Self, **kwargs: str) -> None:

        # Get the type hints to get the param_name
        type_hints = get_type_hints(setter_function)
        param_name = list(type_hints.keys())[1]

        # Get the param value
        param_value = args[1]

        # Check that the value is not an empty string.
        if param_value is not None and param_value == "":
            raise ValueError(f"The parameter '{param_name:s}' should not be an empty string.")

        # Call the setter function.
        setter_function(*args, **kwargs)

    return setter_no_empty_string_or_none_guarded