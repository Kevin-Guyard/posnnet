from functools import wraps
from typing import get_type_hints, Callable, Self


def setter_strictly_positive_integer_or_none_guard(setter_function: Callable) -> Callable:

    """
    Add a guard to the setter that checks that the provided parameter value is either a strictly positive integer or None.

    Args:
        - setter_function (Callable): The setter function.

    Returns:
        - setter_strictly_positive_integer_or_none_guarded (Callable): The setter function with a value guard.

    Raises:
        - ValueError: If the input is not a strictly positive integer and not None.
    """

    @wraps(setter_function)
    def setter_strictly_positive_integer_or_none_guarded(*args: Self, **kwargs: int) -> None:

        # Get the type hints to get the param_name
        type_hints = get_type_hints(setter_function)
        param_name = list(type_hints.keys())[1]

        # Get the param value
        param_value = args[1]

        # Check that the value is a strictly positive integer.
        if param_value is not None and not param_value > 0:
            raise ValueError(f"The parameter '{param_name:s}' should be a strictly positive integer (0 excluded) but you provided the value {param_value:d}.")

        # Call the setter function
        setter_function(*args, **kwargs)

    return setter_strictly_positive_integer_or_none_guarded