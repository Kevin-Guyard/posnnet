from functools import wraps
from typing import get_type_hints, Callable, Self


def setter_strictly_positive_integer_guard(setter_function: Callable) -> Callable:

    """
    Add a guard to the setter that checks that the provided parameter value is a strictly positive integer.

    Args:
        - setter_function (Callable): The setter function.

    Returns:
        - setter_strictly_positive_integer_guarded (Callable): The setter function with a value guard.

    Raises:
        - ValueError: If the input is not a strictly positive integer.
    """

    @wraps(setter_function)
    def setter_strictly_positive_integer_guarded(*args: Self, **kwargs: int) -> None:

        # Get the type hints to get the param_name
        type_hints = get_type_hints(setter_function)
        param_name = list(type_hints.keys())[1]

        # Get the param value
        param_value = args[1]

        # Check that the value is a strictly positive integer.
        if not param_value > 0:
            raise ValueError(f"The parameter '{param_name:s}' should be a strictly positive integer (0 excluded) but you provided the value {param_value:d}.")

        # Call the setter function
        setter_function(*args, **kwargs)

    return setter_strictly_positive_integer_guarded