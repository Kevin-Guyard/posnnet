from collections import Counter
from functools import wraps
from typing import get_type_hints, Callable, Self


def setter_list_of_unique_strictly_positive_integer_guard(setter_function: Callable) -> Callable:

    """
    Add a guard to the setter that checks that the provided parameter value list is composed by unique strictly positive integers.

    Args:
        - setter_function (Callable): The setter function.

    Returns:
        - setter_list_of_unique_strictly_positive_integer_guarded (Callable): The setter function with a value guard.

    Raises:
        - ValueError: If the input list is not composed by strictly positive integer.
    """

    @wraps(setter_function)
    def setter_list_of_unique_strictly_positive_integer_guarded(*args: Self, **kwargs: int) -> None:

        # Get the type hints to get the param_name
        type_hints = get_type_hints(setter_function)
        param_name = list(type_hints.keys())[1]

        # Get the param value
        param_value = args[1]

        # Check that every value is a strictly positive integer.
        for item in param_value:
            if not item > 0:
                raise ValueError(f"The parameter '{param_name:s}' should be a list of unique strictly positive integer (0 excluded) but the list you provided contains the value {item:d}.")

        if len(param_value) != len(set(param_value)):
            raise ValueError(f"The parameter '{param_name:s}' should be a list of unique strictly positive integer (0 excluded) but the list you provided contains duplicated values: [{', '.join([f"{item:d}" for item, count in Counter(param_value).items() if count > 1])}].")

        # Call the setter function
        setter_function(*args, **kwargs)

    return setter_list_of_unique_strictly_positive_integer_guarded