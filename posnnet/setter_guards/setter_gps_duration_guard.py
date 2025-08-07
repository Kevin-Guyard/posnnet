from collections import Counter
from functools import wraps
from typing import get_type_hints, Callable, Self, List, Tuple


def setter_gps_duration_guard(setter_function: Callable) -> Callable:

    """
    Add a guard to the setter that checks that the provided parameter value list is composed by GPS outage duration.

    NB: A GPS outage duration is ("MM:SS", "MM:SS") where the first timestamp should be superior to zero and inferior to the second.

    Args:
        - setter_function (Callable): The setter function.

    Returns:
        - setter_gps_duration_guarded (Callable): The setter function with a value guard.

    Raises:
        - ValueError: If the input list is not composed by strictly positive integer.
    """

    @wraps(setter_function)
    def setter_gps_duration_guarded(*args: Self, **kwargs: List[Tuple[str, str]]) -> None:

        # Get the type hints to get the param_name
        type_hints = get_type_hints(setter_function)
        param_name = list(type_hints.keys())[1]

        # Get the param value
        param_value = args[1]

        # Check that no duration is duplicated.
        if len(param_value) != len(set(param_value)):
            raise ValueError(f"The parameter '{param_name:s}' contains duplicated values: {', '.join([str(item) for item, count in Counter(param_value).items() if count > 1])}.")

        for duration in param_value:

            # Check that the duration follow the structure "MM:SS".
            if not all([
                len(timestamp) == 5 and timestamp[0:2].isdigit() and timestamp[2] == ":" and timestamp[3:5].isdigit() and (0 <= int(timestamp[0:2]) < 60) and (0 <= int(timestamp[3:5]) < 60)
                for timestamp in duration
            ]):
                raise ValueError(f"The parameter '{param_name:s}' should be a list of duration [('MM:SS', 'MM:SS'), ...] but received at least one element which is not a duration: {duration}.")

            # Check that the minimum is superior to 0.
            if duration[0][0:2] == "00" and duration[0][3:5] == "00":
                raise ValueError(f"The parameter '{param_name:s}' contains at least one duration with a minimal time null: {duration}.")

            # Check that the minimum is inferior to the maximum (duratiion non null).
            if not 60 * int(duration[0][0:2]) + int(duration[0][3:5]) < 60 * int(duration[1][0:2]) + int(duration[1][3:5]):
                raise ValueError(f"The parameter '{param_name}' contains at least one duration with minimal non inferior to maximal: {duration}.")

        # Call the setter function.
        setter_function(*args, **kwargs)

    return setter_gps_duration_guarded