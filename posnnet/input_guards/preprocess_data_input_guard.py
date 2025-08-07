from functools import wraps
from typing import Callable, Any, Union, Self


def preprocess_data_input_guard(preprocess_data: Callable) -> Callable:

    """
    Add a guard to preprocess_data method of Project.

    Args:
        - preprocess_data (Callable): The preprocess_data method.

    Returns:
        - preprocess_data_input_guarded (Callable): The preprocess_data method guarded.

    Raises:
        - TypeError: If the input parameter types are not the one waited.
        - ValueError: If the input parameter values are not in the range / choices waited.
    """

    @wraps(preprocess_data)
    def preprocess_data_input_guarded(
        self: Self,
        verbosity: bool=False
    ) -> None:

        # Check verbosity type.
        if not isinstance(verbosity, bool):
            raise TypeError(f"The parameter 'verbosity' must be a boolean but you provided a {type(verbosity).__name__:s} instead.")

        # Call the project method run_tuning
        preprocess_data(
            self=self,
            verbosity=verbosity
        )

    return preprocess_data_input_guarded