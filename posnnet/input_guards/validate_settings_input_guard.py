from functools import wraps
from typing import Callable, Any, Union, Self


def validate_settings_input_guard(validate_settings: Callable) -> Callable:

    """
    Add a guard to validate_settings method of Project.

    Args:
        - validate_settings (Callable): The validate_settings method.

    Returns:
        - validate_settings_input_guarded (Callable): The validate_settings method guarded.

    Raises:
        - TypeError: If the input parameter types are not the one waited.
    """

    @wraps(validate_settings)
    def validate_settings_input_guarded(
        self: Self,
        force_validation: bool=False
    ) -> None:

        # Check force_validation type.
        if not isinstance(force_validation, bool):
            raise TypeError(f"The parameter 'force_validation' must be a boolean but you provided a {type(force_validation).__name__:s} instead.")

        # Call the project method run_tuning
        validate_settings(
            self=self,
            force_validation=force_validation
        )

    return validate_settings_input_guarded