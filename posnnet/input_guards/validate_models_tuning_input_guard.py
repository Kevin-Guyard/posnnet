from functools import wraps
from typing import Callable, Any, Union, Self


def validate_models_tuning_input_guard(validate_models_tuning: Callable) -> Callable:

    """
    Add a guard to validate_models_tuning method of Project.

    Args:
        - validate_models_tuning (Callable): The validate_models_tuning method.

    Returns:
        - validate_models_tuning_input_guarded (Callable): The validate_models_tuning method guarded.

    Raises:
        - TypeError: If the input parameter types are not the one waited.
    """

    @wraps(validate_models_tuning)
    def validate_models_tuning_input_guarded(
        self: Self,
        force_validation: bool=False
    ) -> None:

        # Check force_validation type.
        if not isinstance(force_validation, bool):
            raise TypeError(f"The parameter 'force_validation' must be a boolean but you provided a {type(force_validation).__name__:s} instead.")

        # Call the project method run_tuning
        validate_models_tuning(
            self=self,
            force_validation=force_validation
        )

    return validate_models_tuning_input_guarded