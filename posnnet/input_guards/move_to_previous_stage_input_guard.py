from functools import wraps
from typing import Callable, Any, Union, Self


def move_to_previous_stage_input_guard(move_to_previous_stage: Callable) -> Callable:

    """
    Add a guard to move_to_previous_stage method of Project.

    Args:
        - move_to_previous_stage (Callable): The move_to_previous_stage method.

    Returns:
        - move_to_previous_stage_input_guarded (Callable): The move_to_previous_stage method guarded.

    Raises:
        - TypeError: If the input parameter types are not the one waited.
        - ValueError: If the input parameter values are not in the range / choices waited.
    """

    @wraps(move_to_previous_stage)
    def move_to_previous_stage_input_guarded(
        self: Self,
        desired_stage: int,
        force_move: bool=False
    ) -> None:

        # Check desired_stage type.
        if not isinstance(desired_stage, int):
            raise TypeError(f"The parameter 'desired_stage' must be an integer but you provided a {type(desired_stage).__name__:s} instead.")

        # Check desired_stage value
        if not (0 <= desired_stage <= 5):
            raise ValueError(f"The value of the parameter 'desired_stage' has to represent a valid stage (0 = SETTINGS_DEFINITION_STAGE / 1 = DATA_PREPROCESSING_STAGE / 2 = SCALER_FIT_STAGE / 3 = MODELS_TUNING_STAGE / 4 = AVERAGING_CONFIGURATION_STAGE / 5 = PRODUCTION_STAGE) but you provided the value {desired_stage:d}.")

        # Check force_move type.
        if not isinstance(force_move, bool):
            raise TypeError(f"The parameter 'force_move' must be a boolean but you provided a {type(force_move).__name__:s} instead.")

        # Call the project method run_tuning
        move_to_previous_stage(
            self=self,
            desired_stage=desired_stage,
            force_move=force_move
        )

    return move_to_previous_stage_input_guarded