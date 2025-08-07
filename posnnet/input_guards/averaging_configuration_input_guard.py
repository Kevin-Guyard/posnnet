from functools import wraps
from typing import Callable, Self, List


def averaging_configuration_input_guard(averaging_configuration: Callable) -> Callable:

    """
    Add a guard to averaging_configuration method of Project.

    Args:
        - averaging_configuration (Callable): The averaging_configuration method.

    Returns:
        - averaging_configuration_input_guard_input_guarded (Callable): The averaging_configuration method guarded.

    Raises:
        - TypeError: If the input parameter types are not the one waited.
        - ValueError: If the input parameter values are not in the range / choices waited.
    """

    @wraps(averaging_configuration)
    def averaging_configuration_input_guarded(
        self: Self,
        model_selection_levels: List[float],
        external_source: bool=False,
        verbosity: bool=False
    ) -> None:

        # Check model_name type.
        if not isinstance(model_selection_levels, list):
            raise TypeError(f"The parameter 'model_selection_levels' must be a list of float but you provided a {type(model_selection_levels).__name__:s} instead.")

        for item in model_selection_levels:
            if not isinstance(item, float):
                raise TypeError(f"The parameters 'model_selection_levels' should be a list of float but at least one element is a {type(item).__name__:s}.")
            if not 0 < item <= 1:
                raise ValueError(f"The parameters 'model_selection_levels' should contains float strictly superior to 0 and inferior or equal to 1, you provided: {item!r}.")

        # Check external_source type.
        if not isinstance(external_source, bool):
            raise TypeError(f"The parameter 'external_source' must be a boolean but you provided a {type(external_source).__name__:s} instead.")

        # Check verbosity type.
        if not isinstance(verbosity, bool):
            raise TypeError(f"The parameter 'verbosity' must be a boolean but you provided a {type(verbosity).__name__:s} instead.")
        
        # Call the project method averaging_configuration
        averaging_configuration(
            self=self,
            model_selection_levels=model_selection_levels,
            external_source=external_source,
            verbosity=verbosity
        )

    return averaging_configuration_input_guarded