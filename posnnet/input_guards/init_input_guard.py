from functools import wraps
from typing import Callable, Any, Union, Self


def init_input_guard(init: Callable) -> Callable:

    """
    Add a guard to __init__ method of Project.

    Args:
        - init (Callable): The __init__ method.

    Returns:
        - init_input_guarded (Callable): The run_tuning method guarded.

    Raises:
        - TypeError: If the input parameter types are not the one waited.
        - ValueError: If the input parameter values are not in the range / choices waited.
    """

    @wraps(init)
    def init_input_guarded(
        self: Self,
        project_name: str
    ) -> None:

        # Check project_name type.
        if not isinstance(project_name, str):
            raise TypeError(f"The parameter 'project_name' must be a string but you provided a {type(project_name).__name__:s} instead.")

        # Check that project_name is not an empty string.
        if project_name == "":
            raise ValueError(f"The parameter 'project_name' must be a none empty string.")

        # Check that project_name is not an empty string.
        if "/" in project_name or ":" in project_name:
            raise ValueError(f"The parameter 'project_name' cannot contains the characters '/' or ':'.")
        
        # Call the project method run_tuning
        init(
            self=self,
            project_name=project_name
        )

    return init_input_guarded