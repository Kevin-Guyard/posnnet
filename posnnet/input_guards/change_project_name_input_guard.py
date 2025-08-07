from functools import wraps
from typing import Callable, Any, Union, Self


def change_project_name_input_guard(change_project_name: Callable) -> Callable:

    """
    Add a guard to change_project_name method of Project.

    Args:
        - change_project_name (Callable): The change_project_name method.

    Returns:
        - change_project_name_input_guarded (Callable): The run_tuning method guarded.

    Raises:
        - TypeError: If the input parameter types are not the one waited.
        - ValueError: If the input parameter values are not in the range / choices waited.
    """

    @wraps(change_project_name)
    def change_project_name_input_guarded(
        self: Self,
        new_project_name: str
    ) -> None:

        # Check new_project_name type.
        if not isinstance(new_project_name, str):
            raise TypeError(f"The parameter 'new_project_name' must be a string but you provided a {type(new_project_name).__name__:s} instead.")

        # Check that new_project_name is not an empty string.
        if new_project_name == "":
            raise ValueError(f"The parameter 'new_project_name' must be a none empty string.")

        # Check that new_project_name is not an empty string.
        if "/" in new_project_name or ":" in new_project_name:
            raise ValueError(f"The parameter 'new_project_name' cannot contains the characters '/' or ':'.")
        
        # Call the project method run_tuning
        change_project_name(
            self=self,
            new_project_name=new_project_name
        )

    return change_project_name_input_guarded