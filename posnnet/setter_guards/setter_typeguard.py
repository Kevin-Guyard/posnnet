from functools import wraps
from typing import get_type_hints, get_origin, get_args,  Callable, List, Union, Tuple, Any, Self


def setter_typeguard(setter_function: Callable) -> Callable:

    """
    Check the type for setter functions.

    Args:
        - setter_function (Callable): The setter function

    Returns:
        - setter_typeguarded (Callable): The setter function with a type guard.

    Raises:
        - TypeError: If the type of the variable is not one of those waited.
    """

    @wraps(setter_function)
    def setter_typeguarded(*args: Self, **kwargs: Any) -> None:

        # Get the type hints and remove the instance self and return type hint.
        type_hints = get_type_hints(setter_function)
        type_hints.pop("self")
        type_hints.pop("return")

        # Get the param name and the type hint.
        (param_name, type_hint), = type_hints.items()

        # Get the param value.
        param_value = args[1]

        # Check for integer, float and str.
        if type_hint in [int, float, str]:
            
            if not isinstance(param_value, type_hint):
                raise TypeError(f"The parameter '{param_name:s}' must be a {type_hint.__name__:s} but you provided a {type(param_value).__name__:s} instead.")
                
        # Check for list if int, float or str
        elif get_origin(type_hint) == list and get_args(type_hint)[0] in [int, float, str]:
            
            if not isinstance(param_value, list):
                raise TypeError(f"The parameter '{param_name:s}' must be a list of {get_args(type_hint)[0].__name__:s} but you provided a {type(param_value).__name__:s} instead.")

            for item in param_value:
                if not isinstance(item, get_args(type_hint)[0]):
                    raise TypeError(f"The parameter '{param_name:s}' must be a list of {get_args(type_hint)[0].__name__:s} but you provided a list with at least one element of type {type(item).__name__:s} instead.")

        # Check for list of typle of int
        elif type_hint == List[Tuple[str, str]]:

            if not isinstance(param_value, list):
                raise TypeError(f"The parameter '{param_name:s}' must be a list of tuple of two strings but you provided a {type(param_value).__name__:s} instead.")

            for item in param_value:
                
                if not isinstance(item, tuple):
                    raise TypeError(f"The parameter '{param_name:s}' must be a list of tuple of two strings but you provided a list with at least one element of type {type(item).__name__:s} instead.")
                    
                if not len(item) == 2:
                    raise TypeError(f"The parameter '{param_name:s}' must be a list of tuple of two strings but you provided a list with at least one tuple of a length {len(item):d} instead.")

                if not isinstance(item[0], str) or not isinstance(item[1], str):
                    
                    raise TypeError(f"The parameter '{param_name:s}' must be a list of tuple of two strings but you provided a list with at least one tuple with an element of type {[type(subitem) for subitem in item if not isinstance(subitem, str)][0].__name__:s} instead.")
                    

        # Check for integer, float and str with null value possible.
        elif get_origin(type_hint) == Union and get_args(type_hint)[0] in [str, int, float] and get_args(type_hint)[1] == type(None):

            if not type(param_value) in get_args(type_hint):
                raise TypeError(f"The parameter '{param_name:s}' must be a either a {get_args(type_hint)[0].__name__:s} or None but you provided a {type(param_value).__name__:s} instead.")

        # Check for union of integer, float and str
        elif get_origin(type_hint) == Union and set(get_args(type_hint)).issubset({str, int, float}):

            if not type(param_value) in get_args(type_hint):
                raise TypeError(f"The parameter '{param_name:s}' must be one of the following choices: {[choice.__name__ for choice in get_args(type_hint)]}, but you provided a {type(param_value).__name__:s} instead.")
                
        else:
            
            raise NotImplementedError(f"setter_typeguard not implemented for type {type_hint}.")

        # Call the setter function
        setter_function(*args, **kwargs)

    return setter_typeguarded