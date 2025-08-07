from functools import wraps
from typing import Callable, Any, Union, Self, Tuple


def evaluate_configuration_input_guard(evaluate_configuration: Callable) -> Callable:

    """
    Add a guard to evaluate_configuration method of Project.

    Args:
        - evaluate_configuration (Callable): The evaluate_configuration method.

    Returns:
        - evaluate_configuration_input_guard_input_guarded (Callable): The evaluate_configuration method guarded.

    Raises:
        - TypeError: If the input parameter types are not the one waited.
        - ValueError: If the input parameter values are not in the range / choices waited.
    """

    @wraps(evaluate_configuration)
    def evaluate_configuration_input_guarded(
        self: Self,
        model_name: str,
        use_adversarial: Union[str, Any],
        training_type: str,
        coeff_frequency_division: int,
        gps_outage_duration: Tuple[str, str],
        external_source: bool=False,
        verbosity: bool=False
    ) -> None:

        # Check model_name type.
        if not isinstance(model_name, str):
            raise TypeError(f"The parameter 'model_name' must be a string but you provided a {type(model_name).__name__:s} instead.")

        # Check model_name value.
        if not model_name in ["CLSTMTFAWB", "CLSTMTWB", "STTFAWB", "STTWB", "TCANWB"]:
            raise ValueError(f"The parameter 'model_name' has to be one of the available model ('CLSTMTFAWB', 'CLSTMTWB', 'STTFAWB', 'STTWB', 'TCANWB') but you provided {model_name:s}.")

        # Check use_adversarial type.
        if use_adversarial is not None and not isinstance(use_adversarial, str):
            raise TypeError(f"The parameter 'use_adversarial' must be a string or None but you provided a {type(use_adversarial).__name__:s} instead.")

        # Check use_adversarial value.
        if use_adversarial is not None and use_adversarial not in ["imu", "full"]:
            raise ValueError(f"The parameter 'use_adversarial' has to be either 'imu', 'full' or None.")

        # Check training_type type.
        if not isinstance(training_type, str):
            raise TypeError(f"The parameter 'training_type' must be a string but you provided a {type(training_type).__name__:s} instead.")

        # Check training_type value.
        if not training_type in ["beginning", "centered", "end", "random"]:
            raise ValueError(f"The parameter 'training_type' has to be one of the available model ('beginning', 'centered', 'end', 'random') but you provided {training_type:s}.")

        # Check coeff_frequency_division type.
        if not isinstance(coeff_frequency_division, int):
            raise TypeError(f"The parameter 'coeff_frequency_division' must be an integer but you provided a type {type(coeff_frequency_division).__name__:s} instead.")

        # Check that coeff_frequency_division is a strictly positive integer.
        if not coeff_frequency_division > 0:
            raise ValueError(f"The parameter 'coeff_frequency_division' must be a strictly positive integer but you provided the value {coeff_frequency_division:d}.")

        # Check gps_outage_duration type:
        if not isinstance(gps_outage_duration, tuple):
            raise TypeError(f"The parameter 'gps_outage_duration' must be a tuple but you provided the {type(gps_outage_duration).__name__:s} instead.")

        # Check that the length of gps_outage_duration is 2.
        if not len(gps_outage_duration) == 2:
            raise ValueError(f"The parameter 'gps_outage_duration' must have two items but you provided {len(gps_outage_duration):d} instead.")

        # Check the type of the 2 items of gps_outage_duration.
        if not isinstance(gps_outage_duration[0], str):
            raise ValueError(f"The first value of 'gps_outage_duration' must be a string but you provided a {type(gps_outage_duration[0]).__name__:s} instead.")
        if not isinstance(gps_outage_duration[1], str):
            raise ValueError(f"The second value of 'gps_outage_duration' must be a string but you provided a {type(gps_outage_duration[1]).__name__:s} instead.")

        # Check the format of the 2 items of gps_outage_duration.
        if not len(gps_outage_duration[0]) == 5 or not gps_outage_duration[0][0:2].isdigit() or not gps_outage_duration[0][2] == ":" or not gps_outage_duration[0][3:5].isdigit():
            raise ValueError(f"The first value of 'gps_outage_duration' must follow the format 'MM:SS' but you provided {gps_outage_duration[0]}.")
        if not len(gps_outage_duration[1]) == 5 or not gps_outage_duration[1][0:2].isdigit() or not gps_outage_duration[1][2] == ":" or not gps_outage_duration[1][3:5].isdigit():
            raise ValueError(f"The second value of 'gps_outage_duration' must follow the format 'MM:SS' but you provided {gps_outage_duration[1]}.")

        # Check that the first item is not null
        if gps_outage_duration[0] == "00:00":
            raise ValueError("The first value of 'gps_outage_duration' cannot be null but you provided '00:00'.")

        # Check that the first item is inferior to the second.
        if 60 * int(gps_outage_duration[0][0:2]) + int(gps_outage_duration[0][3:5]) >= 60 * int(gps_outage_duration[1][0:2]) + int(gps_outage_duration[1][3:5]):
            raise ValueError(f"The first value of 'gps_outage_duration' must be inferior to the second but you provided {gps_outage_duration}.")

        # Check external_source type.
        if not isinstance(external_source, bool):
            raise TypeError(f"The parameter 'external_source' must be a boolean but you provided a {type(external_source).__name__:s} instead.")

        # Check verbosity type.
        if not isinstance(verbosity, bool):
            raise TypeError(f"The parameter 'verbosity' must be a boolean but you provided a {type(verbosity).__name__:s} instead.")
        
        # Call the project method evaluate_configuration
        evaluate_configuration(
            self=self,
            model_name=model_name,
            use_adversarial=use_adversarial,
            training_type=training_type,
            coeff_frequency_division=coeff_frequency_division,
            gps_outage_duration=gps_outage_duration,
            external_source=external_source,
            verbosity=verbosity
        )

    return evaluate_configuration_input_guarded