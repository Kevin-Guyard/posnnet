from functools import wraps
from typing import Callable, Any, Union, Self, Tuple


def run_tuning_input_guard(run_tuning: Callable) -> Callable:

    """
    Add a guard to run_tuning method of Project.

    Args:
        - run_tuning (Callable): The run_tuning method.

    Returns:
        - run_tuning_input_guarded (Callable): The run_tuning method guarded.

    Raises:
        - TypeError: If the input parameter types are not the one waited.
        - ValueError: If the input parameter values are not in the range / choices waited.
    """

    @wraps(run_tuning)
    def run_tuning_input_guarded(
        self: Self,
        n_experiments: int,
        model_name: str,
        use_adversarial: Union[str, Any],
        training_type: str,
        coeff_frequency_division: int,
        gps_outage_duration: Tuple[str, str],
        verbosity: int
    ) -> None:

        # Check n_experiments type.
        if not isinstance(n_experiments, int):
            raise TypeError(f"The parameter 'n_experiments' must be an integer but you provided a {type(n_experiments).__name__:s} instead.")

        # Check that n_experiments is a strictly positive integer.
        if not n_experiments > 0:
            raise ValueError(f"The parameter 'n_experiments' must be a strictly positive integer but you provided the value {n_experiments:d}.")

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

        # Check verbosity type.
        if not isinstance(verbosity, int):
            raise TypeError(f"The parameter 'verbosity' must be an integer but you provided a {type(verbosity).__name__:s} instead.")

        # Check verbosity value.
        if not 0 <= verbosity <= 2:
            raise ValueError(f"The parameter 'verbosity' should be 0 for no verbosity, 1 for final model's results or 2 for intermediate mdoel's results but you provided the value {verbosity:d}.")
        
        # Call the project method run_tuning
        run_tuning(
            self=self,
            n_experiments=n_experiments,
            model_name=model_name,
            use_adversarial=use_adversarial,
            training_type=training_type,
            coeff_frequency_division=coeff_frequency_division,
            gps_outage_duration=gps_outage_duration,
            verbosity=verbosity
        )

    return run_tuning_input_guarded