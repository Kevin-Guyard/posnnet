from typing import Self, List, Tuple, Any

from posnnet.setter_guards import (
    setter_typeguard, 
    setter_strictly_positive_integer_guard,
    setter_gps_duration_guard
)
from posnnet.settings.base_settings import BaseSettings


class ObjectiveSettings(BaseSettings):

    def __init__(
        self: Self
    ) -> None:

        """
        Initialize the objective settings.

        Settings:
            - n_minimum_seconds_operational_gps (int): The minimum number of seconds of operational GPS that can be ensured next to every GPS outage.
                                                       Have to be a strictly positive integer
            - gps_outage_durations_eval_at_beginning (List[Tuple[str, str]]): The list of cases of GPS outage durations that will be simulated at the beginning, 
                                                                              following the format [(N1, N2), (N3, N4), ...] to simulate GPS outage from 
                                                                              N1 to N2 seconds and N3 to N4 seconds etc. Format of Nx: "MM:SS".
            - gps_outage_durations_eval_within (List[Tuple[str, str]]): The list of cases of GPS outage durations that will be simulated within the session, 
                                                                        following the format [(N1, N2), (N3, N4), ...] to simulate GPS outage from 
                                                                        N1 to N2 seconds and N3 to N4 seconds etc. Format of Nx: "MM:SS".
            - gps_outage_durations_eval_at_end (List[Tuple[str, str]]): The list of cases of GPS outage durations that will be simulated at the end, 
                                                                        following the format [(N1, N2), (N3, N4), ...] to simulate GPS outage from 
                                                                        N1 to N2 seconds and N3 to N4 seconds etc. Format of Nx: "MM:SS".
        """

        self.n_minimum_seconds_operational_gps = 4
        self.gps_outage_durations_eval_at_beginning = []
        self.gps_outage_durations_eval_within = []
        self.gps_outage_durations_eval_at_end = []

    @property
    def n_minimum_seconds_operational_gps(
        self: Self
    ) -> int:

        return self._n_minimum_seconds_operational_gps

    @n_minimum_seconds_operational_gps.setter
    @setter_typeguard
    @setter_strictly_positive_integer_guard
    def n_minimum_seconds_operational_gps(
        self: Self,
        n_minimum_seconds_operational_gps: int
    ) -> None:

        self._n_minimum_seconds_operational_gps = n_minimum_seconds_operational_gps

    @property
    def gps_outage_durations_eval_at_beginning(
        self: Self
    ) -> List[Tuple[str, str]]:

        return self._gps_outage_durations_eval_at_beginning

    @gps_outage_durations_eval_at_beginning.setter
    @setter_typeguard
    @setter_gps_duration_guard
    def gps_outage_durations_eval_at_beginning(
        self: Self,
        gps_outage_durations_eval_at_beginning: List[Tuple[str, str]]
    ) -> None:

        self._gps_outage_durations_eval_at_beginning = gps_outage_durations_eval_at_beginning

    @property
    def gps_outage_durations_eval_within(
        self: Self
    ) -> List[Tuple[str, str]]:

        return self._gps_outage_durations_eval_within

    @gps_outage_durations_eval_within.setter
    @setter_typeguard
    @setter_gps_duration_guard
    def gps_outage_durations_eval_within(
        self: Self,
        gps_outage_durations_eval_within: List[Tuple[str, str]]
    ) -> None:

        self._gps_outage_durations_eval_within = gps_outage_durations_eval_within

    @property
    def gps_outage_durations_eval_at_end(
        self: Self
    ) -> List[Tuple[str, str]]:

        return self._gps_outage_durations_eval_at_end

    @gps_outage_durations_eval_at_end.setter
    @setter_typeguard
    @setter_gps_duration_guard
    def gps_outage_durations_eval_at_end(
        self: Self,
        gps_outage_durations_eval_at_end: List[Tuple[str, str]]
    ) -> None:

        self._gps_outage_durations_eval_at_end = gps_outage_durations_eval_at_end

    def validate_settings(
        self: Self,
        **kwargs: Any
    ) -> Tuple[List[str], List[str]]:

        """
        Validate the value of the different setting parameters.

        Returns:
            - warnings (List[str]): A list of warnings on the different setting parameter values.
            - errors (List[str]): A list of errors on the different setting parameter values.
        """

        warnings, errors = [], []

        # If the minimum number of seconds of operational GPS is not 4 seconds, generate a warning.
        if self.n_minimum_seconds_operational_gps != 4:
            warnings.append(f"The minimum number of seconds of operational GPS specified you specified is {self.n_minimum_seconds_operational_gps:d}. The framework POSNNET has been developped and tested using the value of 4 seconds. Be aware that you can have different results based on your choice. Ignore this message if you are sure about your choice.")

        # If the minimum number of seconds of operational GPS is not even, generate an error.
        if self.n_minimum_seconds_operational_gps % 2 != 0:
            errors.append(f"The minimum number of seconds of operational GPS specified you specified is {self.n_minimum_seconds_operational_gps:d}. The number have to be even. Please provide an even number of seconds.")

        # If GPS outage durations of evaluation for beginning, end and within are all empty, generate an error.
        if len(self.gps_outage_durations_eval_at_beginning) == 0 and len(self.gps_outage_durations_eval_within) == 0 and len(self.gps_outage_durations_eval_at_end) == 0:
            errors.append("You have not selected any duration for evaluation. Please provide at least one duration for beginning or within or end case.")

        return warnings, errors