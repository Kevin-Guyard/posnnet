from typing import Self, Tuple, List, Any

from posnnet.setter_guards import (
    setter_typeguard, 
    setter_strictly_positive_integer_guard,
)
from posnnet.settings.base_settings import BaseSettings


class TuningSettings(BaseSettings):

    def __init__(
        self: Self
    ) -> None:

        """
        Initialize the tuning settings.

        Settings:
            - n_startup_trials (int): The number of experiments randomly sampled before using Bayesian optimization to search in the hyperparameter space.
                                      Have to be a strictly positive integer.
            - save_every_state_dicts (bool): Either to save or not every model state dicts (if False, only the best model state dict is kept).
        """

        self.n_startup_trials = 10
        self.save_every_state_dicts = False

    @property
    def n_startup_trials(
        self: Self
    ) -> int:

        return self._n_startup_trials

    @n_startup_trials.setter
    @setter_typeguard
    @setter_strictly_positive_integer_guard
    def n_startup_trials(
        self: Self,
        n_startup_trials: int
    ) -> None:

        self._n_startup_trials = n_startup_trials

    @property
    def save_every_state_dicts(
        self: Self
    ) -> int:

        return self._save_every_state_dicts

    @save_every_state_dicts.setter
    @setter_typeguard
    def save_every_state_dicts(
        self: Self,
        save_every_state_dicts: int
    ) -> None:

        self._save_every_state_dicts = save_every_state_dicts

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

        # If the number of starting trial is inferior to 10, generate a warning.
        if self.n_startup_trials < 10:
            warnings.append(f"The number of startup trials for the tuning is set to {self.n_startup_trials:d}. In general, it is recommended to have at least 10 startup trials. Ignore this message if you are sure about your choice.")

        # If save every set dict option is set to True, generate a warning.
        if self.save_every_state_dicts == True:
            warnings.append("You selected the option 'save every state dict'. This option is mainly used for debug and results tracking. Be aware of the fact that it will save a lot of heavy file on disk, taking up to hundred of Go. If you do not want to debug or track results, it is recommended to disable this option. Ignore this message if you are sure to activate this option.")

        return warnings, errors