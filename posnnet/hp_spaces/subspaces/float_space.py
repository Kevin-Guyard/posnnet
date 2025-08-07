import optuna
from typing import Self, Union


class FloatSpace:

    def __init__(
        self: Self,
        name: str,
        low: float,
        high: float,
        log: bool=False,
        step: Union[float, None]=None
    ) -> None:

        """
        Initiate a float space (a space where the sampling is done on a continuous and bounded space).

        Args:
            - name (str): The name of the space.
            - low (float): The lower bound of the space.
            - high (float): The lower bound of the space.
            - log (bool): Either to sample from the linear domain (False) or in the log domain (True). In other words, 
                          if set to True, smaller values are more likely to be sampled. Default = False.
            - step (Union[float, None]): If not set to None, the continuous domain is discretized using the value provided. 
                                         Impossible to use with log set to True. Default = None.
        """

        self.name = name
        self.low = low 
        self.high = high
        self.log = log
        self.step = step

    def sample_value(
        self: Self,
        trial: optuna.trial.Trial
    ) -> float:

        """
        Sample a value from from the space.

        Args:
            - trial (optuna.trial.Trial): The Trial object of the actual trial.

        Returns:
            - value (float): The sampled value from the space.
        """

        value = trial.suggest_float(name=self.name, low=self.low, high=self.high, log=self.log, step=self.step)

        return value