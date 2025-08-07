import numpy as np
import optuna
from typing import Self, Union


class IntegerSpace:

    def __init__(
        self: Self,
        name: str,
        low: int,
        high: int,
        log2: bool=False,
        step: int=1,
    ) -> None:

        """
        Initiate an integer space (a space where the sampling is done on a discret and bounded space of integer).

        Args:
            - name (str): The name of the space.
            - low (float): The lower bound of the space.
            - high (float): The lower bound of the space.
            - log2 (bool): Either to sample with an uniform probability every integer in [low, high] (False) or to sample
                           with an uniform probability only the integer i which satisfy i = 2 ^ x where x is any positive 
                           integer or zero (True). Example for log2 = True, low = 2 and high = 8: the space is composed
                           of [2, 4, 8] where every value has a probability of 1/3 to be sampled.
            - step (int): The step of the integer that can be sampled between low and high. Cannot be set to another value
                          than 1 if log2 is True. Example with step = 2, low = 2 and high = 8: the space is composed
                          of [2, 4, 6, 8] where every value has a probability of 1/4 to be sampled.
        """

        self.name = name
        self.low = low 
        self.high = high
        self.log2 = log2
        self.step = step

    def sample_value(
        self: Self,
        trial: optuna.trial.Trial,
        constraint_high: Union[int, None]=None
    ) -> int:

        """
        Sample a value from from the space.

        Args:
            - trial (optuna.trial.Trial): The Trial object of the actual trial.
            - constraint_high (Union[int, None]): If not None, limit the upper bound of the space. Default = None.

        Returns:
            - value (float): The sampled value from the space.
        """

        if constraint_high is None:
            high = self.high
        else:
            high = min(self.high, constraint_high)

        if self.log2:
            value = 2 ** trial.suggest_int(name=f"log2_{self.name}", low=int(np.log2(self.low)), high=int(np.log2(high)))
        else:
            value = trial.suggest_int(name=self.name, low=self.low, high=high, step=self.step)

        return value