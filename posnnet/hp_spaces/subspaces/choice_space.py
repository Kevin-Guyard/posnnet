import optuna
from typing import Self, List, Any


class ChoiceSpace:

    def __init__(
        self: Self,
        name: str,
        choices: List[Any]
    ) -> None:

        """
        Initiate a choice space (a space where the sampling is done from a list of object of any type).

        Args:
            - name (str): The name of the space.
            - choices (List[Any]): The list of objects from which to sample.
        """

        self.name = name
        self.choices = choices

    def sample_value(
        self: Self,
        trial: optuna.trial.Trial
    ) -> Any:

        """
        Sample a value from from the space.

        Args:
            - trial (optuna.trial.Trial): The Trial object of the actual trial.

        Returns:
            - value (Any): The sampled object from the list provided during instantiation.
        """

        value = trial.suggest_categorical(name=self.name, choices=self.choices)

        return value