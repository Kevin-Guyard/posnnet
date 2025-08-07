from typing import Self, Union, Tuple, List, Any

from posnnet.setter_guards import (
    setter_typeguard, 
    setter_positive_integer_guard,
    setter_strictly_positive_integer_guard,
    setter_strictly_positive_integer_or_none_guard,
    setter_string_choice_guard
)
from posnnet.settings.base_settings import BaseSettings

class TrainingSettings(BaseSettings):

    def __init__(
        self: Self
    ) -> None:

        """
        Initialize the training settings.

        Settings:
            - coeff_sampling_training (int): The ratio of sample that will be used as window beginning every epoch for training (e.g. coeff_sampling = 10000 means 1/10000 
                                             of the dataset sample will be used to create a window). Have to be a strictly positive integer
            - coeff_sampling_validation (int): The ratio of sample that will be used as window beginning every epoch for validation (e.g. coeff_sampling = 10000 means 1/10000
                                               of the dataset sample will be used to create a window). Have to be a strictly positive integer.
            - n_epochs_sampling (int): The number of epochs for a complete sampling rotation of the dataset. Have to be a strictly positive integer.
            - n_epochs (int): The number of epochs for which train the models (if early stopping is not triggered). Have to be a strictly positive integer.
            - patience (Union[int, None]): The number of epochs without validation results improvement before early stopping training.
                                           If None, training is never early stopped. If an integer is provided, it has to be strictly positive.
            - num_workers (int): The number of process that will instanciate a dataloader instance to load the data in parallel. 0 means that
                                 the dataloader will be loaded in the main process. A rule of thumb is to use 4 processes by GPU. Have to be a positive integer.
            - use_mixed_precision (bool): Either to use or not float16 mixed precision for training.
            - n_epochs_training_checkpoint (int): The number of epochs of training after which to save a checkpoint. Have to be a strictly positive integer.
        """

        self.coeff_sampling_training = 10000
        self.coeff_sampling_validation = 2000
        self.n_epochs_sampling = 20
        self.n_epochs = 100
        self.patience = 10
        self.num_workers = 0
        self.use_mixed_precision = True
        self.n_epochs_training_checkpoint = 10
        self.velocity_loss = "mse"

    @property
    def coeff_sampling_training(
        self: Self
    ) -> int:

        return self._coeff_sampling_training

    @coeff_sampling_training.setter
    @setter_typeguard
    @setter_strictly_positive_integer_guard
    def coeff_sampling_training(
        self: Self,
        coeff_sampling_training: int
    ) -> None:

        self._coeff_sampling_training = coeff_sampling_training

    @property
    def coeff_sampling_validation(
        self: Self
    ) -> int:

        return self._coeff_sampling_validation

    @coeff_sampling_validation.setter
    @setter_typeguard
    @setter_strictly_positive_integer_guard
    def coeff_sampling_validation(
        self: Self,
        coeff_sampling_validation: int
    ) -> None:

        self._coeff_sampling_validation = coeff_sampling_validation

    @property
    def n_epochs_sampling(
        self: Self
    ) -> int:

        return self._n_epochs_sampling

    @n_epochs_sampling.setter
    @setter_typeguard
    @setter_strictly_positive_integer_guard
    def n_epochs_sampling(
        self: Self,
        n_epochs_sampling: int
    ) -> None:

        self._n_epochs_sampling = n_epochs_sampling

    @property
    def n_epochs(
        self: Self
    ) -> int:

        return self._n_epochs

    @n_epochs.setter
    @setter_typeguard
    @setter_strictly_positive_integer_guard
    def n_epochs(
        self: Self,
        n_epochs: int
    ) -> None:

        self._n_epochs = n_epochs

    @property
    def patience(
        self: Self
    ) -> Union[int, None]:

        return self._patience

    @patience.setter
    @setter_typeguard
    @setter_strictly_positive_integer_or_none_guard
    def patience(
        self: Self,
        patience: Union[int, None]
    ) -> None:

        self._patience = patience

    @property
    def num_workers(
        self: Self
    ) -> int:

        return self._num_workers

    @num_workers.setter
    @setter_typeguard
    @setter_positive_integer_guard
    def num_workers(
        self: Self,
        num_workers: int
    ) -> None:

        self._num_workers = num_workers

    @property
    def use_mixed_precision(
        self: Self
    ) -> int:

        return self._use_mixed_precision

    @use_mixed_precision.setter
    @setter_typeguard
    def use_mixed_precision(
        self: Self,
        use_mixed_precision: int
    ) -> None:

        self._use_mixed_precision = use_mixed_precision

    @property
    def n_epochs_training_checkpoint(
        self: Self
    ) -> int:

        return self._n_epochs_training_checkpoint

    @n_epochs_training_checkpoint.setter
    @setter_typeguard
    @setter_strictly_positive_integer_guard
    def n_epochs_training_checkpoint(
        self: Self,
        n_epochs_training_checkpoint: int
    ) -> None:

        self._n_epochs_training_checkpoint = n_epochs_training_checkpoint

    @property
    def velocity_loss(
        self: Self
    ) -> str:

        return self._velocity_loss

    @velocity_loss.setter
    @setter_typeguard
    @setter_string_choice_guard(allowed_choices=["mae", "mse"])
    def velocity_loss(
        self: Self,
        velocity_loss: str
    ) -> None:

        self._velocity_loss = velocity_loss

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

        # If the mixed precision is not activated, generate a warning.
        if self.use_mixed_precision == False:
            warnings.append("You have not selected 'use_mixed_precision=True'. Mixed precision can help reducing memory consumption and training time with nearly no cost on prediction accuracy. It is recommended to use it. Ignore this message if you are sure about your choice.")

        # If the number of epochs for checkpoint is superior to the number of epochs, generate a warning.
        if self.n_epochs_training_checkpoint >= self.n_epochs:
            warnings.append(f"You have set the number of training epochs to {self.n_epochs:d} and the number of epochs between two training checkpoints to {self.n_epochs_training_checkpoint:d}. Thus, no training checkpoint will be saved. If you are sure about the fact that your training will not be stopped by any external source, keep this choice. If the training can be shut down at any moment (for e.g. personal computer, computing nodes with maximum computation time, etc), it is recommended to consider to use a number of epochs between two training checkpoints inferior to the number of epochs.")

        return warnings, errors