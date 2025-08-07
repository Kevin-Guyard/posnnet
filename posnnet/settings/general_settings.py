import torch
from typing import Self, Tuple, List, Any

from posnnet.setter_guards import (
    setter_typeguard, 
    setter_positive_integer_guard,
    setter_string_choice_guard
)
from posnnet.settings.base_settings import BaseSettings


class GeneralSettings(BaseSettings):

    def __init__(
        self: Self
    ) -> None:

        """
        Initialize the general settings.

        Settings:
            - random_seed (int): A positive integer that is used to ensure determinism of the framework.
            - dtype (str): The dtype of the models and the data (have to be either 'float16', 'float32' or 'float64').
            - device (str): The device on which perform the training and the inference (have to be either 'cuda' for GPU, 'cpu' for CPU or 'auto'
                            in which case the framework select the GPU if available else the CPU).
        """

        self.random_seed = 42
        self.dtype = "float32"
        self.device = "auto"

    @property
    def random_seed(
        self: Self
    ) -> int:

        return self._random_seed

    @random_seed.setter
    @setter_typeguard
    @setter_positive_integer_guard
    def random_seed(
        self: Self,
        random_seed: int
    ) -> None:

        self._random_seed = random_seed

    @property
    def dtype(
        self: Self
    ) -> str:

        return self._dtype

    @dtype.setter
    @setter_typeguard
    @setter_string_choice_guard(allowed_choices=["float16", "float32", "float64"])
    def dtype(
        self: Self,
        dtype: str
    ) -> None:

        self._dtype = dtype

    @property
    def device(
        self: Self
    ) -> torch.device:

        return self._device

    @device.setter
    @setter_typeguard
    @setter_string_choice_guard(allowed_choices=["cuda", "cpu", "auto"])
    def device(
        self: Self,
        device: str
    ) -> None:

        if device == "cuda":
            self._device = torch.device("cuda")
        elif device == "cpu":
            self._device = torch.device("cpu")
        elif device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available else "cpu")
        else:
            pass # Should not happen.

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

        # If dtype is 'float16', generate a warning.
        if self.dtype == 'float16':
            warnings.append("The dtype provided is 'float16'. Using 16 bits precision can help to limit memory consumption and decrease training/inference time, but it comes with the cost of slightly worse accuracy of reconstruction. It is recommended to use 'float32' as dtype and mixed_precision=True to reduce memory consumption and training/inference time without reducing accuracy. Ignore this message if you are sure to use 'float16' as dtype.")

        # If dtype is 'float64', generate a warning.
        if self.dtype == 'float64':
            warnings.append("The dtype provided is 'float64'. Most of the time, using 64 bits precision leads to near-zero improvements but highly increase the memory consuption and training/inference time. It is recommended to use 'float32' as dtype. Ignore this message if you are sure to use 'float64' as dtype.")

        # If device is 'cuda' but cuda is not available, generate an error.
        if self.device == torch.device("cuda") and not torch.cuda.is_available():
            errors.append("You selected the device 'cuda' for training and inference but CUDA seems to be unavailable on your system. If your system does not have a CUDA compatible GPU, please select 'cpu' or 'auto' mode instead. If your system does have a CUDA compatible GPU, please check pytorch-cuda installation.")

        # If device is 'cpu', generate a warning.
        if self.device == torch.device("cpu"):
            warnings.append("You selected the device 'cpu' for training and inference. Due to this choice, the training and inference will be performed on your CPU even if your system has a CUDA compatible GPU, resulting in significant higher computation time. If you are not sure about your system having a CUDA compatible GPU or not, it is recommended to select the mode 'auto' to let the framework determine if CUDA is available. Ignore this message if you are sure to use the CPU.")

        return warnings, errors