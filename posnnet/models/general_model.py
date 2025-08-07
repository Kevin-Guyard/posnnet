import torch
from typing import Self, Union, Dict, Any, OrderedDict, Generator

from posnnet.models.convolutional_long_short_term_memory_transformer_frequency_aware_with_bypass import ConvolutionalLongShortTermMemoryTransformerFrequencyAwareWithBypass
from posnnet.models.convolutional_long_short_term_memory_transformer_with_bypass import ConvolutionalLongShortTermMemoryTransformerWithBypass
from posnnet.models.spatio_temporal_transformer_frequency_aware_with_bypass import SpatioTemporalTransformerFrequencyAwareWithBypass
from posnnet.models.spatio_temporal_transformer_with_bypass import SpatioTemporalTransformerWithBypass
from posnnet.models.temporal_convolutional_attentive_network_with_bypass import TemporalConvolutionalAttentiveNetworkWithBypass


class GeneralModel:

    def __init__(
        self: Self,
        model_name: str,
        model_params: Dict[str, Any],
        model_state_dict: Union[OrderedDict[str, torch.Tensor], None]=None
    ) -> None:

        """
        Initiate an instance of a GeneralModel. The GeneralModel is a wrap to handle all the models of the project through one access point.
        The GeneralModel instance follows the structure of torch.nn.Module on the specified methods.

        Args:
            - model_name (str): The name of the model to use (can be either 'CLSTMTFAWB', 'CLSTMWB', 'STTFAWB', 'STTWB' or 'TCANWB').
            - model_params (Dict[str, Any]): The parameters that will be provided to the model class.
            - model_state_dict (Union[OrderedDict[str, torch.Tensor], None]): If not None, the instance of the model will use it to reload a model state.
        """

        # Create a mapping between model name and model class
        mapping_model_name_to_model_class = {
            "CLSTMTFAWB": ConvolutionalLongShortTermMemoryTransformerFrequencyAwareWithBypass,
            "CLSTMTWB": ConvolutionalLongShortTermMemoryTransformerWithBypass,
            "STTFAWB": SpatioTemporalTransformerFrequencyAwareWithBypass,
            "STTWB": SpatioTemporalTransformerWithBypass,
            "TCANWB": TemporalConvolutionalAttentiveNetworkWithBypass,
        }

        # Get the class of the model
        model_class = mapping_model_name_to_model_class.get(model_name)

        # Instanciate a new model
        self.__model = model_class(**model_params)

        # If the state dict is provided, load it
        if model_state_dict is not None:
            self.__model.load_state_dict(model_state_dict)

    def to(
        self: Self,
        device: Union[torch.device, None]=None,
        dtype: Union[torch.dtype, None]=None
    ) -> Self:

        """
        Perform a device and/or a dtype change of the model.

        Args:
            - device (Union[torch.device, None]): The device on which set the model (no action if set to None). Default value = None.
            - dtype (Union[torch.device, None]): The dtype on which set the model (no action if set to None). Default value = None.

        Returns:
            - self (Self): the instance.
        """

        # If only the device is provided, perform the device change.
        if device is not None and dtype is None:
            self.__model = self.__model.to(device=device)
        # Else if only the dtype is provided, perform the dtype change.
        elif device is None and dtype is not None:
            self.__model = self.__model.to(dtype=dtype)
        # Else if both are provided, perform device and dtype change.
        elif device is not None and dtype is not None:
            self.__model = self.__model.to(device=device, dtype=dtype)
        # Else, no actions
        else:
            pass

        return self

    def train(
        self: Self
    ) -> Self:

        """
        Set the model in training mode.

        Returns:
            - self (Self): the instance.
        """

        # Set the model in training mode
        self.__model = self.__model.train()

        return self

    def eval(
        self: Self
    ) -> Self:

        """
        Set the model in evaluation mode.

        Returns:
            - self (Self): the instance.
        """

        # Set the model in evaluation mode
        self.__model = self.__model.eval()

        return self

    def forward(
        self: Self,
        x_accelerometer: torch.Tensor,
        x_gyroscope: torch.Tensor,
        x_magnetometer: torch.Tensor,
        x_velocity_gps: torch.Tensor,
        x_velocity_fusion: torch.Tensor,
        x_orientation_fusion: torch.Tensor,
        x_mask: torch.Tensor,
    ) -> torch.Tensor:

        """
        Forward function.

        Args:
            - x_accelerometer (torch.Tensor): The input tensor that contains accelerometer data of size (batch, seq, n_accelerometer) with 0 <= n_accelerometer <= 3.
            - x_gyroscope (torch.Tensor): The input tensor that contains gyroscope data of size (batch, seq, n_gyroscope) with 0 <= n_gyroscope <= 3.
            - x_magnetometer (torch.Tensor): The input tensor that contains magnetometer data of size (batch, seq, n_magnetometer) with 0 <= n_magnetometer <= 3.
            - x_velocity_gps (torch.Tensor): The input tensor that contains GPS velocity data of size (batch, seq, n_velocity_gps) with 0 <= n_velocity_gps <= 3.
            - x_velocity_fusion (torch.Tensor): The input tensor that contains Kalman fusion output velocity data of size (batch, seq, n_velocity_fusion) with 1 <= n_velocity_fusion <= 3.
            - x_orientation_fusion (torch.Tensor): The input tensor that contains Kalman fusion output orientation data of size (batch, seq, n_orientation_fusion) with 0 <= n_orientation_fusion <= 3.
            - x_mask (torch.Tensor): The input tensor that contains mask (GPS outage indication) data of size (batch, seq, 1)

        Returns
            - y (torch.Tensor): The output tensor of the layer of size (batch, seq, output_size) with 1 <= output_size <= 3.
        """

        y = self.__model(
            x_accelerometer=x_accelerometer,
            x_gyroscope=x_gyroscope,
            x_magnetometer=x_magnetometer,
            x_velocity_gps=x_velocity_gps,
            x_velocity_fusion=x_velocity_fusion,
            x_orientation_fusion=x_orientation_fusion,
            x_mask=x_mask
        )

        return y

    def __call__(
        self: Self,
        x_accelerometer: torch.Tensor,
        x_gyroscope: torch.Tensor,
        x_magnetometer: torch.Tensor,
        x_velocity_gps: torch.Tensor,
        x_velocity_fusion: torch.Tensor,
        x_orientation_fusion: torch.Tensor,
        x_mask: torch.Tensor,
    ) -> torch.Tensor:

        """
        Forward function.

        Args:
            - x_accelerometer (torch.Tensor): The input tensor that contains accelerometer data of size (batch, seq, n_accelerometer) with 0 <= n_accelerometer <= 3.
            - x_gyroscope (torch.Tensor): The input tensor that contains gyroscope data of size (batch, seq, n_gyroscope) with 0 <= n_gyroscope <= 3.
            - x_magnetometer (torch.Tensor): The input tensor that contains magnetometer data of size (batch, seq, n_magnetometer) with 0 <= n_magnetometer <= 3.
            - x_velocity_gps (torch.Tensor): The input tensor that contains GPS velocity data of size (batch, seq, n_velocity_gps) with 0 <= n_velocity_gps <= 3.
            - x_velocity_fusion (torch.Tensor): The input tensor that contains Kalman fusion output velocity data of size (batch, seq, n_velocity_fusion) with 1 <= n_velocity_fusion <= 3.
            - x_orientation_fusion (torch.Tensor): The input tensor that contains Kalman fusion output orientation data of size (batch, seq, n_orientation_fusion) with 0 <= n_orientation_fusion <= 3.
            - x_mask (torch.Tensor): The input tensor that contains mask (GPS outage indication) data of size (batch, seq, 1)

        Returns
            - y (torch.Tensor): The output tensor of the layer of size (batch, seq, output_size) with 1 <= output_size <= 3.
        """

        y = self.__model(
            x_accelerometer=x_accelerometer,
            x_gyroscope=x_gyroscope,
            x_magnetometer=x_magnetometer,
            x_velocity_gps=x_velocity_gps,
            x_velocity_fusion=x_velocity_fusion,
            x_orientation_fusion=x_orientation_fusion,
            x_mask=x_mask
        )

        return y

    def state_dict(
        self: Self
    ) -> Dict[str, torch.Tensor]:

        """
        Return the state dict of the model.

        Returns:
            - state_dict (Dict[str, torch.Tensor]): The state dict of the model.
        """

        model_state_dict = self.__model.state_dict()

        return model_state_dict

    def load_state_dict(
        self: Self,
        state_dict: Dict[str, torch.Tensor]
    ) -> None:

        """
        Load the state dict of the model.

        Args:
            - state_dict (Dict[str, torch.Tensor]): The state dict of the model.
        """

        self.__model.load_state_dict(state_dict=state_dict)

    def parameters(
        self: Self
    ) -> Generator[torch.nn.parameter.Parameter, None, None]:

        """
        Returns the model parameters.

        Returns:
            - parameters (Generator[torch.nn.parameter.Parameter]): The model parameters.
        """

        parameters = self.__model.parameters()

        return parameters