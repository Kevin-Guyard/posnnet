import torch
from typing import Self

from posnnet.models.submodules import TemporalConvolutionalAttentiveNetworkLayer


class TemporalConvolutionalAttentiveNetworkWithBypass(torch.nn.Module):

    def __init__(
        self: Self,
        input_size: int,
        output_size: int,
        d_model: int,
        n_head: int,
        n_encoder_layers: int,
        p_dropout: float
    ) -> None:

        """
        Initialize a Temporal Convolutional Attentive Network (TCAN) with a bypass path which follows the structure:
        -> Linear projection in -> TCAN layer 1 -> ... -> TCAN layer N -> Add bypass -> Linear projection out ->
        where the bypass is a residual of the concatenation of the Kalman fusion output velocity and the mask (GPS outage indication).

        Every TCAN layer follows the structure:
        -> CNN -> Add residual -> MultiHeadAttention -> Add residual -> LayerNorm ->

        The CNN of every layer follows the structure:
        -> Conv1d -> BatchNorm -> GELU -> Dropout -> Conv1d -> BatchNorm -> GELU -> Dropout ->

        The convolution of the CNN use a dilation >= 1:
            - the first convolution has a dilation = 2 ** (2 * i_layer)
            - the second convolution has a dilation = 2 ** (2 * i_layer + 1)

        The MultiHeadAttention of every layer follows the basic architecture of Transformer.

        Args:
            - input_size (int): The total size of the input (accelerometer + gyroscope + magnetometer + velocity GPS + velocity fusion + orientation fusion).
            - output_size (int): The total size of the output (velocity fusion).
            - d_model (int): The dimension of the channels for the CNN and the model for the MultiHeadAttention.
            - n_head (int): The number of parallel head in the MultiHeadAttention (d_model / n_head should be an integer).
            - n_encoder_layers (int): The number of tcan encoder layers.
            - p_dropout (float): The probability of dropping a neuron during training.
        """

        super(TemporalConvolutionalAttentiveNetworkWithBypass, self).__init__()

        # Linear projection from input space to model space
        self.linear_projection_in = torch.nn.Linear(in_features=input_size + 1, out_features=d_model) # +1 because of the mask in addition of the sensor features

        # Stack several TCAN layers sequentially
        self.tcan_layers = torch.nn.Sequential(*[
            TemporalConvolutionalAttentiveNetworkLayer(
                d_model=d_model,
                n_head=n_head,
                i_layer=i_layer,
                p_dropout=p_dropout
            )
            for i_layer in range(n_encoder_layers)
        ])

        # Linear projection from model + bypass space to output space
        self.linear_projection_out = torch.nn.Linear(in_features=d_model + output_size + 1, out_features=output_size)

        # Initialize the weight of the network
        self.__init_weights()

    def __init_weights(
        self: Self
    ) -> None:

        """
        Initialize the weights of the network.
        """

        # For input projection, initialize the bias at 0 and the weights with xavier uniform
        torch.nn.init.xavier_uniform_(self.linear_projection_in.weight)
        torch.nn.init.constant_(self.linear_projection_in.bias, 0.0)
        
        # For output projection, initialize the bias at 0 and the weights with xavier uniform
        torch.nn.init.xavier_uniform_(self.linear_projection_out.weight)
        torch.nn.init.constant_(self.linear_projection_out.bias, 0.0)

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

        # Input sizes:
        #  - x_accelerometer: (batch_size, len_seq, n_features_accelerometer) with 0 <= n_features_accelerometer <= 3
        #  - x_gyroscope: (batch_size, len_seq, n_features_gyroscope) with 0 <= n_features_gyroscope <= 3
        #  - x_magnetometer: (batch_size, len_seq, n_features_magnetometer) with 0 <= n_features_magnetometer <= 3
        #  - x_velocity_gps: (batch_size, len_seq, n_features_velocity_gps) with 0 <= n_features_velocity_gps <= 3
        #  - x_velocity_fusion: (batch_size, len_seq, n_features_velocity_fusion) with 1 <= n_features_velocity_fusion <= 3
        #  - x_orientation_fusion: (batch_size, len_seq, n_features_orientation_fusion) with 0 <= n_features_orientation_fusion <= 3
        #  - x_accelerometer: (batch_size, len_seq, 1)

        x = torch.cat([
            x_accelerometer,
            x_gyroscope,
            x_magnetometer,
            x_velocity_gps,
            x_velocity_fusion,
            x_orientation_fusion,
            x_mask
        ], dim=2) # size = (batch_size, len_seq, input_size + 1) with 1 <= input_size <= 18

        bypass = torch.cat([
            x_velocity_fusion,
            x_mask
        ], dim=2) # size = (batch_size, len_seq, bypass_size) with 2 <= bypass_size <= 4

        x = self.linear_projection_in(x) # size = (batch_size, len_seq, d_model)
        x = self.tcan_layers(x) # size = (batch_size, len_seq, d_model)
        x = torch.cat([
            x,
            bypass
        ], dim=2) # size = (batch_size, len_seq, d_model + output_size + 1)
        x = self.linear_projection_out(x) # size = (batch_size, len_seq, output_size)

        return x