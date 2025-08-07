import numpy as np
import torch
from typing import Self

from posnnet.models.submodules.spatial_embedding import SpatialEmbedding
from posnnet.models.submodules.spatial_encoder_layer import SpatialEncoderLayer
from posnnet.models.submodules.temporal_embedding import TemporalEmbedding


class SpatioTemporalTransformerFrequencyAwareWithBypass(torch.nn.Module):

    def __init__(
        self: Self,
        input_size: int,
        output_size: int,
        d_model: int,
        n_head: int,
        hidden_size_ff: int,
        n_encoder_layers: int,
        n_decoder_layers: int,
        p_dropout: float
    ) -> None:

        """
        Initialize a Spatio Temporal Transformer Frequency Aware (STTFA) with a bypass path which follows the structure:
        Signal encoder = -> Spatial embedding -> Spatial encoder layer 1 -> ... -> Spatial encoder layer N -> Signal memory
        FFT magnitude encoder = -> Spatial embedding -> Spatial encoder layer 1 -> ... -> Spatial encoder layer N -> Magnitude memory
        FFT angle encoder = -> Spatial embedding -> Spatial encoder layer 1 -> ... -> Spatial encoder layer N -> Angle memory
        Decoder memory = Concatenation of the signal memory, magnitude memory and angle memory
        Decoder = -> Temporal embedding -> Temporal decoder layer 1 -> ... -> Temporal decoder layer N -> Add bypass -> Linear projection out ->
        where the bypass is a residual of the concatenation of the Kalman fusion output velocity and the mask (GPS outage indication).

        The spatial embedding follows the structure:
        -> Conv1d -> BatchNorm -> Dropout ->

        The temporal embedding follows the structure:
        -> LSTM -> Dropout ->

        Every spatial encoder layer follows the structure:
        -> Conv1d -> BatchNorm1d -> GELU -> SpatialLocalSelfAttention -> Conv1d -> BatchNorm1d -> GELU -> SpatialGlobalSelfAttention -> Conv1d -> BatchNorm1d -> Add residual -> GELU -> Dropout ->

        Every temporal decoder layer follows the basic architecture of Transformer decoder.

        Args:
            - input_size (int): The total size of the input (accelerometer + gyroscope + magnetometer + velocity GPS + velocity fusion + orientation fusion).
            - output_size (int): The total size of the output (velocity fusion).
            - d_model (int): The dimension of the channels for the CNN and the model for the MultiHeadAttention.
            - n_head (int): The number of parallel head in the MultiHeadAttention (d_model / n_head should be an integer).
            - hidden_size_ff (int): The size of the hidden layer of the feed forward of the Transformer encoder layer.
            - n_encoder_layers (int): The number of spatial encoder layers.
            - n_decoder_layers (int): The number of temporal decoder layers.
            - p_dropout (float): The probability of dropping a neuron during training.
        """

        super(SpatioTemporalTransformerFrequencyAwareWithBypass, self).__init__()

        # Projection of the input into the embedding (model) space
        self.spatial_signal_embedding = SpatialEmbedding(
            input_size=input_size + 1,
            d_model=d_model,
            p_dropout=p_dropout
        )
        self.spatial_magnitude_embedding = SpatialEmbedding(
            input_size=input_size + 1,
            d_model=d_model,
            p_dropout=p_dropout
        )
        self.spatial_angle_embedding = SpatialEmbedding(
            input_size=input_size + 1,
            d_model=d_model,
            p_dropout=p_dropout
        )
        
        # Stack several spatial encoder layers sequentially
        self.spatial_signal_encoder = torch.nn.Sequential(*[
            SpatialEncoderLayer(
                d_model=d_model,
                p_dropout=p_dropout
            )
            for _ in range(n_encoder_layers)
        ])
        self.spatial_magnitude_encoder = torch.nn.Sequential(*[
            SpatialEncoderLayer(
                d_model=d_model,
                p_dropout=p_dropout
            )
            for _ in range(n_encoder_layers)
        ])
        self.spatial_angle_encoder = torch.nn.Sequential(*[
            SpatialEncoderLayer(
                d_model=d_model,
                p_dropout=p_dropout
            )
            for _ in range(n_encoder_layers)
        ])

        # Linear projection from the concatenation space (temporal + magnitude + angle) 
        # to the model space (concatenation space size = 3 * model space size).
        self.linear_projection_encoder_to_decoder = torch.nn.Linear(
            in_features=3 * d_model,
            out_features=d_model
        )

        # Projection of the input into the embedding (model) space
        self.temporal_embedding = TemporalEmbedding(
            input_size=input_size + 1,
            d_model=d_model,
            p_dropout=p_dropout
        )
        # Stack several temporal decoder layers sequentially
        self.temporal_decoder = torch.nn.TransformerDecoder(
            decoder_layer=torch.nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=hidden_size_ff,
                dropout=p_dropout,
                batch_first=True
            ),
            num_layers=n_decoder_layers,
            norm=None
        )

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

        # For encode to decoder projection, initialize the bias at 0 and the weights with xavier uniform
        torch.nn.init.xavier_uniform_(self.linear_projection_encoder_to_decoder.weight)
        torch.nn.init.constant_(self.linear_projection_encoder_to_decoder.bias, 0.0)
        
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

        x_temporal = torch.cat([
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

        x_fft = torch.fft.fft(x_temporal, norm="forward", dim=1) # Compute the Fast Fourrier Transform (FFT) of the signal
        x_magnitude = torch.abs(x_fft) # Get the magnitude of the FFT
        x_magnitude = torch.cat([
            (x_magnitude[i, :, :] / x_magnitude[i, :, :].max(dim=0).values).unsqueeze(dim=0)
            for i in range(x_magnitude.size()[0])
        ], dim=0) # Scale
        x_angle = 0.5 + torch.angle(x_fft) / (2 * np.pi) # Get the angle of the FFT

        x_spatial_signal = x_temporal.swapaxes(1, 2) # size = (batch_size, input_size + 1, len_seq)
        x_spatial_signal = self.spatial_signal_embedding(x_spatial_signal) # size = (batch_size, d_model, len_seq)
        x_spatial_signal = self.spatial_signal_encoder(x_spatial_signal) # size = (batch_size, d_model, len_seq)
        x_spatial_signal = x_spatial_signal.swapaxes(1, 2) # size = (batch_size, len_seq, d_model)

        x_spatial_magnitude = x_magnitude.swapaxes(1, 2) # size = (batch_size, input_size + 1, len_seq)
        x_spatial_magnitude = self.spatial_magnitude_embedding(x_spatial_magnitude) # size = (batch_size, d_model, len_seq)
        x_spatial_magnitude = self.spatial_magnitude_encoder(x_spatial_magnitude) # size = (batch_size, d_model, len_seq)
        x_spatial_magnitude = x_spatial_magnitude.swapaxes(1, 2) # size = (batch_size, len_seq, d_model)

        x_spatial_angle = x_angle.swapaxes(1, 2) # size = (batch_size, input_size + 1, len_seq)
        x_spatial_angle = self.spatial_angle_embedding(x_spatial_angle) # size = (batch_size, d_model, len_seq)
        x_spatial_angle = self.spatial_angle_encoder(x_spatial_angle) # size = (batch_size, d_model, len_seq)
        x_spatial_angle = x_spatial_angle.swapaxes(1, 2) # size = (batch_size, len_seq, d_model)

        # Concatenate temporal, FFT magnitude and FFT angle memories into a single memory.
        x_spatial = torch.cat([x_spatial_signal, x_spatial_magnitude, x_spatial_angle], dim=2) # size = (batch_size, len_seq, 3 * d_model)
        x_spatial = self.linear_projection_encoder_to_decoder(x_spatial) # size = (batch_size, len_seq, d_model)

        x_temporal = self.temporal_embedding(x_temporal) # size = (batch_size, len_seq, d_model)
        x_temporal = self.temporal_decoder(tgt=x_temporal, memory=x_spatial, tgt_mask=None, memory_mask=None) # size = (batch_size, len_seq, d_model)

        x = torch.cat([
            x_temporal,
            bypass
        ], dim=2) # size = (batch_size, len_seq, d_model + output_size + 1)
        x = self.linear_projection_out(x) # size = (batch_size, len_seq, output_size)

        return x