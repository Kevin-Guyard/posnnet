import numpy as np
import torch
from typing import Self

from posnnet.models.submodules import ConvolutionalLongShortTermMemoryTransformerEncoderLayer


class ConvolutionalLongShortTermMemoryTransformerFrequencyAwareWithBypass(torch.nn.Module):

    def __init__(
        self: Self,
        input_size: int,
        output_size: int,
        d_model: int,
        n_head: int,
        hidden_size_ff: int,
        kernel_size_conv_1: int,
        kernel_size_conv_2: int,
        n_encoder_layers: int,
        n_decoder_layers: int,
        p_dropout: float
    ) -> None:

        """
        Initialize a Convolutional Long Short Term Memory Transformer Frequency Aware (CLSTMTFA) with a bypass path which follow the structure:
        signal encoder = -> Linear projection in encoder -> Position encoding -> Encoder layer 1 -> ... -> Encoder layer N -> signal memory
        FFT magnitude encoder = -> Linear projection in encoder -> Position encoding -> Encoder layer 1 -> ... -> Encoder layer N -> Magnitude memory
        FFT angle encoder = -> Linear projection in encoder -> Position encoding -> Encoder layer 1 -> ... -> Encoder layer N -> Angle memory
        Decoder memory = Concatenation of the signal memory, magnitude memory and angle memory
        Decoder = -> Linear projection in decoder -> Position encoding -> Decoder layer 1 -> ... -> Decoder layer N -> Add bypass -> Linear projection out ->

        where the bypass is a residual of the concatenation of the Kalman fusion output velocity and the mask (GPS outage indication).

        Every decoder layer follows the basic Transformer decoder structure.

        Every CLSTMT encoder layer follows the structure:
        -> CNN -> Add residual -> GELU -> Dropout -> LSTM -> Add residual -> LayerNorm -> Dropout -> Transformer encoder layer ->
        
        The CNN follows the structure:
        -> Conv1d -> BatchNorm1d -> GELU -> SpatialLocalSelfAttention -> Conv1d -> MaxPool -> BatchNorm1d -> GELU -> SpatialGlobalSelfAttention -> Conv1d -> BatchNorm1d
        
        The Transformer encoder layer follows the basic architecture of Transformer.

        Args:
            - input_size (int): The total size of the input (accelerometer + gyroscope + magnetometer + velocity GPS + velocity fusion + orientation fusion).
            - output_size (int): The total size of the output (velocity fusion).
            - d_model (int): The dimension of the channels for the CNN, the hidden size for the LSTM and the model for the Transformer encoder layer.
            - n_head (int): The number of parallel heads in the MultiHeadAttention of the Transformer encoder layer (d_model / n_head should be an integer).
            - hidden_size_ff (int): The size of the hidden layer of the feed forward of the Transformer encoder layer.
            - kernel_size_conv_1 (int): The size of the square kernel of the first convolution of the CNN.
            - kernel_size_conv_2 (int): The size of the square kernel of the second convolution of the CNN.
            - i_layer (int): The position of the layer in the global network (0 = first layer).
            - n_encoder_layers (int): The number of CLSTMT encoder layers.
            - n_decoder_layers (int): The number of Transformer decoder layers.
            - p_dropout (float): The probability of dropping a neuron during training.
        """

        super(ConvolutionalLongShortTermMemoryTransformerFrequencyAwareWithBypass, self).__init__()

        # Linear projection from input space to model space
        self.linear_projection_in_signal_encoder = torch.nn.Linear(in_features=input_size + 1, out_features=d_model) # +1 because of the mask in addition of the sensor features
        self.linear_projection_in_magnitude_encoder = torch.nn.Linear(in_features=input_size + 1, out_features=d_model) # +1 because of the mask in addition of the sensor features
        self.linear_projection_in_angle_encoder = torch.nn.Linear(in_features=input_size + 1, out_features=d_model) # +1 because of the mask in addition of the sensor features
        self.linear_projection_in_decoder = torch.nn.Linear(in_features=input_size + 1, out_features=d_model) # +1 because of the mask in addition of the sensor features

        # Positional encoding using LSTM
        self.lstm_position_encoding_signal_encoder = torch.nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=False
        )
        self.lstm_position_encoding_magnitude_encoder = torch.nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=False
        )
        self.lstm_position_encoding_angle_encoder = torch.nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=False
        )
        self.lstm_position_encoding_decoder = torch.nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=False
        )

        # Stack several CLSTMT encoder layers sequentially
        self.signal_encoder = torch.nn.Sequential(*[
            ConvolutionalLongShortTermMemoryTransformerEncoderLayer(
                d_model=d_model,
                n_head=n_head,
                hidden_size_ff=hidden_size_ff,
                kernel_size_conv_1=kernel_size_conv_1,
                kernel_size_conv_2=kernel_size_conv_2,
                i_layer=i_layer,
                p_dropout=p_dropout
            )
            for i_layer in range(n_encoder_layers)
        ])
        self.magnitude_encoder = torch.nn.Sequential(*[
            ConvolutionalLongShortTermMemoryTransformerEncoderLayer(
                d_model=d_model,
                n_head=n_head,
                hidden_size_ff=hidden_size_ff,
                kernel_size_conv_1=kernel_size_conv_1,
                kernel_size_conv_2=kernel_size_conv_2,
                i_layer=i_layer,
                p_dropout=p_dropout
            )
            for i_layer in range(n_encoder_layers)
        ])
        self.angle_encoder = torch.nn.Sequential(*[
            ConvolutionalLongShortTermMemoryTransformerEncoderLayer(
                d_model=d_model,
                n_head=n_head,
                hidden_size_ff=hidden_size_ff,
                kernel_size_conv_1=kernel_size_conv_1,
                kernel_size_conv_2=kernel_size_conv_2,
                i_layer=i_layer,
                p_dropout=p_dropout
            )
            for i_layer in range(n_encoder_layers)
        ])

        # Linear projection from the concatenation space (signal + magnitude + angle) 
        # to the model space (concatenation space size = 3 * model space size).
        self.linear_projection_encoder_to_decoder = torch.nn.Linear(
            in_features=3 * d_model,
            out_features=d_model
        )

        # Stack several Transformer decoder layers sequentially
        self.decoder = torch.nn.TransformerDecoder(
            decoder_layer=torch.nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=hidden_size_ff,
                dropout=p_dropout,
                activation="gelu",
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

        # For input projection, initialize the bias at 0 and the weights with xavier uniform
        torch.nn.init.xavier_uniform_(self.linear_projection_in_signal_encoder.weight)
        torch.nn.init.constant_(self.linear_projection_in_signal_encoder.bias, 0.0)

        torch.nn.init.xavier_uniform_(self.linear_projection_in_magnitude_encoder.weight)
        torch.nn.init.constant_(self.linear_projection_in_magnitude_encoder.bias, 0.0)

        torch.nn.init.xavier_uniform_(self.linear_projection_in_angle_encoder.weight)
        torch.nn.init.constant_(self.linear_projection_in_angle_encoder.bias, 0.0)
        
        torch.nn.init.xavier_uniform_(self.linear_projection_in_decoder.weight)
        torch.nn.init.constant_(self.linear_projection_in_decoder.bias, 0.0)

        # Initialize the linear layers of the LSTMs with xavier uniform for the weights and 0 for the bias
        for module in self.lstm_position_encoding_signal_encoder.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.constant_(module.bias, 0.0)

        for module in self.lstm_position_encoding_magnitude_encoder.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.constant_(module.bias, 0.0)

        for module in self.lstm_position_encoding_angle_encoder.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.constant_(module.bias, 0.0)

        for module in self.lstm_position_encoding_decoder.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.constant_(module.bias, 0.0)

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

        x_fft = torch.fft.fft(x, norm="forward", dim=1) # Compute the Fast Fourrier Transform (FFT) of the signal
        x_mag = torch.abs(x_fft) # Get the magnitude of the FFT
        x_mag = torch.cat([
            (x_mag[i, :, :] / x_mag[i, :, :].max(dim=0).values).unsqueeze(dim=0)
            for i in range(x_mag.size()[0])
        ], dim=0) # Scale
        x_angle = 0.5 + torch.angle(x_fft) / (2 * np.pi) # Get the angle of the FFT
        

        signal_memory = x # size = (batch_size, len_seq, input_size + 1)
        signal_memory = self.linear_projection_in_signal_encoder(signal_memory) # size = (batch_size, len_seq, d_model)
        signal_memory = signal_memory + self.lstm_position_encoding_signal_encoder(signal_memory)[0] # size = (batch_size, len_seq, d_model)
        signal_memory = self.signal_encoder(signal_memory) # size = (batch_size, len_seq, d_model)

        magnitude_memory = x # size = (batch_size, len_seq, input_size + 1)
        magnitude_memory = self.linear_projection_in_magnitude_encoder(magnitude_memory) # size = (batch_size, len_seq, d_model)
        magnitude_memory = magnitude_memory + self.lstm_position_encoding_magnitude_encoder(magnitude_memory)[0] # size = (batch_size, len_seq, d_model)
        magnitude_memory = self.magnitude_encoder(magnitude_memory) # size = (batch_size, len_seq, d_model)

        angle_memory = x # size = (batch_size, len_seq, input_size + 1)
        angle_memory = self.linear_projection_in_angle_encoder(angle_memory) # size = (batch_size, len_seq, d_model)
        angle_memory = angle_memory + self.lstm_position_encoding_angle_encoder(angle_memory)[0] # size = (batch_size, len_seq, d_model)
        angle_memory = self.angle_encoder(angle_memory) # size = (batch_size, len_seq, d_model)

        # Concatenate signal, FFT magnitude and FFT angle memories into a single memory.
        memory = torch.cat([signal_memory, magnitude_memory, angle_memory], dim=2) # size = (batch_size, len_seq, 3 * d_model)
        memory = self.linear_projection_encoder_to_decoder(memory) # size = (batch_size, len_seq, d_model)

        x = self.linear_projection_in_decoder(x) # size = (batch_size, len_seq, d_model)
        x = x + self.lstm_position_encoding_decoder(x)[0] # size = (batch_size, len_seq, d_model)
        x = self.decoder(tgt=x, memory=memory, tgt_mask=None, memory_mask=None) # size = (batch_size, len_seq, d_model)

        x = torch.cat([x, bypass], dim=2) # size = (batch_size, len_seq, d_model + output_size + 1)
        x = self.linear_projection_out(x) # size = (batch_size, len_seq, output_size)

        return x