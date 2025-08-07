import torch
from typing import Self

from posnnet.models.submodules.spatial_global_self_attention import SpatialGlobalSelfAttention
from posnnet.models.submodules.spatial_local_self_attention import SpatialLocalSelfAttention


class ConvolutionalLongShortTermMemoryTransformerEncoderLayer(torch.nn.Module):

    def __init__(
        self: Self,
        d_model: int,
        n_head: int,
        hidden_size_ff: int,
        kernel_size_conv_1: int,
        kernel_size_conv_2: int,
        i_layer: int,
        p_dropout: float
    ) -> None:

        """
        Initialize a Convolutional Long Short Term Memory Transformer Encoder (CLSTMTE) layer which follow the structure:
        -> CNN -> Add residual -> GELU -> Dropout -> LSTM -> Add residual -> LayerNorm -> Dropout -> Transformer encoder layer ->
        
        The CNN follows the structure:
        -> Conv1d -> BatchNorm1d -> GELU -> SpatialLocalSelfAttention -> Conv1d -> MaxPool -> BatchNorm1d -> GELU -> SpatialGlobalSelfAttention -> Conv1d -> BatchNorm1d
        
        The Transformer encoder layer follows the basic architecture of Transformer.

        Args:
            - d_model (int): The dimension of the channels for the CNN, the hidden size for the LSTM and the model for the Transformer encoder layer.
            - n_head (int): The number of parallel heads in the MultiHeadAttention of the Transformer encoder layer (d_model / n_head should be an integer).
            - hidden_size_ff (int): The size of the hidden layer of the feed forward of the Transformer encoder layer.
            - kernel_size_conv_1 (int): The size of the square kernel of the first convolution of the CNN.
            - kernel_size_conv_2 (int): The size of the square kernel of the second convolution of the CNN.
            - i_layer (int): The position of the layer in the global network (0 = first layer).
            - p_dropout (float): The probability of dropping a neuron during training.
        """

        super(ConvolutionalLongShortTermMemoryTransformerEncoderLayer, self).__init__()

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=kernel_size_conv_1,
                padding="same",
                bias=False
            ),
            torch.nn.BatchNorm1d(num_features=d_model),
            torch.nn.GELU(),
            SpatialLocalSelfAttention(d_model=d_model, p_dropout=p_dropout),
            torch.nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=kernel_size_conv_2,
                padding="same",
                bias=False
            ),
            torch.nn.MaxPool1d(kernel_size=2, ceil_mode=True),
            torch.nn.BatchNorm1d(num_features=d_model),
            torch.nn.GELU(),
            SpatialGlobalSelfAttention(d_model=d_model, p_dropout=p_dropout),
            torch.nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=1,
                bias=False
            ),
            torch.nn.BatchNorm1d(num_features=d_model)
        )
        self.activation_out_cnn = torch.nn.GELU()
        self.dropout_out_cnn = torch.nn.Dropout(p=p_dropout)
        self.cnn_residual_scaling = torch.nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1,
            stride=2
        )

        self.lstm = torch.nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=True
        )
        self.linear_projection_out_lstm = torch.nn.Linear(in_features=2 * d_model, out_features=d_model)
        self.layer_norm_out_lstm = torch.nn.LayerNorm(normalized_shape=d_model)
        self.activation_out_lstm = torch.nn.GELU()
        self.dropout_out_lstm = torch.nn.Dropout(p=p_dropout)

        self.transformer_encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=hidden_size_ff,
            dropout=p_dropout,
            activation="gelu",
            batch_first=True
        )

        # Initialize the weight of the layer
        self.__init_weights()

    def __init_weights(
        self: Self
    ) -> None:

        """
        Initialize the weights of the network.
        """

        # Initialize the CNN convolution with xavier uniform for the weights
        torch.nn.init.xavier_uniform_(self.cnn[0].weight) # First convolution
        torch.nn.init.xavier_uniform_(self.cnn[4].weight) # Second convolution
        torch.nn.init.xavier_uniform_(self.cnn[9].weight) # Third convolution

        # Initialize the CNN residual scaling with xavier uniform for the weights and 0 for the bias
        torch.nn.init.xavier_uniform_(self.cnn_residual_scaling.weight) # First convolution
        torch.nn.init.constant_(self.cnn_residual_scaling.bias, 0.0) # First convolution

        # Initialize the LSTM linear layers with xavier uniform for the weights and 0 for the bias
        for module in self.lstm.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.constant_(module.bias, 0.0)

        # Initialize the LSTM output projection with xavier uniform for the weights and 0 for the bias
        torch.nn.init.xavier_uniform_(self.linear_projection_out_lstm.weight) # First convolution
        torch.nn.init.constant_(self.linear_projection_out_lstm.bias, 0.0) # First convolution

        # Initialize the LSTM output LayerNorm with 1 for the weights and 0 for the bias
        torch.nn.init.constant_(self.layer_norm_out_lstm.weight, 1.0)
        torch.nn.init.constant_(self.layer_norm_out_lstm.bias, 0.0)

        # Initialize the Transformer encoder linear layers with xavier uniform for the weights and 0 for the bias
        for module in self.transformer_encoder_layer.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight) # First convolution
                torch.nn.init.constant_(module.bias, 0.0) # First convolution

    def forward(
        self: Self,
        x: torch.Tensor
    ) -> torch.Tensor:

        """
        Forward function.

        Args:
            - x (torch.Tensor): The input tensor of the layer of size (batch_size, len_seq, d_model).

        Returns
            - x (torch.Tensor): The output tensor of the layer of size (batch_size, len_seq, d_model).
        """

        # x size = (batch_size, len_seq, d_model)
        x = x.swapaxes(1, 2) # size = (batch_size, d_model, len_seq)
        x = self.cnn(x) + self.cnn_residual_scaling(x) # apply cnn + scale residual (because of the Pooling in the CNN which reduce sequence length)
        x = self.activation_out_cnn(x)
        x = self.dropout_out_cnn(x)
        x = x.swapaxes(1, 2) # size = (batch_size, len_seq, d_model)

        lstm_residual = x
        x, _ = self.lstm(x) # size = (batch_size, len_seq, 2 * d_model) (output dim doubled due to the bidirectionality of the LSTM)
        x = self.linear_projection_out_lstm(x) # size = (batch_size, len_seq, d_model)
        x = x + lstm_residual
        x = self.layer_norm_out_lstm(x)
        x = self.dropout_out_lstm(x)

        x = self.transformer_encoder_layer(x) # size = (batch_size, len_seq, d_model)

        return x