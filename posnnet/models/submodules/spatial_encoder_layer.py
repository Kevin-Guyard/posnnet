import torch
from typing import Self

from posnnet.models.submodules.spatial_global_self_attention import SpatialGlobalSelfAttention
from posnnet.models.submodules.spatial_local_self_attention import SpatialLocalSelfAttention


class SpatialEncoderLayer(torch.nn.Module):

    def __init__(
        self: Self,
        d_model: int,
        p_dropout: float
    ) -> None:

        """
        Initialize a spatial encoder layer which follows the structure:
        -> Conv1d -> BatchNorm1d -> GELU -> SpatialLocalSelfAttention -> Conv1d -> BatchNorm1d -> GELU -> SpatialGlobalSelfAttention -> Conv1d -> BatchNorm1d -> Add residual -> GELU -> Dropout ->

        Args:
            - d_model (int): The dimension of the channels.
            - p_dropout (float): The probability of dropping a neuron during training.
        """

        super(SpatialEncoderLayer, self).__init__()

        # Conv1d -> BatchNorm1d -> GELU -> SpatialLocalSelfAttention -> Conv1d -> BatchNorm1d -> GELU -> SpatialGlobalSelfAttention -> Conv1d -> BatchNorm1d
        self.cnn = torch.nn.Sequential(*[
            torch.nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=1
            ),
            torch.nn.BatchNorm1d(num_features=d_model),
            torch.nn.GELU(),
            SpatialLocalSelfAttention(d_model=d_model, p_dropout=p_dropout),
            torch.nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=1
            ),
            torch.nn.BatchNorm1d(num_features=d_model),
            torch.nn.GELU(),
            SpatialGlobalSelfAttention(d_model=d_model, p_dropout=p_dropout),
            torch.nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=1
            ),
            torch.nn.BatchNorm1d(num_features=d_model)
        ])
        # Activation and dropout separate because their are applied after residual connection add.
        self.activation = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(p=p_dropout)

        # Initialize the weight of the layer
        self.__init_weights()

    def __init_weights(
        self: Self
    ) -> None:

        """
        Initialize the weights of the network.
        """

        # Initialize the CNN convolution with xavier uniform for the weights and 0 for the bias
        torch.nn.init.xavier_uniform_(self.cnn[0].weight) # First convolution
        torch.nn.init.constant_(self.cnn[0].bias, 0.0)
        
        torch.nn.init.xavier_uniform_(self.cnn[4].weight) # Second convolution
        torch.nn.init.constant_(self.cnn[4].bias, 0.0)
        
        torch.nn.init.xavier_uniform_(self.cnn[8].weight) # Third convolution
        torch.nn.init.constant_(self.cnn[8].bias, 0.0)

    def forward(
        self: Self,
        x: torch.Tensor
    ) -> torch.Tensor:

        """
        Forward function.

        Args:
            - x (torch.Tensor): The input tensor of the layer of size (batch_size, d_model, len_seq).

        Returns
            - x (torch.Tensor): The output tensor of the layer of size (batch_size, d_model, len_seq).
        """

        x = x + self.cnn(x) # Apply the convolution on the sequence axe with the features as channels / size = (batch_size, d_model, len_seq)
        x = self.activation(x) # size = (batch_size, d_model, len_seq)
        x = self.dropout(x) # size = (batch_size, d_model, len_seq)

        return x