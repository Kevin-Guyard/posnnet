import torch
from typing import Self


class SpatialEmbedding(torch.nn.Module):

    def __init__(
        self: Self,
        input_size: int,
        d_model: int,
        p_dropout: float
    ) -> None:

        """
        Initialize a spatial embedding layer which follows the structure:
        -> Conv1d -> BatchNorm -> Dropout ->

        Args:
            - input_size (int): The size of the input tensor.
            - d_model (int): The dimension of the channels.
            - p_dropout (float): The probability of dropping a neuron during training.
        """

        super(SpatialEmbedding, self).__init__()

        self.conv = torch.nn.Conv1d(
            in_channels=input_size,
            out_channels=d_model,
            kernel_size=1
        )
        self.batch_norm = torch.nn.BatchNorm1d(num_features=d_model)
        self.dropout = torch.nn.Dropout(p=p_dropout)

        # Initialize the weight of the layer
        self.__init_weights()

    def __init_weights(
        self: Self
    ) -> None:

        """
        Initialize the weights of the network.
        """

        # Initialize the convolution with xavier uniform for the weights and 0 for the bias
        torch.nn.init.xavier_uniform_(self.conv.weight)
        torch.nn.init.constant_(self.conv.bias, 0.0)

    def forward(
        self: Self,
        x: torch.Tensor
    ) -> torch.Tensor:

        """
        Forward function.

        Args:
            - x (torch.Tensor): The input tensor of the layer of size (batch_size, input_size, len_seq).

        Returns
            - x (torch.Tensor): The output tensor of the layer of size (batch_size, d_model, len_seq).
        """

        x = self.conv(x) # size = (batch_size, d_model, len_seq)
        x = self.batch_norm(x) # size = (batch_size, d_model, len_seq)
        x = self.dropout(x) # size = (batch_size, d_model, len_seq)

        return x