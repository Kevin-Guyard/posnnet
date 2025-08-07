import torch
from typing import Self


class TemporalEmbedding(torch.nn.Module):

    def __init__(
        self: Self,
        input_size: int,
        d_model: int,
        p_dropout: float
    ) -> None:

        """
        Initialize a temporal embedding layer which follows the structure:
        -> LSTM -> Dropout ->

        Args:
            - input_size (int): The size of the input tensor.
            - d_model (int): The dimension of the hidden size of the LSTM.
            - p_dropout (float): The probability of dropping a neuron during training.
        """

        super(TemporalEmbedding, self).__init__()

        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=d_model,
            batch_first=True
        )
        self.dropout = torch.nn.Dropout(p=p_dropout)

        # Initialize the weight of the layer
        self.__init_weights()

    def __init_weights(
        self: Self
    ) -> None:

        """
        Initialize the weights of the network.
        """

        # Initialize the linear layers of the LSTM with xavier uniform for the weights and 0 for the bias
        for module in self.lstm.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.constant_(module.bias, 0.0)

    def forward(
        self: Self,
        x: torch.Tensor
    ) -> torch.Tensor:

        """
        Forward function.

        Args:
            - x (torch.Tensor): The input tensor of the layer of size (batch_size, len_seq, input_size).

        Returns
            - x (torch.Tensor): The output tensor of the layer of size (batch_size, len_seq, d_model).
        """

        x, _ = self.lstm(x) # size = (batch_size, len_seq, d_model)
        x = self.dropout(x) # size = (batch_size, len_seq, d_model)

        return x