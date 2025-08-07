import torch
from typing import Self


class SpatialGlobalSelfAttention(torch.nn.Module):

    def __init__(
        self: Self,
        d_model: int,
        p_dropout: float
    ) -> None:

        """
        Initialize a spatial global self attention layer. The global self attention is computed with a view (batch, features, seq).

        k, q, v = input
        k = conv(k)
        q = conv(q)
        v = conv(v)

        att = softmax(k * q) * v (with the softmax on features dimension)

        Args:
            - d_model (int): The dimension of the channels.
            - p_dropout (float): The probability of dropping a neuron during training.
        """

        super(SpatialGlobalSelfAttention, self).__init__()

        # Convoltion for the key
        self.conv_k = torch.nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1
        )

        # Convolution for the query
        self.conv_q = torch.nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1
        )

        # Convolution for the value
        self.conv_v = torch.nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1
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

        # Iterate over modules
        for module in self.modules():

            # For convolutional layer, initialize the bias at 0 and the weights with xavier uniform
            if isinstance(module, torch.nn.Conv1d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)

            else:
                pass

    def forward(
        self: Self,
        x: torch.Tensor
    ) -> torch.Tensor:

        """
        Forward function

        Args:
            - x (torch.Tensor): The input tensor of the layer of size (batch_size, d_model, len_seq).

        Returns
            - att_out (torch.Tensor): The output tensor of the layer of size (batch_size, d_model, len_seq).
        """

        k, q, v = x, x, x # Self attention, input is duplicated 3 times for key, query and value, size = (batch_size, d_model, len_seq)

        k = self.conv_k(k) # Convolution of the key, k size = (batch_size, d_model, len_seq)
        q = self.conv_q(q) # Convolution of the query, q size = (batch_size, d_model, len_seq)
        v = self.conv_v(v) # Convolution of the value, v size = (batch_size, d_model, len_seq)

        kq = k * q # Product of the key with the query
        kq = torch.softmax(kq, dim=1) # Softmax of the product on the channel dimension

        att_out = kq * v # Product of the key/query with the value
        att_out = self.dropout(att_out)

        return att_out