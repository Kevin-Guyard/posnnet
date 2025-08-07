import torch
from typing import Self


class SpatialLocalSelfAttention(torch.nn.Module):

    def __init__(
        self: Self,
        d_model: int,
        p_dropout: float
    ) -> None:

        """
        Initialize a spatial local self attention layer. The local self attention is computed with a view (batch, features, seq).

        k, q, v = input
        v = conv(v)
        kq = relu(conv(cat(q, conv(k))))
        kqv = kq * v

        then basic dot product attention with query = kqv, key = k and value = kqv

        Args:
            - d_model (int): The dimension of the channels.
            - p_dropout (float): The probability of dropping a neuron during training.
        """

        super(SpatialLocalSelfAttention, self).__init__()

        # Convolution for the key
        self.conv_k = torch.nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1
        )

        # Convolution for the value
        self.conv_v = torch.nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1,
        )

        # Convolution for the product of the key and the query
        self.conv_kq = torch.nn.Conv1d(
            in_channels=2 * d_model,
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

            # For convolutional layer, initialize the bias at 0 and the weights with kaiming normal
            if isinstance(module, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
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
        v = self.conv_v(v) # Convolution of the value , v size = (batch_size, d_model, len_seq)

        kq = torch.cat([k, q], dim=1) # Concatenation of the key with the query on the channel dimension, kq size = (batch_size, 2 * d_model, len_seq)
        kq = self.conv_kq(kq) # Convolution of the concatenate key/query, kq size = (batch_size, d_model, len_seq)
        kq = torch.relu(kq) # Relu activation
        kqv = kq * v # Product of the key/query with the value

        att_out = torch.nn.functional.scaled_dot_product_attention(query=kqv, key=k, value=kqv) # Basic dot product attention
        att_out = self.dropout(att_out)

        return att_out