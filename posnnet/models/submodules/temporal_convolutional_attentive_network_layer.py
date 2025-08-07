import torch
from typing import Self


class TemporalConvolutionalAttentiveNetworkLayer(torch.nn.Module):

    def __init__(
        self: Self,
        d_model: int,
        n_head: int,
        i_layer: int,
        p_dropout: float
    ) -> None:

        """
        Initialize a Temporal Convolutional Attentive Network (TCAN) layer which follows the structure:
        -> CNN -> Add residual -> MultiHeadAttention -> Add residual -> LayerNorm ->

        The CNN follows the structure:
        -> Conv1d -> BatchNorm -> GELU -> Dropout -> Conv1d -> BatchNorm -> GELU -> Dropout ->

        The convolutions of the CNN use a dilation >= 1:
            - the first convolution has a dilation = 2 ** (2 * i_layer)
            - the second convolution has a dilation = 2 ** (2 * i_layer + 1)

        The MultiHeadAttention follows the basic architecture of Transformer.

        Args:
            - d_model (int): The dimension of the channels for the CNN and the model for the MultiHeadAttention.
            - n_head (int): The number of parallel heads in the MultiHeadAttention (d_model / n_head should be an integer).
            - i_layer (int): The position of the layer in the global network (0 = first layer).
            - p_dropout (float): The probability of dropping a neuron during training.
        """

        super(TemporalConvolutionalAttentiveNetworkLayer, self).__init__()

        # The CNN part uses the following structure:
        # Conv1d -> BatchNorm -> GELU -> Dropout -> Conv1d -> BatchNorm -> GELU -> Dropout
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=3,
                stride=1,
                padding="same",
                dilation=2 ** (2 * i_layer)
            ),
            torch.nn.BatchNorm1d(num_features=d_model),
            torch.nn.GELU(),
            torch.nn.Dropout(p=p_dropout),
            torch.nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=3,
                stride=1,
                padding="same",
                dilation=2 ** (2 * i_layer + 1)
            ),
            torch.nn.BatchNorm1d(num_features=d_model),
            torch.nn.GELU(),
            torch.nn.Dropout(p=p_dropout)
        )

        # Basis multi head attention
        self.multi_head_attention = torch.nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            dropout=p_dropout,
            batch_first=True
        )

        # Basic LayerNorm
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=d_model)

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

            # For linear or convolutional layer, initialize the bias at 0 and the weights with xavier uniform
            if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv1d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)

            # For layer normalization, initialize the bias at 0 and the weights at 1
            elif isinstance(module, torch.nn.LayerNorm):
                torch.nn.init.constant_(module.weight, 1.0)
                torch.nn.init.constant_(module.bias, 0.0)

            else:
                pass

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
        x = x.swapaxes(1, 2) # (batch_size, d_model, len_seq)
        x = x + self.cnn(x) # Apply the convolution on the sequence axe with the features as channels
        x = x.swapaxes(1, 2) # (batch_size, len_seq, d_model)
        x = x + self.multi_head_attention(query=x, key=x, value=x, need_weights=False)[0] # self attention
        x = self.layer_norm(x)

        return x