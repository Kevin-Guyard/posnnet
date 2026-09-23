import torch
from typing import Self

from comparison.source.ctin.local_self_attention import LocalSelfAttention
from comparison.source.ctin.global_self_attention import GlobalSelfAttention


class SpatialEncoderLayer(torch.nn.Module):

    def __init__(
        self: Self,
        d_model: int,
        p_dropout: float
    ) -> None:

        super(SpatialEncoderLayer, self).__init__()

        self.block_1 = torch.nn.Sequential(
            (torch.nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)),
            (torch.nn.BatchNorm1d(num_features=d_model)),
            (torch.nn.ReLU())
        )

        self.local_self_attention = LocalSelfAttention(d_model=d_model)

        self.block_2 = torch.nn.Sequential(
            (torch.nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)),
            (torch.nn.BatchNorm1d(num_features=d_model)),
            (torch.nn.ReLU())
        )

        self.global_self_attention = GlobalSelfAttention(d_model=d_model)

        self.block_3 = torch.nn.Sequential(
            (torch.nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)),
            (torch.nn.BatchNorm1d(num_features=d_model))
        )

        self.dropout = torch.nn.Dropout(p=p_dropout)

        self.activation_out = torch.nn.ReLU()

    def forward(
        self: Self,
        x: torch.Tensor
    ) -> torch.Tensor:

        residual = x

        x = self.block_1(x)
        x = self.local_self_attention(x)
        x = self.block_2(x)
        x = self.global_self_attention(x)
        x = self.block_3(x)
        x = self.dropout(x)
        x = x + residual
        x = self.activation_out(x)

        return x