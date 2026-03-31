import torch
from typing import Self


class LocalSelfAttention(torch.nn.Module):

    def __init__(
        self: Self,
        d_model: int
    ) -> None:

        super(LocalSelfAttention, self).__init__()

        self.conv_k = torch.nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding="same"
        )

        self.conv_c1q = torch.nn.Conv1d(
            in_channels=2 * d_model,
            out_channels=d_model,
            kernel_size=1
        )

        self.activation_c1q = torch.nn.ReLU()

        self.conv_v = torch.nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1
        )

        self.conv_c1qc2 = torch.nn.Conv1d(
            in_channels=2 * d_model,
            out_channels=d_model,
            kernel_size=1
        )

    def forward(
        self: Self,
        x: torch.Tensor
    ) -> torch.Tensor:

        k, q, v = x, x, x

        c1 = self.conv_k(k)

        c1q = torch.cat([c1, q], dim=1)
        c1q = self.conv_c1q(c1q)
        c1q = self.activation_c1q(c1q)

        v = self.conv_v(v)
        c2 = c1q * v

        c1qc2 = torch.cat([c1, c2], dim=1)
        out = self.conv_c1qc2(c1qc2)

        return out