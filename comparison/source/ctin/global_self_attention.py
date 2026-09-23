import torch
from typing import Self


class GlobalSelfAttention(torch.nn.Module):

    def __init__(
        self: Self,
        d_model: int
    ) -> None:

        super(GlobalSelfAttention, self).__init__()

        self.conv_k = torch.nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1
        )

        self.conv_q = torch.nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1
        )

        self.conv_v = torch.nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1
        )

    def forward(
        self: Self,
        x: torch.Tensor
    ) -> torch.Tensor:

        k, q, v = x, x, x

        k = self.conv_k(k)
        q = self.conv_q(q)
        v = self.conv_v(v)

        kq = k * q
        kq = torch.softmax(kq, dim=1)

        out = kq * v

        return out