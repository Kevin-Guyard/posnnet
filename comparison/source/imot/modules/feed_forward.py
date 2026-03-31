import torch
from typing import Self


class FeedForward(torch.nn.Module):

    def __init__(self: Self, dim: int, p_dropout: float) -> None:

        super(FeedForward, self).__init__()

        self.ff = torch.nn.Sequential(
            torch.nn.Linear(in_features=dim, out_features=4 * dim),
            torch.nn.GELU(),
            torch.nn.Dropout(p=p_dropout),
            torch.nn.Linear(in_features=4 * dim, out_features=dim)
        )

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:

        x = self.ff(x)

        return x