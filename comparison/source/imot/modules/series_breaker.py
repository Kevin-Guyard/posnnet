import torch
from typing import Self


class SeriesBreaker(torch.nn.Module):

    def __init__(self: Self, k1: int=9, k2: int=3) -> None:

        super(SeriesBreaker, self).__init__()

        self.k1 = k1
        self.k2 = k2

    def _avg_pool_1d(self: Self, x: torch.Tensor, k: int) -> torch.Tensor:

        pad = (k - 1) // 2

        x = torch.nn.functional.pad(x, (pad, pad), mode="replicate")
        x = torch.nn.functional.avg_pool1d(x, kernel_size=k, stride=1)

        return x

    def forward(self: Self, a: torch.Tensor) -> torch.Tensor:

        a_t = self._avg_pool_1d(a, k=self.k1)
        a_t = self._avg_pool_1d(a_t, k=self.k2)
        a_s = a - a_t
        
        return a_s, a_t