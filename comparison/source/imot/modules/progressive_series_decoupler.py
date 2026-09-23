import torch
from typing import Self

from comparison.source.imot.modules.mlp import MLP
from comparison.source.imot.modules.series_breaker import SeriesBreaker


class ProgressiveSeriesDecoupler(torch.nn.Module):

    def __init__(self: Self, len_seq: int, p_dropout: float) -> None:

        super(ProgressiveSeriesDecoupler, self).__init__()

        self.series_breaker = SeriesBreaker()
        self.mlp_1 = MLP(dim=len_seq, n_layers=2, p_dropout=p_dropout)
        self.mlp_2 = MLP(dim=len_seq, n_layers=2, p_dropout=p_dropout)

    def forward(self: Self, a: torch.Tensor) -> torch.Tensor:

        a_s1, a_t1 = self.series_breaker(a)
        a_s1 = self.mlp_1(a_s1) + a_s1
        
        a_s2, a_t2 = self.series_breaker(a_s1)

        a_s = self.mlp_2(a_s2)
        a_t = a_t1 + a_t2
        a_tilde = torch.cat([a, a_s, a_t], dim=1)

        return a_tilde