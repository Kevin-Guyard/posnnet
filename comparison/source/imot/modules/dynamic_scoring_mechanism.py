import torch
from typing import Self

from comparison.source.imot.modules.mlp import MLP


class DynamicScoringMechanism(torch.nn.Module):

    def __init__(self: Self, n_particules: int, p_dropout: float) -> None:

        super(DynamicScoringMechanism, self).__init__()

        self.mlp = MLP(dim=n_particules, n_layers=2, p_dropout=p_dropout)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self: Self, v_m: torch.Tensor) -> torch.Tensor:

        v_m = v_m.swapaxes(1, 2) # (B, D, P)
        S_d = self.mlp(v_m) # (B, D, P)
        W = self.softmax(S_d) # (B, D, P)
        v = (v_m * W).sum(dim=-1) # (B, D)

        return v