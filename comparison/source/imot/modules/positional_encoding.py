import torch
from typing import Self

from comparison.source.imot.modules.mlp import MLP


class PositionalEncoding(torch.nn.Module):

    def __init__(self: Self, n_dims: int, n_particules: int, len_seq: int, p_dropout: float) -> None:

        super(PositionalEncoding, self).__init__()

        self.pe = torch.nn.Parameter(torch.sin((2 * torch.pi) * torch.arange(0, n_particules, 1) / n_particules))
        self.projection = torch.nn.Linear(in_features=n_dims, out_features=len_seq)
        self.mlp = MLP(dim=len_seq, n_layers=2, p_dropout=p_dropout)

    def forward(self: Self, v_m: torch.Tensor) -> torch.Tensor:

        e_v = v_m + self.pe.unsqueeze(-1)
        e_v = self.projection(e_v)
        e_v = self.mlp(e_v)

        return e_v