import math
import torch
from typing import Self

from comparison.source.imot.modules.mlp import MLP


class AdaptivePositionalEncoding(torch.nn.Module):

    def __init__(self: Self, n_dims: int, len_seq: int, p_dropout: float) -> None:

        super(AdaptivePositionalEncoding, self).__init__()

        self.mlp_acc = MLP(dim=len_seq, n_layers=2, p_dropout=p_dropout)
        self.mlp_gyro = MLP(dim=len_seq, n_layers=2, p_dropout=p_dropout)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        dtype = torch.float32

        self.e_a = torch.nn.Parameter(torch.sin((2 * torch.pi) * torch.arange(0, len_seq, 1) / len_seq))

    def forward(self: Self, a_a_tilde: torch.Tensor, a_g_tilde: torch.Tensor) -> torch.Tensor:

        a_a_tilde = self.mlp_acc(a_a_tilde) * self.e_a
        a_g_tilde = self.mlp_gyro(a_g_tilde) * self.e_a

        e_a_tilde = torch.cat([a_a_tilde, a_g_tilde], dim=1)

        return e_a_tilde