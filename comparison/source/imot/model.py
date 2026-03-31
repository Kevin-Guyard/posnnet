import torch
from typing import Self, Tuple

from comparison.source.imot.modules.decoder import Decoder
from comparison.source.imot.modules.dynamic_scoring_mechanism import DynamicScoringMechanism
from comparison.source.imot.modules.encoder import Encoder


class IMOT(torch.nn.Module):

    def __init__(
        self: Self,
        n_dims: int=3,
        n_particules: int=128,
        len_seq: int=100,
        n_heads: int=4,
        n_encoder_layers: int=2,
        n_decoder_layers: int=2,
        p_dropout: float=0.1
    ) -> None:

        super(IMOT, self).__init__()

        self.encoder = Encoder(n_dims=n_dims, len_seq=len_seq, n_heads=n_heads, n_encoder_layers=n_encoder_layers, p_dropout=p_dropout)
        self.decoder = Decoder(n_dims=n_dims, n_particules=n_particules, len_seq=len_seq, n_heads=n_heads, n_decoder_layers=n_decoder_layers, p_dropout=p_dropout)
        self.dsm = DynamicScoringMechanism(n_particules=n_particules, p_dropout=p_dropout)

    def forward(self: Self, a_a: torch.Tensor, a_g: torch.Tensor) -> torch.Tensor:

        a_tilde, e_a_tilde = self.encoder(a_a=a_a, a_g=a_g)
        c_m, v_m = self.decoder(a_tilde=a_tilde, e_a_tilde=e_a_tilde)
        v = self.dsm(v_m=v_m)

        return v