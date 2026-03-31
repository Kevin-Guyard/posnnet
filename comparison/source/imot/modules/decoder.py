import torch
from typing import Self, Tuple

from comparison.source.imot.modules.decoder_layer import DecoderLayer


class Decoder(torch.nn.Module):

    def __init__(self: Self, n_dims: int, n_particules: int, len_seq: int, n_heads: int, n_decoder_layers: int, p_dropout: float) -> None:

        super(Decoder, self).__init__()

        self.n_particules = n_particules
        self.len_seq = len_seq

        self.v_m = torch.nn.Parameter(torch.zeros(1, n_particules, n_dims))

        self.decoder_layers = torch.nn.ModuleList([
            DecoderLayer(n_dims=n_dims, n_particules=n_particules, len_seq=len_seq, n_heads=n_heads, p_dropout=p_dropout)
            for _ in range(n_decoder_layers)
        ])

    def forward(self: Self, a_tilde: torch.Tensor, e_a_tilde: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size = a_tilde.size(dim=0)
        
        c_m = torch.zeros((batch_size, self.n_particules,self. len_seq), device="cuda" if torch.cuda.is_available() else "cpu")
        v_m = self.v_m

        for decoder_layer in self.decoder_layers:

            c_m, v_m = decoder_layer(c_m=c_m, v_m=v_m, a_tilde=a_tilde, e_a_tilde=e_a_tilde)

        return c_m, v_m