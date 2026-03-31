import torch
from typing import Self, Tuple

from comparison.source.imot.modules.adaptive_positional_encoding import AdaptivePositionalEncoding
from comparison.source.imot.modules.encoder_layer import EncoderLayer
from comparison.source.imot.modules.progressive_series_decoupler import ProgressiveSeriesDecoupler


class Encoder(torch.nn.Module):

    def __init__(self: Self, n_dims: int, len_seq: int, n_heads: int, n_encoder_layers: int, p_dropout: float) -> None:

        super(Encoder, self).__init__()

        self.n_dims = n_dims

        self.psd_accelerometer = ProgressiveSeriesDecoupler(len_seq=len_seq, p_dropout=p_dropout)
        self.psd_gyroscope = ProgressiveSeriesDecoupler(len_seq=len_seq, p_dropout=p_dropout)

        self.adaptive_positional_encoding = AdaptivePositionalEncoding(n_dims=n_dims, len_seq=len_seq, p_dropout=p_dropout)

        self.encoder_layers = torch.nn.ModuleList([
            EncoderLayer(n_dims=n_dims, len_seq=len_seq, n_heads=n_heads, p_dropout=p_dropout)
            for _ in range(n_encoder_layers)
        ])

    def forward(self: Self, a_a: torch.Tensor, a_g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        a_a_tilde = self.psd_accelerometer(a_a) # (B, 3D, T)
        a_g_tilde = self.psd_gyroscope(a_g) # (B, 3D, T)

        e_a_tilde = self.adaptive_positional_encoding(a_a_tilde=a_a_tilde, a_g_tilde=a_g_tilde) # (B, 6D, T)

        for encoder_layer in self.encoder_layers:
            a_tilde = encoder_layer(a_a_tilde=a_a_tilde, a_g_tilde=a_g_tilde, e_a_tilde=e_a_tilde) # (B,6D, T)
            a_a_tilde = a_tilde[:, 0 : 3 * self.n_dims, :] # (B, 3D, T)
            a_g_tilde = a_tilde[:, 3 * self.n_dims : 6 * self.n_dims, :] # (B, 3D, T)

        return a_tilde, e_a_tilde