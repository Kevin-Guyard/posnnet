import torch
from typing import Self, Tuple

from comparison.source.imot.modules.feed_forward import FeedForward
from comparison.source.imot.modules.mlp import MLP
from comparison.source.imot.modules.multihead_cross_attention import MultiHeadCrossAttention
from comparison.source.imot.modules.positional_encoding import PositionalEncoding


class DecoderLayer(torch.nn.Module):

    def __init__(self: Self, n_dims: int, n_particules: int, len_seq: int, n_heads: int, p_dropout: float) -> None:

        super(DecoderLayer, self).__init__()

        self.positional_encoding = PositionalEncoding(n_dims=n_dims, n_particules=n_particules, len_seq=len_seq, p_dropout=p_dropout)
        self.mlp_c_m = MLP(dim=len_seq, n_layers=2, p_dropout=p_dropout)
        self.mlp_out = MLP(dim=len_seq, n_layers=2, p_dropout=p_dropout)
        self.linear_proj_out = torch.nn.Linear(in_features=len_seq, out_features=n_dims)

        self.multihead_self_attention = torch.nn.MultiheadAttention(
            embed_dim=len_seq,
            num_heads=n_heads,
            dropout=p_dropout,
            batch_first=True
        )
        self.multihead_cross_attention = MultiHeadCrossAttention(
            n_dims=n_dims,
            n_particules=n_particules,
            len_seq=len_seq,
            n_heads=n_heads,
            p_dropout=p_dropout
        )
        self.feed_forward = FeedForward(dim=len_seq, p_dropout=p_dropout)

        self.layer_norm_1 = torch.nn.LayerNorm(normalized_shape=len_seq)
        self.layer_norm_2 = torch.nn.LayerNorm(normalized_shape=len_seq)
        self.layer_norm_3 = torch.nn.LayerNorm(normalized_shape=len_seq)

    def forward(self: Self, c_m: torch.Tensor, v_m: torch.Tensor, a_tilde: torch.Tensor, e_a_tilde: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        e_v = self.positional_encoding(v_m) # (B, P, T)
        e_v_tilde = self.mlp_c_m(c_m) * e_v # (B, P, T)

        self_attn_out, _ = self.multihead_self_attention(query=c_m + e_v, key=c_m + e_v, value=c_m, need_weights=False) # (B, P, T)
        c_sa = self.layer_norm_1(self_attn_out + c_m) # (B, P, T)

        c = self.multihead_cross_attention(c_sa=c_sa, e_v_tilde=e_v_tilde, a_tilde=a_tilde, e_a_tilde=e_a_tilde) # (B, P, T)
        c = self.layer_norm_2(c + c_sa) # (B, P, T)

        ff_out = self.feed_forward(c)
        c_m = self.layer_norm_3(ff_out + c)

        d_v = self.mlp_out(c_m)
        d_v = self.linear_proj_out(d_v)
        v_m = v_m + d_v

        return c_m, v_m