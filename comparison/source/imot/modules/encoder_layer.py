import torch
from typing import Self

from comparison.source.imot.modules.adaptive_spacial_sync import AdaptiveSpatialSync
from comparison.source.imot.modules.feed_forward import FeedForward


class EncoderLayer(torch.nn.Module):

    def __init__(self: Self, n_dims: int, len_seq: int, n_heads: int, p_dropout: float) -> None:

        super(EncoderLayer, self).__init__()

        self.multihead_attention = torch.nn.MultiheadAttention(
            embed_dim=len_seq,
            num_heads=n_heads,
            dropout=p_dropout,
            batch_first=True
        )
        self.feed_forward = FeedForward(dim=len_seq, p_dropout=p_dropout)

        self.adaptive_spatial_sync_1 = AdaptiveSpatialSync(n_dims=n_dims, len_seq=len_seq)
        self.adaptive_spatial_sync_2 = AdaptiveSpatialSync(n_dims=n_dims, len_seq=len_seq)
        
        self.layer_norm_1 = torch.nn.LayerNorm(normalized_shape=len_seq)
        self.layer_norm_2 = torch.nn.LayerNorm(normalized_shape=len_seq)

    def forward(self: Self, a_a_tilde: torch.Tensor, a_g_tilde: torch.Tensor, e_a_tilde: torch.Tensor) -> torch.Tensor:

        a_tilde = torch.cat([a_a_tilde, a_g_tilde], dim=1) # (B, 6D, T)

        attn_out, _ = self.multihead_attention(query=a_tilde + e_a_tilde, key=a_tilde + e_a_tilde, value=a_tilde, need_weights=False) # (B, 6D, T)
        residual_ass = self.adaptive_spatial_sync_1(a_a_tilde, a_g_tilde) # (B, 6D, T)
        a_tilde = self.layer_norm_1(a_tilde + attn_out + residual_ass) # (B, 6D, T)

        ff_out = self.feed_forward(a_tilde) # (B, 6D, T)
        residual_ass = self.adaptive_spatial_sync_2(a_a_tilde, a_g_tilde) # (B, 6D, T)
        a_tilde = self.layer_norm_2(a_tilde + ff_out + residual_ass) # (B, 6D, T)

        return a_tilde