import torch
from typing import Self

from comparison.source.imot.modules.mlp import MLP


class MultiHeadCrossAttention(torch.nn.Module):

    def __init__(self: Self, n_dims: int, n_particules: int, len_seq: int, n_heads: int, p_dropout: float) -> None:

        super(MultiHeadCrossAttention, self).__init__()

        self.n_dims = n_dims

        self.multihead_attention_acc = torch.nn.MultiheadAttention(
            embed_dim=len_seq,
            kdim=2 * len_seq,
            vdim=len_seq,
            num_heads=n_heads,
            dropout=p_dropout,
            batch_first=True
        )
        self.multihead_attention_gyro = torch.nn.MultiheadAttention(
            embed_dim=len_seq,
            kdim=2 * len_seq,
            vdim=len_seq,
            num_heads=n_heads,
            dropout=p_dropout,
            batch_first=True
        )
        self.mlp_out = MLP(dim=len_seq, n_layers=2, p_dropout=p_dropout)
        self.linear_proj_out = torch.nn.Linear(in_features=4 * n_particules, out_features=n_particules)

    def forward(self: Self, c_sa: torch.Tensor, e_v_tilde: torch.Tensor, a_tilde: torch.Tensor, e_a_tilde: torch.Tensor) -> torch.Tensor:

        a_a_tilde = a_tilde[:, 0 : 3 * self.n_dims, :] # (B, 3D, T)
        a_g_tilde = a_tilde[:, 3 * self.n_dims : 6 * self.n_dims, :] # (B, 3D, T)

        e_a_a_tilde = e_a_tilde[:, 0 : 3 * self.n_dims, :] # (B, 3D, T)
        e_a_g_tilde = e_a_tilde[:, 3 * self.n_dims : 6 * self.n_dims, :] # (B, 3D, T)

        c_a, _ = self.multihead_attention_acc( # (B, 2P, T)
            query=torch.cat([c_sa, e_v_tilde], dim=1), # (B, 2P, T)
            key=torch.cat([a_a_tilde, e_a_a_tilde], dim=2), # (B, 3D, 2T)
            value=a_a_tilde, # (B, 3D, T)
            need_weights=False
        )

        c_g, _ = self.multihead_attention_gyro( # (B, 2P, T)
            query=torch.cat([c_sa, e_v_tilde], dim=1), # (B, 2P, T)
            key=torch.cat([a_g_tilde, e_a_g_tilde], dim=2), # (B, 3D, 2T)
            value=a_g_tilde, # (B, 3D, T)
            need_weights=False
        )

        c = torch.cat([c_a, c_g], dim=1)  # (B, 4P, T)
        c = self.mlp_out(c) # (B, 4P, T)

        c = c.swapaxes(1, 2) # (B, T, 4P)
        c = self.linear_proj_out(c) # (B, T, P)
        c = c.swapaxes(1, 2) # (B, P, T)

        return c