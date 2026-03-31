import torch
from typing import Self


class AdaptiveSpatialSync(torch.nn.Module):

    def __init__(self: Self, n_dims: int, len_seq: int) -> None:

        super(AdaptiveSpatialSync, self).__init__()

        self.n_dims = n_dims
        self.len_seq = len_seq

        self.conv1x3_intra = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 1), padding="same", bias=True)
        self.conv1x1_intra = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, bias=True)
        self.conv1x1_out = torch.nn.Conv1d(in_channels=6 * n_dims, out_channels=6 * n_dims, kernel_size=1, bias=True)

        self.sigmoid = torch.nn.Sigmoid()
        self.gelu = torch.nn.GELU()

    def forward(self: Self, a_a_tilde: torch.Tensor, a_g_tilde: torch.Tensor) -> torch.Tensor:

        a_t_tilde = torch.cat([a_a_tilde, a_g_tilde], dim=2) # (B, 3D, 2T)

        a_t_tilde_1 = self.conv1x3_intra(a_t_tilde.unsqueeze(1)).squeeze(1) # (B, 3D, 2T)

        a_t_tilde_2 = a_t_tilde.mean(dim=1, keepdim=True) # (B, 1, 2T)
        a_t_tilde_2 = self.conv1x1_intra(a_t_tilde_2) # (B, 1, 2T)
        a_t_tilde_2 = self.sigmoid(a_t_tilde_2) # (B, 1, 2T)

        a_t_tilde = a_t_tilde_1 * a_t_tilde_2 # (B, 3D, 2T)

        a_s_tilde = a_t_tilde.reshape(a_t_tilde.size(dim=0), 6 * self.n_dims, self.len_seq)
        a_s_tilde = self.conv1x1_out(a_s_tilde)
        a_s_tilde = self.gelu(a_s_tilde)

        return a_s_tilde