import torch
from typing import Self


class SpatialEmbedding(torch.nn.Module):

    def __init__(
        self: Self,
        input_dim: int,
        d_model: int
    ) -> None:

        super(SpatialEmbedding, self).__init__()

        self.conv = torch.nn.Conv1d(
            in_channels=input_dim,
            out_channels=d_model,
            kernel_size=3,
            padding="same"
        )
        self.batch_norm = torch.nn.BatchNorm1d(num_features=d_model)
        self.linears = torch.nn.Sequential(
            (torch.nn.Linear(in_features=d_model, out_features=d_model)),
            (torch.nn.Linear(in_features=d_model, out_features=d_model))
        )

    def forward(
        self: Self,
        x: torch.Tensor
    ) -> torch.Tensor:

        x = x.swapaxes(1, 2)
        x = self.conv(x)
        x = self.batch_norm(x)
        x = x.swapaxes(1, 2)
        x = self.linears(x)

        return x