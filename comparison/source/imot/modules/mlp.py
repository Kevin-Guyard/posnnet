import torch
from typing import Self


class MLP(torch.nn.Module):

    def __init__(self: Self, dim: int, n_layers: int, p_dropout: float) -> None:

        super(MLP, self).__init__()

        layers = []

        for i_layer in range(n_layers):

            layers.append(torch.nn.Linear(in_features=dim, out_features=dim))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(p=p_dropout))

        self.net = torch.nn.Sequential(*layers)

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:

        x = self.net(x)

        return x