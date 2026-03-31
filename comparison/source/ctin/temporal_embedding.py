import torch
from typing import Self


class TemporalEmbedding(torch.nn.Module):

    def __init__(
        self: Self,
        input_dim: int,
        d_model: int,
        max_len: int=10000
    ) -> None:

        super(TemporalEmbedding, self).__init__()

        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=d_model//2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.pe = torch.nn.Parameter(torch.rand(1, max_len, d_model))

    def forward(
        self: Self,
        x: torch.Tensor
    ) -> torch.Tensor:

        _, len_seq, _ = x.size()

        x, _ = self.lstm(x)
        x = x + self.pe[:, :len_seq, :]

        return x