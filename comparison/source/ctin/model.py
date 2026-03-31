import torch
from typing import Self

from comparison.source.ctin.spatial_encoder_layer import SpatialEncoderLayer
from comparison.source.ctin.spatial_embedding import SpatialEmbedding
from comparison.source.ctin.temporal_embedding import TemporalEmbedding


class CTIN(torch.nn.Module):

    def __init__(
        self: Self,
        input_dim: int,
        out_dim: int,
        d_model: int,
        n_head: int=8,
        n_encoder_layers: int=1,
        n_decoder_layers: int=4,
        p_dropout_encoder: float=0.5,
        p_dropout_decoder: float=0.05
    ) -> None:

        super(CTIN, self).__init__()

        self.d_model = d_model

        self.spatial_embedding = SpatialEmbedding(input_dim=input_dim, d_model=d_model)
        
        self.temporal_embedding = TemporalEmbedding(input_dim=input_dim, d_model=d_model)

        self.spatial_encoder = torch.nn.Sequential(*[
            SpatialEncoderLayer(d_model=d_model, p_dropout=p_dropout_encoder)
            for _ in range(n_encoder_layers)
        ])

        self.temporal_decoder = torch.nn.TransformerDecoder(
            decoder_layer=torch.nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=4 * d_model,
                dropout=p_dropout_decoder,
                batch_first=True
            ),
            num_layers=n_decoder_layers,
            norm=None
        )

        self.mlp_velocity = torch.nn.Sequential(
            (torch.nn.Linear(in_features=d_model, out_features=out_dim)),
            (torch.nn.LayerNorm(normalized_shape=out_dim))
        )
        self.mlp_covariance = torch.nn.Sequential(
            (torch.nn.Linear(in_features=d_model, out_features=out_dim)),
            (torch.nn.LayerNorm(normalized_shape=out_dim))
        )

    def forward(
        self: Self,
        x: torch.Tensor
    ) -> None:

        _, len_seq, _ = x.size()

        x_encoder = x
        x_encoder = self.spatial_embedding(x_encoder)
        x_encoder = x_encoder.swapaxes(1, 2)
        x_encoder = self.spatial_encoder(x_encoder)
        x_encoder = x_encoder.swapaxes(1, 2)

        x = self.temporal_embedding(x)
        h = self.temporal_decoder(tgt=x, memory=x_encoder, tgt_mask=torch.tril(torch.ones(len_seq, len_seq)).cuda(), memory_mask=None)

        vel = self.mlp_velocity(h)
        cov = self.mlp_covariance(h)

        return vel, cov