from functools import partial

from torch import nn

from normal_diffusion.models.gnn_layers import (
    PositionInvariantMessagePassingWithLocalAttention,
    PositionInvariantMessagePassingWithMPL,
)
from normal_diffusion.models.utils import BackboneWrapper, DirectionalEmbedding


def PositionInvariantModel(
    time_embed_dim: int = 32,
    direction_embed_dim=12,
    N: int = 64,
    attention: bool = True,
    attention_dim: int = 32,
    aggregation: str = "mean",
):
    """
    example usage:
        graph_data = pg.data.Data(x=noisy_normals, pos=positions, edge_index or adj_t)
        model = PositionInvariantModel()
        predicted_normals = model(graph_data, time)
    """
    direction_embed = DirectionalEmbedding(direction_embed_dim)

    if attention:
        gnn_layer = partial(
            PositionInvariantMessagePassingWithLocalAttention,
            attention_dim=attention_dim,
        )
    else:
        gnn_layer = partial(
            PositionInvariantMessagePassingWithMPL, aggr=aggregation
        )

    return BackboneWrapper(
        gnn_layer(
            layes_output_dims=(N // 2, N),
            x_features=3,
            time_embed_dim=time_embed_dim,
            direction_embedding=direction_embed,
        ),
        nn.ReLU(),
        gnn_layer(
            layes_output_dims=(N, N // 2),
            x_features=N,
            time_embed_dim=time_embed_dim,
            direction_embedding=direction_embed,
        ),
        nn.ReLU(),
        gnn_layer(
            layes_output_dims=(N // 2, 3),
            x_features=N // 2,
            time_embed_dim=time_embed_dim,
            direction_embedding=direction_embed,
        ),
        time_embed_dim=time_embed_dim,
    )
