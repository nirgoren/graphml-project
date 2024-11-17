import torch
from torch import Tensor, nn
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.typing import Adj
from torch_geometric.nn import MessagePassing


def calculate_messege_features_shape(x_dim, time_dim, direction_embedding_dim):
    return x_dim * 2 + direction_embedding_dim + time_dim


def calculate_direction_embedding_dim(direction_embedding: nn.Module):
    _, D = direction_embedding(torch.zeros(1, 3)).shape
    return D


class PositionInvariantMessagePassing(MessagePassing):
    def __init__(self, aggr: str = "sum", direction_embedding: nn.Module | None = None):
        super().__init__(aggr=aggr)
        if direction_embedding is None:
            self.direction_embedding = nn.Identity()
        else:
            self.direction_embedding = direction_embedding

    def forward(self, x: Tensor, pos: Tensor, edge_index: Adj, time: Tensor) -> Tensor:
        return self.propagate(edge_index, x=x, pos=pos, time=time)

    def message(
        self, x_i: Tensor, x_j: Tensor, pos_i: Tensor, pos_j: Tensor, time_j: Tensor
    ) -> Tensor:
        # x_i: The features of central node as shape [num_edges, in_channels]
        # x_j: The features of neighbors as shape [num_edges, in_channels]
        # pos_j: The position of neighbors as shape [num_edges, 3]
        # pos_i: The central node position as shape [num_edges, 3]
        direction_embeddings = self.direction_embedding(pos_j - pos_i)

        return torch.cat([x_i, x_j, direction_embeddings, time_j], dim=-1)


class PositionInvariantMessagePassingWithMPL(PositionInvariantMessagePassing):
    def __init__(
        self,
        layes_output_dims=(128, 32, 3),
        x_features=3,
        time_embed_dim=32,
        aggr: str = "mean",
        direction_embedding: nn.Module | None = None,
    ):
        super().__init__(aggr=aggr, direction_embedding=direction_embedding)
        layers = []
        previous_dim = calculate_messege_features_shape(
            x_features,
            time_embed_dim,
            calculate_direction_embedding_dim(direction_embedding),
        )
        for i, hidden_dim in enumerate(layes_output_dims):
            if i > 0:
                layers.append(nn.ReLU())
            layers.append(nn.Linear(previous_dim, hidden_dim))
            previous_dim = hidden_dim

        self.message_transorm = nn.Sequential(*layers)

    def message(
        self, x_i: Tensor, x_j: Tensor, pos_i: Tensor, pos_j: Tensor, time: Tensor
    ) -> Tensor:
        return self.message_transorm(super().message(x_i, x_j, pos_i, pos_j, time))


class PositionInvariantMessagePassingWithLocalAttention(
    PositionInvariantMessagePassing
):
    def __init__(
        self,
        layes_output_dims=(128, 32, 3),
        x_features=3,
        time_embed_dim=32,
        attention_dim=32,
        direction_embedding: nn.Module | None = None,
    ):
        layers = []
        input_dim = calculate_messege_features_shape(
            x_features,
            time_embed_dim,
            calculate_direction_embedding_dim(direction_embedding),
        )
        previous_dim = input_dim
        for i, hidden_dim in enumerate(layes_output_dims):
            if i > 0:
                layers.append(nn.ReLU())
            layers.append(nn.Linear(previous_dim, hidden_dim))
            previous_dim = hidden_dim

        aggr = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(input_dim, attention_dim),
                nn.ReLU(),
                nn.Linear(attention_dim, 1),
            ),
            nn=nn.Sequential(*layers),
        )

        super().__init__(aggr=aggr, direction_embedding=direction_embedding)
