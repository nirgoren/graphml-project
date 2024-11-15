import torch
from torch import Tensor, nn
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.typing import Adj
from torch_geometric.nn import MessagePassing

class PositionInvariantMessagePassing(MessagePassing):
    def __init__(self, aggr: str = 'sum'):
        super().__init__(aggr=aggr)

    def forward(self,
        x: Tensor,
        pos: Tensor,
        edge_index: Adj,
        time: Tensor
    ) -> Tensor:
        return self.propagate(edge_index, x=x, pos=pos, time=time)

    def message(self,
        x_i: Tensor,
        x_j: Tensor,
        pos_i: Tensor,
        pos_j: Tensor,
        time_j: Tensor
    ) -> Tensor:
        # x_j: The features of neighbors as shape [num_edges, in_channels]
        # pos_j: The position of neighbors as shape [num_edges, 3]
        # pos_i: The central node position as shape [num_edges, 3]

        return torch.cat([x_i, x_j, pos_j - pos_i, time_j], dim=-1)

    @staticmethod
    def calculate_messege_features_shape(x_dim, time_dim):
        return x_dim * 2 + 3 + time_dim


class PositionInvariantMessagePassingWithMPL(PositionInvariantMessagePassing):
    def __init__(self, layes_output_dims = (128, 32, 3), x_features = 3, time_embed_dim = 32, aggr: str = 'mean'):
        super().__init__(aggr=aggr)
        layers = []
        previous_dim = self.calculate_messege_features_shape(x_features, time_embed_dim)
        for i, hidden_dim in enumerate(layes_output_dims):
            if i > 0:
                layers.append(nn.ReLU())
            layers.append(nn.Linear(previous_dim, hidden_dim))
            previous_dim = hidden_dim

        self.message_transorm = nn.Sequential(*layers)
        
    def message(self,
        x_i: Tensor,
        x_j: Tensor,
        pos_i: Tensor,
        pos_j: Tensor,
        time: Tensor
    ) -> Tensor:
        return self.message_transorm(super().message(x_i, x_j, pos_i, pos_j, time))

class PositionInvariantMessagePassingWithLocalAttention(PositionInvariantMessagePassing):
    def __init__(self, layes_output_dims = (128, 32, 3), x_features = 3, time_embed_dim = 32, attention_dim = 32):
        layers = []
        input_dim = x_features * 2 + 3 + time_embed_dim
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
                nn.Linear(attention_dim, 1)
            ),
            nn=nn.Sequential(*layers)
        )

        super().__init__(aggr=aggr)