# implement gcn model that takes a graph with nodes that have positional features (xyz) and noised normal features (normalized xyz) and timestamp t and predicts the denoised xyz

from torch import nn
from torch_geometric.nn import GCNConv
from .utils import BackboneWrapper, Sequential, Activation, ConcatCondition


def GCNModel(hidden_dim: int = 64, **gcnconv_kwargs):
    """
    example usage:
        graph_data = pg.data.Data(x=noisy_normals, pos=positions, edge_index or adj_t)
        model = GCNModel()
        predicted_normals = model(graph_data, t)
    """
    return BackboneWrapper(
        Sequential(
            ConcatCondition(
                GCNConv(7, hidden_dim, **gcnconv_kwargs), condition_on=["pos", "t"]
            ),
            Activation(nn.ReLU()),
            ConcatCondition(
                GCNConv(hidden_dim + 4, hidden_dim, **gcnconv_kwargs),
                condition_on=["pos", "t"],
            ),
            Activation(nn.ReLU()),
            ConcatCondition(
                GCNConv(hidden_dim + 4, 3, **gcnconv_kwargs), condition_on=["pos", "t"]
            ),
        )
    )
