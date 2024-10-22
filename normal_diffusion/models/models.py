# implement gcn model that takes a graph with nodes that have positional features (xyz) and noised normal features (normalized xyz) and timestamp t and predicts the denoised xyz

from torch import nn
from torch_geometric.nn import GCNConv
from normal_diffusion.models.utils import BackboneWrapper, Sequential, Activation, ConcatCondition


def GCNModel(hidden_dim: int = 64, time_embed_dim: int = 32, **gcnconv_kwargs):
    """
    example usage:
        graph_data = pg.data.Data(x=noisy_normals, pos=positions, edge_index or adj_t)
        model = GCNModel()
        predicted_normals = model(graph_data, t)
    """
    return BackboneWrapper(
        Sequential(
            ConcatCondition(
                GCNConv(3+3+time_embed_dim, hidden_dim, **gcnconv_kwargs), condition_on=["pos", "t"]
            ),
            Activation(nn.ReLU()),
            ConcatCondition(
                GCNConv(hidden_dim + 3 + time_embed_dim, hidden_dim, **gcnconv_kwargs),
                condition_on=["pos", "t"],
            ),
            Activation(nn.ReLU()),
            ConcatCondition(
                GCNConv(hidden_dim + 3 + time_embed_dim, 3, **gcnconv_kwargs), condition_on=["pos", "t"]
            ),
        ), time_embed_dim=time_embed_dim
    )
