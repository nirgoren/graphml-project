import math
import torch
from torch import nn
from torch_geometric.data import Data as GraphData
import inspect

class DirectionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv = nn.Conv1d(1, dim, 1)
        self.act = nn.ReLU()

    def forward(self, direction):
        B, _ = direction.shape
        conv_output = self.conv(direction.unsqueeze(1)).view(B, -1)
        return self.act(conv_output)

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class BackboneWrapper(nn.Module):
    def __init__(
        self,
        *layers: nn.Module,
        time_embed_dim=32,
    ):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.layers = nn.ModuleList(layers)
        
    @staticmethod
    def _forward_wrapper(module: nn.Module, x, edge_index, pos, time, **kwargs):
        spec = inspect.getfullargspec(module.forward)
        with_pos = "pos" in spec.args
        with_time = "time" in spec.args
        with_edge_index = "edge_index" in spec.args
        if with_pos:
            kwargs["pos"] = pos
        if with_time:
            kwargs["time"] = time
        if with_edge_index:
            kwargs["edge_index"] = edge_index
        return module(x, **kwargs)

    def forward(self, graph_data: GraphData, time: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        graph_data: torch_geometric.data.Data with the following attributes:
            - x: node features
            - pos: node positions
            - edge_index: edge index
        
        time: torch.Tensor with shape [batch_size (num nodes in batch)]
        """
        x = graph_data.x

        time = self.time_embed(time)

        pos = graph_data.pos

        if (
            hasattr(graph_data, "adj_t")
            and graph_data.adj_t is not None
        ):
            edge_index = graph_data.adj_t
        else:
            edge_index = graph_data.edge_index

        for layer in self.layers:
            x = self._forward_wrapper(layer, x, pos=pos, edge_index=edge_index, time=time, **kwargs)
        
        return x

