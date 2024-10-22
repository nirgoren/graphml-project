from functools import reduce
import math
import torch
from torch import nn
from torch_geometric.data import Data as GraphData

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
        module: nn.Module,
        use_sparse_adj=True,
        use_edge_weight=True,
        pass_pos_in_kwargs=True,
        time_embed_dim=32,
    ):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.module = module
        self.use_sparse_adj = use_sparse_adj
        self.use_edge_weight = use_edge_weight
        self.pass_pos_in_kwargs = pass_pos_in_kwargs

    def forward(self, graph_data: GraphData, t, **kwargs):
        x = graph_data.x

        kwargs["t"] = self.time_embed(t)

        # TODO: Maybe we also want sinusoidal embeddings for the positions (like in Nerfs)
        if self.pass_pos_in_kwargs:
            kwargs["pos"] = graph_data.pos

        if (
            self.use_sparse_adj
            and hasattr(graph_data, "adj_t")
            and graph_data.adj_t is not None
        ):
            kwargs["edge_index"] = graph_data.adj_t
        else:
            kwargs["edge_index"] = graph_data.edge_index

        if (
            self.use_edge_weight
            and hasattr(graph_data, "edge_weight")
            and graph_data.edge_weight is not None
        ):
            kwargs["edge_weight"] = graph_data.edge_weight

        return self.module(x, **kwargs)


class Parallel(nn.Module):
    def __init__(self, *modules: nn.Module, reduction=torch.add):
        super().__init__()
        self.modules_ = nn.ModuleList(modules)
        self.reduction = reduction

    def forward(self, x, **kwargs):
        results = (module(x, **kwargs) for module in self.modules_)
        return reduce(self.reduction, results)


class Sequential(nn.Module):
    def __init__(self, *modules: nn.Module):
        super().__init__()
        self.modules_ = nn.ModuleList(modules)

    def forward(self, x, **kwargs):
        for module in self.modules_:
            x = module(x, **kwargs)
        return x


class Activation(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x, **_):
        return self.module(x)


class ConcatCondition(nn.Module):
    def __init__(self, module: nn.Module, condition_on: str | list[str]):
        super().__init__()
        self.module = module
        if isinstance(condition_on, str):
            condition_on = [condition_on]
        self.condition_on = condition_on

    def forward(self, x, **kwargs):
        for condition in self.condition_on:
            c = kwargs.pop(condition)
            x = torch.cat((x, c), dim=-1)
        return self.module(x, **kwargs)
