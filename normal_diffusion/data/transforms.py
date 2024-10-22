import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_scatter.composite import scatter_softmax


class DistanceToEdgeWeight(BaseTransform):
    r"""Creates edge weights based on Euclidean distance of linked nodes.
    Distances are normalized per target using softmax.

    Args:
        temperature (float): The temperature of the softmax.
    """

    def __init__(
        self,
        temperature: float = 1.0,
    ):
        self.temperature = temperature

    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        assert data.edge_index is not None
        (src, dst), pos = data.edge_index, data.pos
        dist = torch.norm(pos[dst] - pos[src], p=2, dim=-1)
        weight = scatter_softmax(dist / self.temperature, dst, dim=0)
        data.edge_weight = weight
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(temperature={self.temperature})"

class KeepNormals(BaseTransform):
    r"""Keeps the normal vectors in the data object.
    """

    def forward(self, data: Data) -> Data:
        assert data.x is not None
        data.x = data.x[:, :3].clone()
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"