import torch
from torch_geometric.data import Batch, Dataset
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import BaseTransform
from itertools import islice

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class PatchDataloader(NeighborLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        knn: int = 6,
        hops: int = 10,
        transform: BaseTransform | None = None,
        limit_num_batches: int = 10_000,
        device: torch.device = DEVICE,
    ):
        """
        Dataloader-like class for generating patches from a pointcloud dataset
        using `torch_geometric.loader.NeighborLoader`.

        Args:
            dataset (Dataset): Dataset containing pointclouds.
            batch_size (int): Number of patches per batch.
            knn (int): Number of neighbors to consider for constructing the patch bfs, should be the same value as KNNGraph.
            hops (int): Number of bfs hops to take for constructing a patch, determines patch size.
            transform (BaseTransform | None): Transform to apply to the patches.
            limit_num_batches (int): Maximum number of batches to generate.
        """

        self.limit_num_batches = limit_num_batches
        batch = Batch.from_data_list(dataset).to(device)
        super().__init__(
            data=batch,
            num_neighbors=[knn] * hops,
            subgraph_type="induced",
            batch_size=batch_size,
            shuffle=True,
            transform=transform,
        )

    def __iter__(self):
        return islice(super().__iter__(), self.limit_num_batches)

    def __len__(self):
        try:
            length = super().__len__()
        except Exception as e:
            raise ValueError(f"Cannot determine length of dataloader. {e}")

        if self.limit_num_batches is None:
            return length

        return min(length, self.limit_num_batches)


if __name__ == "__main__":
    from torch_geometric.datasets import PCPNetDataset
    from torch_geometric.transforms import Compose, KNNGraph

    from normal_diffusion.data.transforms import KeepNormals
    from normal_diffusion.utils.visualization import visualize_pcd

    root = "data/PCPNetDataset"
    dataset = PCPNetDataset(
        root=root,
        category="NoNoise",
        split="train",
        transform=Compose([KeepNormals(), KNNGraph(k=6)]),
    )
    dataloader = PatchDataloader(dataset, batch_size=2, hops=10, limit_num_batches=10)
    it = iter(dataloader)
    batch = next(it)
    print(batch)
    print(batch.batch)
    # visualize_pcd(batch.cpu())
