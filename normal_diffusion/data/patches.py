from torch_geometric.data import Batch, Dataset
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import BaseTransform


class PatchDataloader(NeighborLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        K: int = 6,
        hops: int = 10,
        transform: BaseTransform | None = None,
    ):
        """
        Dataloader-like class for generating patches from a pointcloud dataset
        using `torch_geometric.loader.NeighborLoader`.
        Will generate a patch for each point in the pointcloud dataset in a random order
        so can be used with `itertools.islice` to limit the number of batches.

        Args:
            dataset (Dataset): Dataset containing pointclouds.
            batch_size (int): Number of patches per batch.
            K (int): Number of neighbors to consider for constructing the patch bfs, should be the same value as KNNGraph.
            hops (int): Number of bfs hops to take for constructing a patch, determines patch size.
            transform (BaseTransform | None): Transform to apply to the patches.
        """

        batch = Batch.from_data_list(dataset)
        super().__init__(
            data=batch,
            num_neighbors=[K] * hops,
            subgraph_type="induced",
            batch_size=batch_size,
            shuffle=True,
            transform=transform,
        )


if __name__ == "__main__":
    from torch_geometric.datasets import PCPNetDataset
    from torch_geometric.transforms import Compose, KNNGraph

    from normal_diffusion.data.transforms import DistanceToEdgeWeight, KeepNormals
    from normal_diffusion.utils.visualization import visualize_pcd

    root = "../data/PCPNetDataset"
    dataset = PCPNetDataset(
        root=root,
        category="NoNoise",
        split="train",
        transform=Compose([KeepNormals(), KNNGraph(k=6)]),
    )
    dataloader = PatchDataloader(dataset, batch_size=128, hops=10, transform=DistanceToEdgeWeight()) # can add ToSparseTensor conversion here 
    it = iter(dataloader)
    batch = next(it)
    print(batch)
    visualize_pcd(batch)
