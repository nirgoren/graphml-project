from itertools import chain
from torch_geometric.datasets import PCPNetDataset, ShapeNet
from torch_geometric.transforms import Compose, KNNGraph
from torch_geometric.loader import DataLoader
from normal_diffusion.data.transforms import KeepNormals


def root(typ: type):
    return "data/{name}".format(name=typ.__name__)


def get_dataloader(batch_size=1, knn=6, split="train", shuffle=True, category="NoNoise"):
    """
    Get a dataloader for training.
    The dataloader is an iterable created from chaining multiple dataloaders.
    Every batch is tuned to contain ~100000*batch_size points.
    """
    pcpnet_dataset = PCPNetDataset(
        root=root(PCPNetDataset),
        category=category,
        split=split,
        transform=Compose(
            [KeepNormals(), KNNGraph(k=knn)]
        ),
    )
    pcpnet_dataloader = DataLoader(
        pcpnet_dataset, batch_size=batch_size, shuffle=shuffle
    )

    return pcpnet_dataloader

    # if download fails, download manually from https://www.kaggle.com/datasets/mitkir/shapenet/download?datasetVersionNumber=1
    # and unzip into data/ShapeNet/raw
    shapenet_dataset = ShapeNet(
        root=root(ShapeNet),
        split=split,
        include_normals=True,
        transform=KNNGraph(k=knn),
    )
    shapenet_dataloader = DataLoader(
        shapenet_dataset, batch_size=38 * batch_size, shuffle=shuffle
    )

    return CombinedDataLoader(pcpnet_dataloader, shapenet_dataloader)


class CombinedDataLoader:
    def __init__(self, *dataloaders):
        self.dataloaders = dataloaders

    def __iter__(self):
        return chain(*self.dataloaders)

    def __len__(self):
        return sum([len(dl) for dl in self.dataloaders])


if __name__ == "__main__":
    from itertools import islice

    dataloader = get_dataloader(batch_size=2)
    print(len(dataloader))
    for data in islice(dataloader, 10):
        print(data)
    dataloader = get_dataloader(batch_size=2, split="test")
    print(len(dataloader))
    for data in islice(dataloader, 20):
        print(data)
