from itertools import chain
from math import ceil
from torch_geometric.datasets import PCPNetDataset, ShapeNet
from torch_geometric.transforms import Compose, KNNGraph
from torch_geometric.loader import DataLoader
from normal_diffusion.data.transforms import KeepNormals
from normal_diffusion.data.patches import PatchDataloader

def root(typ: type):
    return "data/{name}".format(name=typ.__name__)


def get_dataloader(batch_size=1, knn=6, split="train", shuffle=True, category="NoNoise", shapenet=False):
    """
    Get a dataloader for training.
    The dataloader is an iterable created from chaining multiple dataloaders.
    Every batch is tuned to contain ~100_000*batch_size points.
    """
    pcpnet_dataset = PCPNetDataset(
        root=root(PCPNetDataset),
        category=category,
        split=split,
        transform=Compose(
            [KeepNormals(), KNNGraph(k=knn)]
        ),
    )
    if batch_size >= 1:
        pcpnet_dataloader = DataLoader(
            pcpnet_dataset, batch_size=batch_size, shuffle=shuffle
        )
    else:
        patch_size, hops = check_patch_hops(pcpnet_dataset, knn)
        print(f"using patches for pcpnet with {hops} hops and patch size ~{patch_size}")
        target_batch_size = batch_size * 100_000
        num_patches = int(target_batch_size // patch_size)
        num_batches = int(len(pcpnet_dataset) * ceil(100_000 // patch_size) * 2)
        pcpnet_dataloader = PatchDataloader(
            dataset=pcpnet_dataset,
            batch_size=num_patches,
            knn=knn,
            hops=hops,
            limit_num_batches=num_batches,
        )

    if not shapenet:
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
        shapenet_dataset, batch_size=int(38 * batch_size), shuffle=shuffle
    )

    return CombinedDataLoader(pcpnet_dataloader, shapenet_dataloader)


class CombinedDataLoader:
    def __init__(self, *dataloaders):
        self.dataloaders = dataloaders

    def __iter__(self):
        return chain(*self.dataloaders)

    def __len__(self):
        return sum([len(dl) for dl in self.dataloaders])

def check_patch_hops(dataset, knn):
    first_graph = dataset[0]
    target_size = 10_000
    patch_size = 0
    hops = 3
    while patch_size < target_size:
        hops += 1
        patch_dl = PatchDataloader(
            dataset=[first_graph],
            batch_size=1,
            knn=knn,
            hops=hops,
            limit_num_batches=1
        )
        patch = next(iter(patch_dl))
        patch_size = patch.x.size(0)
        if hops == 10:
            target_size = 5_000
        if hops == 20:
            target_size = 2_000
        if hops == 30:
            break
    return patch_size, hops

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
    dataloader = get_dataloader(batch_size=0.2, knn=50)
    for data in islice(dataloader, 10):
        print(data)