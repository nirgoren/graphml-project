import torch
from torch_geometric.datasets import PCPNetDataset
from torch_geometric.transforms import ToSparseTensor, KNNGraph, Compose
from normal_diffusion.data.transforms import DistanceToEdgeWeight, KeepNormals
from torch_geometric.loader import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
# Choose the root directory where you want to save the dataset
root = "data/PCPNetDataset"
dataset = PCPNetDataset(
    root=root,
    category="NoNoise",
    split="train",
    transform=Compose([KeepNormals(), KNNGraph(k=6), DistanceToEdgeWeight(), ToSparseTensor()]),
)
# dataloader = PatchDataloader(dataset, batch_size=256, hops=15, transform=Compose([DistanceToEdgeWeight(), ToSparseTensor()]), limit_num_batches=1000) # can add ToSparseTensor conversion here 

dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
print(len(dataloader))
first_collection = next(iter(dataloader))
first_collection = first_collection.to(device)
print(first_collection.x.shape)
print(first_collection.adj_t)
print(first_collection)

from normal_diffusion.models import GCNModel
model = GCNModel().to(device)


import datetime
from diffusers import DDPMScheduler
from normal_diffusion.training.training import train_diffusion
from torch.utils.tensorboard import SummaryWriter
scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", clip_sample=False)
# Setup TensorBoard
log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

writer = SummaryWriter(log_dir=log_dir)

train_diffusion(model=model, dataloader=dataloader, scheduler=scheduler, n_epochs=1000, lr=1e-3, writer=writer, device=device)

