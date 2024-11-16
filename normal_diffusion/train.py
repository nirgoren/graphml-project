import argparse
import datetime
from pathlib import Path

import torch
from diffusers import DDIMScheduler
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import PCPNetDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, KNNGraph
from tqdm import tqdm

from normal_diffusion.data.transforms import KeepNormals
from normal_diffusion.eval import inference_eval
from normal_diffusion.models import PositionInvariantModel
from normal_diffusion.training.training import train_diffusion


def train_and_eval(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Choose the root directory where you want to save the dataset
    root = "data/PCPNetDataset"

    k = config.dataset.knn

    dataset = PCPNetDataset(
        root=root,
        category="NoNoise",
        split="train",
        transform=Compose([KeepNormals(), KNNGraph(k=k)]),
    )

    # dataloader = PatchDataloader(dataset, batch_size=256, hops=15, transform=Compose([DistanceToEdgeWeight(), ToSparseTensor()]), limit_num_batches=1000) # can add ToSparseTensor conversion here

    test_dataset = PCPNetDataset(
        root=root,
        category="NoNoise",
        split="test",
        transform=Compose([KeepNormals(), KNNGraph(k=k)]),
    )

    dataloader = DataLoader(
        dataset, batch_size=config.training.batch_size, shuffle=False
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.inference.batch_size, shuffle=False
    )

    model = PositionInvariantModel(N=config.model.model_dim, attention=config.model.attention).to(device)

    scheduler = DDIMScheduler(
        num_train_timesteps=config.scheduler.num_train_timesteps,
        beta_schedule=config.scheduler.beta_schedule,
        clip_sample=config.scheduler.clip_sample,
        prediction_type="sample",
    )
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs/" + now)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Setup TensorBoard
    log_dir = run_dir / "logs"

    writer = SummaryWriter(log_dir=log_dir)

    train_diffusion(
        model=model,
        train_dataloader=dataloader,
        test_dataloader=test_dataloader,
        scheduler=scheduler,
        n_epochs=config.training.n_epochs,
        lr=config.training.lr,
        writer=writer,
        device=device,
    )

    # save the model
    torch.save(model.state_dict(), run_dir / "model.pth")

    inference_eval(config, model, test_dataloader, scheduler, writer, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the model with the given config file."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the config file",
        default="configs/config.yaml",
    )
    args = parser.parse_args()

    config_path = args.config_path
    config = OmegaConf.load(config_path)

    train_and_eval(config)
