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
from normal_diffusion.evaluation.evaluation import rms_angle_difference
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

    print(len(dataloader))
    first_collection = next(iter(dataloader))
    first_collection = first_collection.to(device)
    print(first_collection.x.shape)
    print(first_collection.edge_index)
    print(first_collection)

    model = PositionInvariantModel(N=config.model.model_dim, attention=config.model.attention).to(device)

    scheduler = DDIMScheduler(
        num_train_timesteps=config.scheduler.num_train_timesteps,
        beta_schedule=config.scheduler.beta_schedule,
        clip_sample=config.scheduler.clip_sample,
    )
    scheduler.num_inference_steps = config.scheduler.num_inference_steps
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs/" + now)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Setup TensorBoard
    log_dir = Path("logs/" + now)

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

    # Evaluate the model rms angle difference on the test set
    model.eval()
    with torch.inference_mode():
        for i, graph_data in enumerate(test_dataloader):
            graph_data = graph_data.to(device=device)
            batch_size = graph_data.size(0)
            noise = torch.randn_like(graph_data.x, device=device)
            clean_normals = graph_data.x.cpu().numpy()
            graph_data.x = noise
            graph_data.x /= torch.norm(graph_data.x, dim=-1, keepdim=True)
            for t in tqdm(range(scheduler.num_inference_steps), desc="Inference"):
                graph_data.x = model(
                    graph_data, torch.tensor([t] * batch_size, device=device).float()
                )
                graph_data.x = scheduler.add_noise(
                    graph_data.x,
                    graph_data.x,
                    torch.tensor([t] * batch_size, device=device),
                )
                graph_data.x /= torch.norm(graph_data.x, dim=-1, keepdim=True)
            estimated_normals = graph_data.x.cpu().numpy()
            rms = rms_angle_difference(estimated_normals, clean_normals)
            print(f"RMS angle difference on test batch {i}: {rms:.4f}")
            writer.add_text(f"RMS angle difference on test batch {i}", f"{rms:.4f}")


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
