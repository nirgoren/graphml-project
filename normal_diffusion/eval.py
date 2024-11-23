import argparse
import datetime
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import PCPNetDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, KNNGraph
from tqdm import tqdm

from normal_diffusion.data.transforms import KeepNormals
from normal_diffusion.evaluation.evaluation import (
    count_angle_difference_less_than,
    rms_angle_difference,
    squared_angle_difference_sum,
)
from normal_diffusion.models import PositionInvariantModel


def evaluate(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Choose the root directory where you want to save the dataset
    root = "data/PCPNetDataset"

    k = config.dataset.knn

    test_dataset = PCPNetDataset(
        root=root,
        category=config.dataset.category,
        split="test",
        transform=Compose([KeepNormals(), KNNGraph(k=k)]),
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=config.inference.batch_size, shuffle=False
    )

    model = PositionInvariantModel(
        N=config.model.model_dim,
        attention=config.model.attention,
        aggregation=config.model.aggregation,
    ).to(device)
    model.load_state_dict(torch.load(config.model.model_path))

    scheduler = DDIMScheduler(
        num_train_timesteps=config.scheduler.num_train_timesteps,
        beta_schedule=config.scheduler.beta_schedule,
        clip_sample=config.scheduler.clip_sample,
        prediction_type="sample",
    )
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("eval/" + now)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Setup TensorBoard
    log_dir = run_dir / "logs"
    writer = SummaryWriter(log_dir=log_dir)
    inference_eval(config, model, test_dataloader, scheduler, writer, device)


def inference_eval(config, model, test_dataloader, scheduler, writer, device):
    # Evaluate the model rms angle difference on the test set
    scheduler.set_timesteps(config.scheduler.num_inference_steps)
    model.eval()
    alpha = 10
    nodes_total = 0
    squared_angle_diff_sum = 0
    less_than_alpha = 0
    with torch.inference_mode():
        for i, batch_data in tqdm(enumerate(test_dataloader), desc="Batch"):
            batch_data = batch_data.to(device=device)
            batch_node_count = batch_data.size(0)
            nodes_total += batch_node_count
            noise = torch.randn_like(batch_data.x, device=device)
            clean_normals = batch_data.x.cpu().numpy()
            batch_data.x = noise
            batch_data.x /= torch.norm(batch_data.x, dim=-1, keepdim=True)
            for t in tqdm(scheduler.timesteps, desc="Inference"):
                predicted = model(
                    batch_data,
                    torch.tensor([t] * batch_node_count, device=device).float(),
                )
                predicted /= torch.norm(predicted, dim=-1, keepdim=True)
                batch_data.x = scheduler.step(
                    predicted,
                    t,
                    batch_data.x,
                ).prev_sample
                batch_data.x /= torch.norm(batch_data.x, dim=-1, keepdim=True)
            estimated_normals = batch_data.x.cpu().numpy()
            squared_angle_diff_sum += squared_angle_difference_sum(
                estimated_normals, clean_normals
            )
            less_than_alpha += count_angle_difference_less_than(
                estimated_normals, clean_normals, alpha
            )
    rms_angle_diff = np.sqrt(squared_angle_diff_sum / nodes_total)
    less_than_alpha = less_than_alpha / nodes_total
    print(
        f"Percentage of angle differences less than {alpha} degrees: {less_than_alpha}"
    )
    writer.add_scalar("RMS angle difference", rms_angle_diff)
    print(f"RMS angle difference: {rms_angle_diff}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the model with the given config file."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the config file",
        default="configs/evaluation/eval_config.yaml",
    )
    args = parser.parse_args()

    config_path = args.config_path
    config = OmegaConf.load(config_path)

    evaluate(config)
