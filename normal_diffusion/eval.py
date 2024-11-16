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


def evaluate(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Choose the root directory where you want to save the dataset
    root = "data/PCPNetDataset"

    k = config.dataset.knn

    test_dataset = PCPNetDataset(
        root=root,
        category="NoNoise",
        split="test",
        transform=Compose([KeepNormals(), KNNGraph(k=k)]),
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=config.inference.batch_size, shuffle=False
    )

    model = PositionInvariantModel(
        N=config.model.model_dim, attention=config.model.attention
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
    with torch.inference_mode():
        for i, batch_data in enumerate(test_dataloader):
            batch_data = batch_data.to(device=device)
            batch_node_count = batch_data.size(0)
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
            rms = rms_angle_difference(estimated_normals, clean_normals)
            print(f"RMS angle difference on test batch {i}: {rms:.4f}")
            writer.add_text(f"RMS angle difference on test batch {i}", f"{rms:.4f}")


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
