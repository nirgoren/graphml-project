import argparse
import datetime
from pathlib import Path

import torch
from diffusers import DDIMScheduler
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from normal_diffusion.data.dataloader import get_dataloader
from normal_diffusion.eval import inference_eval
from normal_diffusion.models import PositionInvariantModel
from normal_diffusion.training.training import train_diffusion


def train_and_eval(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Choose the root directory where you want to save the dataset

    k = config.dataset.knn

    dataloader = get_dataloader(
        batch_size=config.training.batch_size, knn=k, split="train", category="NoNoise"
    )
    test_dataloader = get_dataloader(
        batch_size=config.inference.batch_size,
        knn=k,
        split="test",
        shuffle=False,
        category=config.dataset.category,
    )
    dataloader = get_dataloader(
        batch_size=config.training.batch_size,
        knn=k,
        split="train",
        shapenet=config.dataset.shapenet,
        category="NoNoise",
    )
    test_dataloader = get_dataloader(
        batch_size=config.inference.batch_size,
        knn=k,
        split="test",
        shuffle=False,
        shapenet=config.dataset.shapenet,
        category=config.dataset.category,
    )

    model = PositionInvariantModel(
        N=config.model.model_dim,
        attention=config.model.attention,
        aggregation=config.model.aggregation,
    ).to(device)

    scheduler = DDIMScheduler(
        num_train_timesteps=config.scheduler.num_train_timesteps,
        beta_schedule=config.scheduler.beta_schedule,
        clip_sample=config.scheduler.clip_sample,
        prediction_type="sample",
    )
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs/" + now)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, run_dir / "config.yaml")

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
        min_training_timestep=config.training.min_training_timestep,
        flip_normals=config.training.flip_normals,
        save_path=run_dir / "model.pth",
    )

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
