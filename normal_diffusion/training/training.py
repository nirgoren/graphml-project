from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from diffusers import DDPMScheduler
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from normal_diffusion.noise.gaussian_noise import gaussian_noise_loss


def evaluate_diffusion(
    model: nn.Module,
    test_dataloader: DataLoader,
    scheduler: DDPMScheduler,
    epoch: int,
    n_epochs: int,
    writer: SummaryWriter | None = None,
    device: torch.device | str = "cpu",
    min_training_timestep: int = 0,
    flip_normals: bool = False,
):
    with torch.inference_mode():
        total_loss = 0
        for i, batch_data in enumerate(test_dataloader):
            batch_data = batch_data.to(device=device)
            batch_size = len(batch_data)
            noise = torch.randn_like(batch_data.x)
            clean_normals = batch_data.x
            timesteps = torch.randint(
                min_training_timestep, scheduler.config.num_train_timesteps, (batch_size,)
            ).to(device=device)
            timesteps = timesteps[batch_data.batch]
            batch_data.x = scheduler.add_noise(batch_data.x, noise, timesteps)
            if flip_normals:
                # randomly flip normals
                random_signs = torch.randint(0, 2, (batch_data.x.shape[0],), device=device) * 2 - 1  # Random 0 or 1 mapped to -1 or 1
                batch_data.x = batch_data.x * random_signs.view(-1, 1)
            batch_data.x /= torch.norm(batch_data.x, dim=-1, keepdim=True)

            estimated_normals = model(batch_data, timesteps.float())
            normalized_estimated_normals = estimated_normals / torch.norm(
                estimated_normals, dim=-1, keepdim=True
            )
            loss = gaussian_noise_loss(normalized_estimated_normals, clean_normals)
            total_loss += loss.item()
        if writer is not None:
            writer.add_scalar("test_loss", total_loss / (i + 1), epoch)
        print(f"Epoch {epoch+1}/{n_epochs} Test Loss: {total_loss / (i+1):.4f}")


def train_diffusion(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    scheduler: DDPMScheduler,
    n_epochs: int = 100,
    lr: float = 1e-3,
    writer: SummaryWriter | None = None,
    device: torch.device | str = "cpu",
    min_training_timestep: int = 0,
    flip_normals: bool = False,
    save_path: Path | str | None = None,
):
    if writer is not None:
        writer.add_text("model", str(model))
        writer.add_text("scheduler", str(scheduler))
        writer.add_text("n_epochs", str(n_epochs))
        writer.add_text("lr", str(lr))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        total_loss = 0
        for i, batch_data in enumerate(train_dataloader):
            batch_data = batch_data.to(device=device)
            optimizer.zero_grad()
            batch_size = len(batch_data)
            noise = torch.randn_like(batch_data.x)
            clean_normals = batch_data.x
            timesteps = torch.randint(
                min_training_timestep, scheduler.config.num_train_timesteps, (batch_size,)
            ).to(device=device)
            if batch_data.batch is None:
                # If the batch data does not have a batch attribute, we use a single timestep for all points
                batch_data.batch = torch.zeros(batch_data.x.size(0), dtype=torch.long)
            timesteps = timesteps[batch_data.batch]
            batch_data.x = scheduler.add_noise(batch_data.x, noise, timesteps)
            if flip_normals:
                # randomly flip normals
                random_signs = torch.randint(0, 2, (batch_data.x.shape[0],), device=device) * 2 - 1  # Random 0 or 1 mapped to -1 or 1
                batch_data.x = batch_data.x * random_signs.view(-1, 1)
            batch_data.x /= torch.norm(batch_data.x, dim=-1, keepdim=True)

            estimated_normals = model(batch_data, timesteps.float())
            normalized_estimated_normals = estimated_normals / torch.norm(
                estimated_normals, dim=-1, keepdim=True
            )
            loss = gaussian_noise_loss(normalized_estimated_normals, clean_normals)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if writer is not None and i % 10 == 0:
                writer.add_scalar(
                    "loss", loss.item(), epoch * len(train_dataloader) + i
                )
        
        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {total_loss / (i+1):.4f}")
        if (epoch + 1) % 10 == 0:
            evaluate_diffusion(
                model, test_dataloader, scheduler, epoch, n_epochs, writer, device, min_training_timestep, flip_normals
            )
            torch.save(model.state_dict(), save_path)

