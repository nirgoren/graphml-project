import torch
import torch.nn as nn
import torch.optim as optim

from normal_diffusion.noise.gaussian_noise import gaussian_noise_loss

def evaluate_diffusion(model, test_dataloader, scheduler, epoch, n_epochs, writer=None, device='cpu'):
    with torch.inference_mode():
        total_loss = 0
        for i, graph_data in enumerate(test_dataloader):
            graph_data = graph_data.to(device=device)
            batch_size = graph_data.size(0)
            noise = torch.randn_like(graph_data.x)
            clean_normals = graph_data.x
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,)).to(device=device)
            graph_data.x = scheduler.add_noise(graph_data.x, noise, timesteps)
            graph_data.x /= torch.norm(graph_data.x, dim=-1, keepdim=True)

            estimated_normals = model(graph_data, timesteps.float())
            normalized_estimated_normals = estimated_normals / torch.norm(estimated_normals, dim=-1, keepdim=True)
            loss = gaussian_noise_loss(normalized_estimated_normals, clean_normals)
            total_loss += loss.item()
        if writer is not None:
            writer.add_scalar("test_loss", total_loss / (i+1), epoch)
        print(f"Epoch {epoch+1}/{n_epochs} Test Loss: {total_loss / (i+1):.4f}")

def train_diffusion(model, train_dataloader, test_dataloader, scheduler, n_epochs=100, lr=1e-3, writer=None, device='cpu'):
    if writer is not None:
        writer.add_text("model", str(model))
        writer.add_text("scheduler", str(scheduler))
        writer.add_text("n_epochs", str(n_epochs))
        writer.add_text("lr", str(lr))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        total_loss = 0
        for i, graph_data in enumerate(train_dataloader):
            graph_data = graph_data.to(device=device)
            optimizer.zero_grad()
            batch_size = graph_data.size(0)
            noise = torch.randn_like(graph_data.x)
            clean_normals = graph_data.x
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,)).to(device=device)
            graph_data.x = scheduler.add_noise(graph_data.x, noise, timesteps)
            graph_data.x /= torch.norm(graph_data.x, dim=-1, keepdim=True)
            
            estimated_normals = model(graph_data, timesteps.float())
            normalized_estimated_normals = estimated_normals / torch.norm(estimated_normals, dim=-1, keepdim=True)
            loss = gaussian_noise_loss(normalized_estimated_normals, clean_normals)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if writer is not None and i % 10 == 0:
                writer.add_scalar("loss", loss.item(), epoch * len(train_dataloader) + i)
            
        
        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {total_loss / (i+1):.4f}")
        if (epoch + 1) % 10 == 0:
            evaluate_diffusion(model, test_dataloader, scheduler, epoch, n_epochs, writer, device)