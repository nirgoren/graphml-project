import torch
import torch.nn as nn
import torch.optim as optim

from normal_diffusion.noise.gaussian_noise import gaussian_noise_loss

def train_diffusion(model, dataloader, scheduler, n_epochs=100, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        total_loss = 0
        for graph_data in dataloader:
            optimizer.zero_grad()
            batch_size = graph_data.size(0)
            noise = torch.randn_like(graph_data.x)
            clean_normals = graph_data.x
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,))
            graph_data.x = scheduler.add_noise(graph_data.x, noise, timesteps)
            graph_data.x /= torch.norm(graph_data.x, dim=-1, keepdim=True)
            
            estimated_normals = model(graph_data, timesteps.float())
            normalized_estimated_normals = estimated_normals / torch.norm(estimated_normals, dim=-1, keepdim=True)
            loss = gaussian_noise_loss(normalized_estimated_normals, clean_normals)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss / len(dataloader):.4f}")