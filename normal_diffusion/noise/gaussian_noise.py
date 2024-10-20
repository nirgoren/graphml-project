import open3d
import open3d.visualization
import torch


def add_gaussian_noise(mu, sigma):
    noise = torch.randn_like(mu) * sigma
    mu += noise
    # renormalize
    mu /= torch.norm(mu, dim=-1, keepdim=True)
    return mu


def gaussian_noise_loss(data, predicted_data):
    # Loss based on cosine similarity
    return -torch.mean(torch.sum(data * predicted_data, axis=-1))


if __name__ == "__main__":
    from normal_diffusion.utils.visualization import visualize_pcd

    mu = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
    sigma = 0.1
    noisy_mu = add_gaussian_noise(mu, sigma)
    print(noisy_mu)
    print(noisy_mu.norm(dim=-1))

    root = "./data/PCPNetDataset"
    from torch_geometric.datasets import PCPNetDataset

    dataset = PCPNetDataset(root=root, category="NoNoise", split="train")
    data = dataset[0]
    data.pos = data.pos[:10000]
    data.x = data.x[:10000]
    normals = data.x[:, :3]
    visualize_pcd(data)
    for sigma in [0.1, 1, 10]:
        data.x[:, :3] = add_gaussian_noise(normals, sigma)
        visualize_pcd(data)
