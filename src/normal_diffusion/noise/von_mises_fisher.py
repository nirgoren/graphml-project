# methods to add von mises fisher noise to the data and calculate loss

import numpy as np
from scipy.stats import vonmises_fisher
from tqdm import tqdm

from normal_diffusion.utils.visualization import visualize_pcd


# TODO: perhaps optimize this?
def add_von_mises_fisher_noise(data, kappa):
    """
    Adds independent von Mises-Fisher noise to the input data.
    Parameters:
    data (numpy.ndarray): The input data array where each row is a vector to which noise will be added.
    kappa (float): The concentration parameter of the von Mises-Fisher distribution. Higher values result in samples closer to the mean direction.
    Returns:
    numpy.ndarray: The noisy data array with the same shape as the input data.
    """

    noisy_data = np.zeros_like(data)
    for i, vector in tqdm(
        enumerate(data), total=len(data), desc="Adding von Mises-Fisher noise"
    ):
        noisy_data[i] = vonmises_fisher(mu=vector, kappa=kappa).rvs(1)[0]
    return noisy_data


def von_mises_fisher_loss(data, predicted_data):
    # Loss based on cosine similarity
    return -torch.mean(torch.sum(data * predicted_data, axis=-1))


if __name__ == "__main__":
    data = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float64)
    kappa = 20
    noisy_data = add_von_mises_fisher_noise(data, kappa)
    print(noisy_data)

    import torch

    root = "./data/PCPNetDataset"
    from torch_geometric.datasets import PCPNetDataset

    dataset = PCPNetDataset(root=root, category="NoNoise", split="train")
    data = dataset[0]
    data.pos = data.pos[:10000]
    data.x = data.x[:10000]
    print(data.x.dtype)
    visualize_pcd(data)
    normals = data.x[:, :3].numpy().astype(np.float64)
    for kappa in [50, 10, 0.01]:
        noised_normals = add_von_mises_fisher_noise(normals, kappa)
        data.x[:, :3] = torch.tensor(
            noised_normals, dtype=torch.float32, device=data.x.device
        )
        visualize_pcd(data)
