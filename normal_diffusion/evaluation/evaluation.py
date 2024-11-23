# Function to compute RMS angle difference
import torch
from torch_geometric.datasets import PCPNetDataset
from torch_geometric.transforms import ToSparseTensor, KNNGraph, Compose
from normal_diffusion.data.transforms import DistanceToEdgeWeight, KeepNormals
from torch_geometric.loader import DataLoader
import numpy as np

def squared_angle_difference_sum(pred_normals, gt_normals):
    # Ensure vectors are normalized
    pred_normals = pred_normals / np.linalg.norm(pred_normals, axis=-1, keepdims=True)
    gt_normals = gt_normals / np.linalg.norm(gt_normals, axis=-1, keepdims=True)
    
    # Compute dot product and clamp values to avoid numerical issues
    dot_products = np.abs(np.clip(np.sum(pred_normals * gt_normals, axis=-1), -1.0, 1.0))
    
    # Calculate angle in radians
    angles = np.arccos(dot_products)
    
    # Convert angles to degrees if needed
    angles_degrees = np.degrees(angles)

    return np.sum(angles_degrees ** 2)

def rms_angle_difference(pred_normals, gt_normals):
    
    squared_sum = squared_angle_difference_sum(pred_normals, gt_normals)
    rms = np.sqrt(squared_sum / pred_normals.shape[0])
    
    return rms

if __name__ == "__main__":
    root = "data/PCPNetDataset"
    test_dataset = PCPNetDataset(
        root=root,
        category="NoNoise",
        split="test",
        transform=Compose([KeepNormals(), KNNGraph(k=6), DistanceToEdgeWeight(), ToSparseTensor()]),
    )
    first_collection = next(iter(test_dataset))
    gt_normals = first_collection.x.numpy()
    # print(gt_normals)
    # print(np.linalg.norm(gt_normals, axis=1, keepdims=True))
    # print(gt_normals.shape)
    print("RMS angle difference GT vs GT")
    print(rms_angle_difference(gt_normals, gt_normals))
    noise = np.random.normal(size=gt_normals.shape)
    pred_normals = gt_normals + noise
    print("RMS angle difference GT vs GT+noise")
    print(rms_angle_difference(gt_normals, pred_normals))