import open3d
import open3d.visualization
from torch_geometric.datasets import PCPNetDataset
import numpy as np

# Choose the root directory where you want to save the dataset
root = './data/PCPNetDataset'

# Download and load the training dataset
dataset = PCPNetDataset(root=root, category='NoNoise', split='train')

# Print dataset information
print(f'Number of samples: {len(dataset)}')
print(f'Dataset features: {dataset[0]}')

for i in range(len(dataset)):
# Visualize the point cloud with open3d
    pcd = open3d.geometry.PointCloud()
    print(pcd)
    # From numpy to Open3D
    pcd.points = open3d.utility.Vector3dVector(dataset[i].pos.numpy().astype(np.float64))
    pcd.normals = open3d.utility.Vector3dVector(dataset[i].x[:,:3].numpy().astype(np.float64))
    print("visualizing...")
    open3d.visualization.draw_geometries([pcd], point_show_normal=True)
    print("done.")


