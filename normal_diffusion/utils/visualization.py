import numpy as np
import open3d
import open3d.visualization

def visualize_pcd(data):
    pcd = open3d.geometry.PointCloud()
    # From numpy to Open3D
    pcd.points = open3d.utility.Vector3dVector(data.pos.numpy().astype(np.float64))
    pcd.normals = open3d.utility.Vector3dVector(data.x[:,:3].numpy().astype(np.float64))
    print("visualizing...")
    open3d.visualization.draw_geometries([pcd], point_show_normal=True)