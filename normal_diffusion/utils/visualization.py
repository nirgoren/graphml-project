from matplotlib import pyplot as plt
import numpy as np
import open3d
import open3d.visualization

def visualize_preds(data, preds):
    pred_normals = preds[:,:3].numpy()
    gt_normals = data.x[:,:3].numpy()
    scores = 1-np.abs(np.clip(np.sum(pred_normals * gt_normals, axis=-1), -1.0, 1.0))
    colormap = plt.cm.winter
    colors = colormap(scores)[:,:3]
    pcd = open3d.geometry.PointCloud()
    # From numpy to Open3D
    pcd.points = open3d.utility.Vector3dVector(data.pos.numpy().astype(np.float64))
    # pcd.normals = open3d.utility.Vector3dVector(data.x[:,:3].numpy().astype(np.float64))
    pcd.colors = open3d.utility.Vector3dVector(colors.astype(np.float64))
    print("visualizing...")
    open3d.visualization.draw_geometries([pcd])

def visualize_pcd(data):
    pcd = open3d.geometry.PointCloud()
    # From numpy to Open3D
    pcd.points = open3d.utility.Vector3dVector(data.pos.numpy().astype(np.float64))
    pcd.normals = open3d.utility.Vector3dVector(data.x[:,:3].numpy().astype(np.float64))
    pcd.colors = open3d.utility.Vector3dVector(data.x[:,:3].numpy().astype(np.float64))
    print("visualizing...")
    open3d.visualization.draw_geometries([pcd])