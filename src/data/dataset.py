import logging
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

DATASET_DEFAULT_PATH = "./bin/PU1K"


def load_h5_data(h5_filename: str) -> tuple[np.ndarray, np.ndarray]:
    logger.info("loading data from h5 file: {}".format(h5_filename))
    with h5py.File(h5_filename, "r") as f:
        input = f["poisson_256"][:]
        gt = f["poisson_1024"][:]

    assert len(input) == len(gt)

    logger.info("total {} samples".format(len(input)))

    return input, gt


def normalize_point_cloud(
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(data.shape) == 2:
        axis = 0
        logger.info("normalizing a single point cloud")
    elif len(data.shape) == 3:
        axis = 1
        logger.info("normalizing a batch of point clouds")
    else:
        raise ValueError(
            "data shape should be either (N, 3) or (B, N, 3), got {}".format(data.shape)
        )

    centroid = np.mean(data, axis=axis, keepdims=True)
    data = data - centroid
    furthest_distance = np.amax(
        np.sqrt(np.sum(data**2, axis=-1, keepdims=True)), axis=axis, keepdims=True
    )
    data = data / furthest_distance
    return data, centroid, furthest_distance


def load_xyz_dirs_data(*dir_name: str) -> np.ndarray | tuple[np.ndarray]:
    logger.info("loading data from xyz files in directories: {}".format(dir_name))
    # check all directories exists
    for d in dir_name:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"directory {d} not found")

    # check all directories has the same files
    files_in_dirs = []
    for d in dir_name:
        files = os.listdir(d)
        files = set(f for f in files if f.endswith(".xyz"))
        files_in_dirs.append(files)
    assert all(
        f == files_in_dirs[0] for f in files_in_dirs
    ), "directories have different files"

    # load data
    file_names = sorted(files_in_dirs[0])
    data = []
    for d in dir_name:
        data_in_dir = []
        for f in file_names:
            data_in_file = []
            with open(os.path.join(d, f), "r") as file:
                for line in file:
                    coor = list(map(float, line.split()))
                    assert len(coor) == 3, "each line should have 3 coordinates"
                    data_in_file.append(coor)
            data_in_dir.append(np.array(data_in_file, dtype=np.float32))
        data.append(data_in_dir)

    if len(data) == 1:
        data = data[0]

    return tuple(data)


class PU1KTrainDataset(Dataset):
    def __init__(self, dataset_path=DATASET_DEFAULT_PATH, normalize=True):
        train_data_path = os.path.join(dataset_path, "train")

        h5_files = (
            f for f in os.listdir(train_data_path) if os.path.splitext(f)[1] == ".h5"
        )

        try:
            h5_file = next(iter(h5_files))
        except StopIteration:
            raise FileNotFoundError("No h5 file found in {}".format(train_data_path))

        self.data, self.ground_truth = load_h5_data(
            os.path.join(train_data_path, h5_file)
        )
        if normalize:
            self.data, self.centroid, self.furthest_distance = normalize_point_cloud(
                self.data
            )
            self.ground_truth = (
                self.ground_truth - self.centroid
            ) / self.furthest_distance

        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.ground_truth = torch.tensor(self.ground_truth, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.ground_truth[idx]


class PU1KTestDataset(Dataset):
    def __init__(
        self, dataset_path=DATASET_DEFAULT_PATH, test_name="input_256", normalize=True
    ):
        test_dir = os.path.join(dataset_path, "test", test_name)
        assert os.path.isdir(test_dir)

        dirs = os.listdir(test_dir)
        input_dir = [d for d in dirs if d.startswith("input")][0]
        input_dir = os.path.join(test_dir, input_dir)
        gt_dir = [d for d in dirs if d.startswith("gt")][0]
        gt_dir = os.path.join(test_dir, gt_dir)

        self.input, self.gt = load_xyz_dirs_data(input_dir, gt_dir)
        if normalize:
            self.input, self.centroid, self.furthest_distance = normalize_point_cloud(
                self.input
            )
            self.gt = (self.gt - self.centroid) / self.furthest_distance

        self.input = torch.tensor(self.input, dtype=torch.float32)
        self.gt = torch.tensor(self.gt, dtype=torch.float32)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx], self.gt[idx]
