import argparse
import os
import zipfile

import gdown

PU1K_GOOGLE_DRIVE_ID = "1oTAx34YNbL6GDwHYL2qqvjmYtTVWcELg"


def main(args):
    dataset_path = os.path.abspath(args.dataset_path)
    os.makedirs(dataset_path, exist_ok=True)

    print("Downloading PU1K dataset...")
    file = gdown.download(
        id=PU1K_GOOGLE_DRIVE_ID,
        output=os.path.join(dataset_path, "PU1K.zip"),
        quiet=False,
    )

    print("Extracting PU1K dataset zip...")
    with zipfile.ZipFile(file, "r") as zip_ref:
        zip_ref.extractall(dataset_path)

    print("done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./bin",
        help="Path to save the dataset",
    )
    args = parser.parse_args()
    main(args)
