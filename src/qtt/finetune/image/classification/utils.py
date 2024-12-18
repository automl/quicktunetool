import os
from pathlib import Path

from torchvision.datasets import ImageFolder  # type: ignore


def extract_image_dataset_metafeat(
    path_root: str | Path, train_split: str = "train", val_split: str = "val"
):
    """
    Extracts metadata features from an image dataset for classification tasks.

    This function analyzes the specified dataset directory to compute metadata
    features, such as the number of samples, number of classes, average number
    of features (image size), and number of channels.

    Args:
        path_root (str | Path): The root directory of the dataset.
        train_split (str, optional): The subdirectory name for training data. Defaults to "train".
        val_split (str, optional): The subdirectory name for validation data. Defaults to "val".

    Returns:
        tuple: A tuple containing:
            - trial_info (dict): Information about the dataset directory and splits.
            - metafeat (dict): Metadata features including:
                - "num-samples": Total number of samples in the dataset.
                - "num-classes": Number of classes in the dataset.
                - "num-features": Average number of features (image size).
                - "num-channels": Number of channels in the images.

    Raises:
        ValueError: If the specified path does not exist or is not a directory.
    """
    # handle path
    path_root = Path(path_root)
    path_root = path_root.expanduser()  # expands ~ to home directory
    path_root = Path(path_root)  # ensure type safety
    path_root = path_root.resolve()  # convert to an absolute path
    if not path_root.exists():
        raise ValueError(f"The specified path does not exist: {path_root}")
    if not path_root.is_dir():
        raise ValueError(f"The specified path is not a directory: {path_root}")

    num_samples = 0
    num_classes = 0
    num_features = 224
    num_channels = 3

    # trainset
    train_path = os.path.join(path_root, train_split)
    if os.path.exists(train_path):
        trainset = ImageFolder(train_path)
        num_samples += len(trainset)
        num_channels = 3 if trainset[0][0].mode == "RGB" else 1
        num_classes = len(trainset.classes)

        for img, _ in trainset:
            num_features += img.size[0]
        num_features //= len(trainset)

    # valset
    val_path = os.path.join(path_root, val_split)
    if os.path.exists(val_path):
        valset = ImageFolder(val_path)
        num_samples += len(valset)

    metafeat = {
        "num-samples": num_samples,
        "num-classes": num_classes,
        "num-features": num_features,
        "num-channels": num_channels,
    }

    trial_info = {
        "data-dir": str(path_root),
        "train-split": train_split,
        "val-split": val_split,
        "num-classes": num_classes,
    }

    return trial_info, metafeat
