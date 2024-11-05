import tarfile
from pathlib import Path

import requests

from qtt.optimizers import QuickOptimizer

VERSION_MAP = {
    "mtlbm/micro": dict(
        url="https://ml.informatik.uni-freiburg.de/research-artifacts/quicktunetool/mtlbm/micro/archive.tar.gz",
        name="archive",
        final_name="model",
        extension="pkl",
    ),
    "mtlbm/mini": dict(
        url="https://ml.informatik.uni-freiburg.de/research-artifacts/quicktunetool/mtlbm/mini/archive.tar.gz",
        name="archive",
        final_name="model",
        extension="pkl",
    ),
    "mtlbm/extended": dict(
        url="https://ml.informatik.uni-freiburg.de/research-artifacts/quicktunetool/mtlbm/extended/archive.tar.gz",
        name="archive",
        final_name="model",
        extension="pkl",
    ),
    "mtlbm/full": dict(
        url="https://ml.informatik.uni-freiburg.de/research-artifacts/quicktunetool/mtlbm/full/archive.tar.gz",
        name="archive",
        final_name="model",
        extension="pkl",
    ),
}


# Helper functions to generate the file names
def FILENAME(version: str) -> str:
    return f"{VERSION_MAP[version].get('name')}.tar.gz"


def FILE_URL(version: str) -> str:
    return f"{VERSION_MAP[version].get('url')}"


def WEIGHTS_FILE_NAME(version: str) -> str:
    return f"{VERSION_MAP[version].get('name')}.{VERSION_MAP[version].get('extension')}"


def WEIGHTS_FINAL_NAME(version: str) -> str:
    return f"{VERSION_MAP[version].get('final_name')}.{VERSION_MAP[version].get('extension')}"


def get_pretrained_optimizer(
    version: str, download: bool = True, path: str = "~/.cache/qtt/pretrained"
) -> QuickOptimizer:
    """Get a pretrained optimizer.

    Args:
        version (str):
            Name of the pretrained optimizer version.

    Returns:
        Optimizer: A pretrained optimizer.
    """
    assert version in VERSION_MAP

    base_dir = Path(path).expanduser() / version
    model_path = base_dir / FILENAME(version)

    if download and not model_path.exists():
        base_dir.mkdir(parents=True, exist_ok=True)
        download_and_decompress(FILE_URL(version), model_path)
    elif not model_path.exists():
        raise ValueError(f"Pretrained optimizer '{version}' not found at {model_path}.")

    return QuickOptimizer.load(str(base_dir))


def download_and_decompress(url: str, path: Path) -> None:
    """Helper function to download a file from a URL and decompress it and store by given name.

    Args:
        url (str): URL of the file to download
        path (Path): Path along with filename to save the downloaded file

    Returns:
        bool: Flag to indicate if the download and decompression was successful
    """
    # Check if the file already exists
    if path.exists():
        return

    # Send a HTTP request to the URL of the file
    response = requests.get(url, allow_redirects=True)

    # Check if the request is successful
    if response.status_code != 200:
        raise ValueError(
            f"Failed to download the surrogate from {url}. "
            f"Received HTTP status code: {response.status_code}."
        )

    # Save the .tar.gz file
    with open(path, "wb") as f:
        f.write(response.content)

    # Decompress the .tar.gz file
    with tarfile.open(path, "r:gz") as tar:
        tar.extractall(path.parent.absolute())
