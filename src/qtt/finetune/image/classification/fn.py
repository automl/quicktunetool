import os
import subprocess
import time
from importlib.util import find_spec

import pandas as pd

hp_list = [
    "batch-size",
    "bss-reg",
    "cotuning-reg",
    "cutmix",
    "decay-rate",
    "decay-epochs",
    "delta-reg",
    "drop",
    "layer-decay",
    "lr",
    "mixup",
    "mixup-prob",
    "model",
    "opt",
    "patience-epochs",
    "pct-to-freeze",
    "sched",
    "smoothing",
    "sp-reg",
    "warmup-epochs",
    "warmup-lr",
    "weight-decay",
]
num_hp_list = ["clip-grad", "layer-decay"]
bool_hp_list = ["amp", "linear-probing", "stoch-norm"]
static_args = ["--pretrained", "--checkpoint-hist", "1", "--epochs", "50", "--workers", "8"]
trial_args = ["train-split", "val-split", "num-classes"]


def fn(trial: dict, trial_info: dict) -> dict:
    """
    Fine-tune a pretrained model on a image dataset.
    Using the [fimm library](<https://github.com/rapanti/fimm>).

    Args:
        trial (dict): A dictionary containing trial-specific configurations. Mandatory keys include:
            - "config-id" (str): Unique identifier for the trial configuration.
            - "config" (dict): Hyperparameter settings for the trial, as a dictionary.
            - "fidelity" (int): Specifies the fidelity level for the trial (e.g., epoch count or number of samples).
        trial_info (dict): A dictionary with additional trial metadata. Mandatory keys include:
            - "data-dir" (str): Path to the directory containing the image dataset.
            - "output-dir" (str): Path to the directory where training results and logs are saved.
            - "train-split" (str): Path to the training split folder.
            - "val-split" (str): Path to the validation split folder.
            - "num-classes" (int): Number of classes in the dataset.

    Returns:
        dict: Updated trial dictionary with:
            - "status" (bool): Indicates whether the training process was successful.
            - "score" (float): Final evaluation score (top-1 accuracy as a decimal).
            - "cost" (float): Time taken for the training process in seconds.
    """

    if not find_spec("fimm"):
        raise ImportError(
            "You need to install fimm to run this script. Run `pip install fimm` in your console."
        )

    config: dict = trial["config"]
    fidelity = str(trial["fidelity"])
    config_id = str(trial["config-id"])

    data_dir: str = trial_info["data-dir"]
    output_dir: str = trial_info["output-dir"]

    args = ["train", "--data-dir", data_dir, "--output", output_dir, "--experiment", str(config_id)]
    args += ["--fidelity", fidelity]
    args += static_args
    for arg in trial_args:
        args += [f"--{arg}", str(trial_info[arg])]

    # DATA AUGMENTATIONS
    match config.get("data-augmentation"):
        case "auto-augment":
            args += ["--aa", config["auto-augment"]]
        case "trivial-augment":
            args += ["--ta"]
        case "random-augment":
            args += ["--ra"]
            args += ["--ra-num-ops", str(config["ra-num-ops"])]
            args += ["--ra-magnitude", str(config["ra-magnitude"])]

    for k, v in config.items():
        if k in hp_list:
            args += [f"--{k}", str(v)]
        elif k in num_hp_list:
            if v > 0:
                args += [f"--{k}", str(v)]
        elif k in bool_hp_list:
            if v:
                args += [f"--{k}"]

    if os.path.exists(os.path.join(output_dir, str(config_id))):
        resume_path = os.path.join(output_dir, str(config_id), "last.pth.tar")
        args += ["--resume", resume_path]

    start = time.time()
    process = subprocess.Popen(args)
    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
    finally:
        if process.poll() is None:
            process.terminate()
    end = time.time()

    report: dict = {}
    if process.returncode == 0:
        output_path = os.path.join(output_dir, str(config_id))
        df = pd.read_csv(os.path.join(output_path, "summary.csv"))
        report["status"] = True
        report["score"] = df["eval_top1"].values[-1] / 100
        report["cost"] = end - start
    else:
        report["status"] = False
        report["score"] = 0.0
        report["cost"] = float("inf")

    trial.update(report)
    return trial
