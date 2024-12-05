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


def fn(trial: dict, trial_info: dict):
    if not find_spec("fimm"):
        raise ImportError(
            "You need to install fimm to run this script. Run `pip install fimm` in your terminal."
        )

    config = trial["config"]
    fidelity = trial["fidelity"]
    config_id = trial["config-id"]

    data_dir = trial_info["data-dir"]
    output_dir = trial_info["output-dir"]

    args = ["train", "--data-dir", data_dir, "--output", output_dir, "--experiment", str(config_id)]
    args += static_args
    for arg in trial_args:
        args += [f"--{arg}", str(trial_info[arg])]

    # DATA AUGMENTATIONS
    match config.get("data_augmentation"):
        case "auto_augment":
            args += ["--aa", config["auto_augment"]]
        case "trivial_augment":
            args += ["--ta"]
        case "random_augment":
            args += ["--ra"]
            args += ["--ra-num-ops", str(config["ra_num_ops"])]
            args += ["--ra-magnitude", str(config["ra_magnitude"])]

    for k, v in config.items():
        k = k.replace("_", "-")
        if k in hp_list:
            args.append(f"--{k}")
            args.append(str(v))
        elif k in num_hp_list:
            if v > 0:
                args += [f"--{k}", str(v)]
        elif k in bool_hp_list:
            if bool(v):
                args.append(f"--{k}")

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

    if process.returncode == 0:
        status = True
        output_path = os.path.join(output_dir, str(config_id))
        df = pd.read_csv(os.path.join(output_path, "summary.csv"))
        score = df["eval_top1"].values[-1] / 100
        cost = end - start
    else:
        status = False
        score = -1
        cost = -1

    return {
        "config_id": config_id,
        "fidelity": fidelity,
        "status": status,
        "score": score,
        "cost": cost,
    }
