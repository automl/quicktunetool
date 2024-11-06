import os
import time

import pandas as pd
import yaml

from fimm.train import main

hp_list = [
    "batch_size",
    "bss_reg",
    "clip_grad",
    "cotuning_reg",
    "cutmix",
    "delta_reg",
    "drop",
    "lr",
    "mixup",
    "mixup_prob",
    "model",
    "opt",
    "pct_to_freeze",
    "sched",
    "smoothing",
    "sp_reg",
    "warmup_epochs",
    "warmup_lr",
    "weight_decay",
]

num_hp_list = [
    "clip_grad",
    "layer_decay",
]

bool_hp_list = [
    "amp",
    "linear_probing",
    "stoch_norm",
]

cond_hp_list = ["decay_rate", "decay_epochs", "patience_epochs"]

static_args = [
    "--pretrained",
    "--checkpoint_hist",
    "1",
    "--epochs",
    "50",
    "--workers",
    "8",
]

trial_args = [
    "train-split",
    "val-split",
    "num-classes",
]


def fn(trial: dict, trial_info: dict):
    x = main()

    return x
