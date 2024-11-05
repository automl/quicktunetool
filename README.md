# Quick-Tune-Tool: A Framework for Efficient Model Selection and Hyperparameter Optimization

[![image](https://img.shields.io/pypi/l/quicktunetool.svg)](https://pypi.python.org/pypi/quicktunetool)
[![image](https://img.shields.io/pypi/pyversions/quikctunetool.svg)](https://pypi.python.org/pypi/quicktunetool)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

Quick-Tune-Tool is a streamlined framework designed to tackle model selection and hyperparameter tuning for pretrained models on new datasets. Built on a Combined Algorithm Selection and Hyperparameter Optimization (CASH) approach within a Bayesian optimization framework, it aims to identify the best-performing model and hyperparameter configuration quickly and efficiently.

### Table of Contents
- [Key Features](#key-features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [References](#references)
- [Citations](#citations)

## Key Features
Quick-Tune-Tool leverages three main techniques to optimize model performance effectively:
1. **Gray-Box Hyperparameter Optimization (HPO)**: Instead of fully training every model, QuickTuneTool only trains each for a few initial epochs and focuses on the most promising models, significantly reducing computation time.
2. **Meta-Learning**: By using prior evaluations on related tasks, the tool accelerates the search process and refines model recommendations.
3. **Cost-Awareness**: QuickTuneTool balances time and performance, maintaining an efficient search for the best configurations.

***Note: Currently, only Image Classification is supported.***

## Getting Started

### Installation

To install QuickTuneTool (`QTT`), you can simply use `pip`:

```bash
pip install quicktunetool
```

#### Install from Source
To install QuickTuneTool directly from the source:

```bash
git clone https://github.com/automl/quicktunetool
pip install -e quicktunetool    # Use -e for editable mode
```

## Usage
QuickTuneTool's interface makes it easy to get started. Here's a simple script to run QuickTuneTool on your dataset:

```python
from ConfigSpace import ConfigurationSpace
from qtt import QuickTuner, QuickOptimizer

cs = ConfigurationSpace({
    "cat": [False, True],  # Categorical
    "float": (0.1, 1.0),   # Uniform Float
    "int": (1, 10),        # Uniform Int
    "constant": Constant("constant": (42)),
})

def fn(trial: dict, task_info: dict):
    config = trial["config"]
    fidelity = trial["fidelity"]
    ...
    # Training logic and checkpoint loading here
    score = ...  # float: 0 - 1
    cost = ...

    report = trial.copy()
    report["score"] = score
    report["cost"] = cost
    return report

opt = QuickOptimizer(cs, max_fidelity=100, cost_aware=True, ...)
tuner = QuickTuner(opt, fn)
tuner.run(fevals=100, time_budget=3600)
```

For more code examples, explore the [examples](examples) folder.

### Usage Examples
For further customization options and advanced usage, please refer to our [documentation](docs).

## References

The concepts and methodologies of QuickTuneTool are detailed in the following workshop paper:

- **Title**: *Quick-Tune-Tool: A Practical Tool and its User Guide for Automatically Fine-Tuning Pretrained Models*  
- **Authors**: Ivo Rapant, Lennart Purucker, Fabio Ferreira, Sebastian Pineda Arango, Arlind Kadra, Josif Grabocka, Frank Hutter  
- **Event**: AutoML 2024 Workshop  
- **Availability**: The full paper is accessible on [OpenReview](https://openreview.net/forum?id=d0Hapti3Uc), where research details and methodology discussions can be found.

### Citation
If you use QuickTuneTool in your research, please also cite the following paper:

```
@inproceedings{
arango2024quicktune,
title={Quick-Tune: Quickly Learning Which Pretrained Model to Finetune and How},
author={Sebastian Pineda Arango and Fabio Ferreira and Arlind Kadra and Frank Hutter and Josif Grabocka},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=tqh1zdXIra}
}
```
