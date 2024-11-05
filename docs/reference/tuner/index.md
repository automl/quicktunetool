# Overview

The `QuickTuner` class is a high-level tuner designed to optimize a given objective function by managing iterative evaluations and coordinating with an `Optimizer`. It provides comprehensive functionality for logging, result tracking, checkpointing, and handling evaluation budgets.

#### Core Methods

- **`run`**: Executes the optimization process within a specified budget of function evaluations (`fevals`) or time (`time_budget`). This method iteratively:
    - Requests new configurations from the optimizer.
    - Evaluates configurations using the objective function `f`.
    - Updates the optimizer with evaluation results and logs progress.
    - Saves results based on the specified `save_freq` setting.

- **`save`** and **`load`**: 
    - `save`: Saves the current state of the tuner, including the incumbent, evaluation history, and tuner state.
    - `load`: Loads a previously saved tuner state to resume optimization from where it left off.

#### Usage Example

The `QuickTuner` is typically used to optimize an objective function with the support of an optimizer, managing configuration sampling, evaluation, and tracking. It is particularly suited for iterative optimization tasks where tracking the best configuration and logging results are essential.
