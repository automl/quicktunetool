# Code References

This section provides references for the core components of Quick-Tune-Tool, describing the primary modules and classes that make up the tool's architecture. The code is organized into three main parts: **Optimizers**, **Predictors**, and **Tuners**.

---

## 1. Optimizers

The Optimizers module is responsible for suggesting configurations for evaluation, using various optimization strategies. Available optimizers include:

- **QuickTune Optimizer**  
  - File: `optimizers/quick.py`
  - Implements the QuickTune algorithm, balancing multi-fidelity expected improvement with cost estimation to select configurations.
  
- **Random Search Optimizer**  
  - File: `optimizers/random.py`
  - Provides a basic random search optimizer as a baseline for comparison with other optimization strategies.

---

## 2. Predictors

The Predictors module includes components that estimate model performance and finetuning costs, enabling efficient configuration selection.

- **Performance Predictor**
    - File: `predictors/perf.py`
    - Uses meta-learning to estimate the potential performance of a model configuration based on historical data and auxiliary task information.

- **Cost Predictor**
    - File: `predictors/cost.py`
    - Evaluates the computational cost associated with different finetuning configurations, helping to balance resource efficiency with optimization goals.

---

## 3. Tuners

The Tuners module coordinates the tuning process, managing environment setup, experiment flow, and result handling.

- **QuickTuner**  
    - File: `tuners/quick.py`
    - Serves as the central class that manages the tuning process, integrating optimizers and predictors to manage iterative evaluations and updates.

- **Image-Classification**  
    - File: `tuners/image/classification/tuner.py`
    - A specialized tuner for image classification, offering a reduced interface where users simply provide the path to the image dataset.

---

## Additional Resources

- **Finetuning Scripts**  
    - Directory: `scripts/`
    - Functions used to evaluate configurations, returning performance metrics for each step.
  
- **Utility Scripts**  
    - Directory: `utils/`
    - A collection of helper functions and utilities to support data processing, result logging, and other ancillary tasks.

---

Refer to each module's in-code documentation for further details on function arguments, usage examples, and dependencies.
