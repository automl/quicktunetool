# Overview

The `Optimizer` class serves as a base class within the Quick-Tune-Tool, providing low-level functionality. It is designed to support flexible configuration management and interact with tuners during the optimization process. Key aspects of the class include directory setup, model saving, and interfacing methods for requesting and reporting trial evaluations.

Here’s an overview of the [`Optimizer`][qtt.optimizers.optimizer] class:

#### Core Methods

- **`ask`**: Abstract method that must be implemented in subclasses. It requests a new configuration trial from the optimizer, returning it as a dictionary. Raises `NotImplementedError` if not overridden.

- **`tell`**: Accepts and processes a report (result) from a trial evaluation. This method allows the optimizer to record outcomes for each configuration and adjust future suggestions accordingly. Supports both single and multiple trial reports.

- **`ante`**: A placeholder method for pre-processing tasks to be performed before requesting a configuration trial (used by tuners). Can be overridden in subclasses for custom pre-processing.

- **`post`**: A placeholder for post-processing tasks, executed after a trial evaluation has been submitted. Designed

This class is intended to be extended for specific optimization strategies, with `ask` and `tell` as the primary methods for interaction with tuners.

---

### Available Optimizers

- [**`RandomOptimizer`**][qtt.optimizers.rndm]
- [**`QuickOptimizer`**][qtt.optimizers.quick]