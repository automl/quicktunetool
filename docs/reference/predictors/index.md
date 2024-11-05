# Overview

The `Predictor` class serves as a base class for implementing predictive models within the Quick-Tune-Tool. It provides core functionality for model setup, data handling, training, and persistence (saving/loading), allowing specific predictive models to extend and customize these methods.

#### Core Methods

- **`fit`** and **`_fit`**: 
    - `fit`: Public method for training the model. It takes feature data `X`, target labels `y`, verbosity level, and any additional arguments.
    - `_fit`: Abstract method where specific model training logic is implemented. Models inheriting from `Predictor` should override `_fit` to implement their own fitting procedures.

- **`preprocess`** and **`_preprocess`**: 
    - `preprocess`: Wrapper method that calls `_preprocess` to prepare data for fitting or prediction.
    - `_preprocess`: Abstract method where data transformation logic should be added. Designed to clean and structure input data before model training or inference.    

- **`load`** and **`save`**:
    - `load`: Class method to load a saved model from disk, optionally resetting its path and logging the location.
    - `save`: Saves the current model to disk in a specified path, providing persistence for trained models.

- **`predict`**:<br>
Abstract method for generating predictions on new data. Specific predictive models should implement this method based on their inference logic.

This `Predictor` class offers a foundation for different predictive models, providing essential methods for data handling, training, and saving/loading, with extensibility for custom implementations.

---

#### Available Predictors

- [**`PerfPredictor`**][qtt.predictors.perf]
  Predicts the performance of a configuration on a new dataset.
- [**`CostPredictor`**][qtt.predictors.cost]
  Predicts the cost of training a configuration on a new dataset.