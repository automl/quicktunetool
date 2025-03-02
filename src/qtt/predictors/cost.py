import logging
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.typing import ArrayLike
from sklearn import preprocessing  # type: ignore
from torch.utils.data import DataLoader, random_split

from ..utils.log_utils import set_logger_verbosity
from .data import (
    SimpleTorchTabularDataset,
    create_preprocessor,
    get_feature_mapping,
    get_types_of_features,
)
from .models import MLP
from .predictor import Predictor
from .utils import MetricLogger, get_torch_device

logger = logging.getLogger(__name__)

DEFAULT_FIT_PARAMS = {
    "learning_rate_init": 0.0001,
    "batch_size": 1024,
    "max_iter": 100,
    "early_stop": True,
    "patience": 5,
    "validation_fraction": 0.1,
    "tol": 1e-4,
}


class CostPredictor(Predictor):
    """A predictor that predicts the cost of training a configuration on a new dataset."""

    temp_file_name = "temp_model.pt"

    def __init__(
        self,
        fit_params: dict = {},
        # refit_params: dict = {},
        path: str | None = None,
        seed: int | None = None,
        verbosity: int = 2,
    ) -> None:
        super().__init__(path=path)

        self.fit_params = self._validate_fit_params(fit_params, DEFAULT_FIT_PARAMS)
        self.seed = seed
        self.verbose = verbosity

        set_logger_verbosity(verbosity, logger)

    @staticmethod
    def _validate_fit_params(fit_params, default_params):
        if not isinstance(fit_params, dict):
            raise ValueError("fit_params must be a dictionary")
        for key in fit_params:
            if key not in default_params:
                raise ValueError(f"Unknown fit parameter: {key}")
        return {**default_params, **fit_params}

    def _get_model(self):
        params = {
            "in_dim": [
                len(self.types_of_features["continuous"]),
                len(self.types_of_features["categorical"]) + len(self.types_of_features["bool"]),
            ],
            "enc_out_dim": 16,
            "enc_nlayers": 3,
            "enc_hidden_dim": 128,
        }
        model = SimpleMLPRegressor(**params)
        return model

    def _validate_fit_data(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas.DataFrame instance")

        if not isinstance(y, np.ndarray):
            raise ValueError("y must be a numpy.ndarray instance")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        if y.shape[1] != 1:
            raise ValueError("y must have only one column")

        if len(set(X.columns)) < len(X.columns):
            raise ValueError(
                "Column names are not unique, please change duplicated column names (in pandas: train_data.rename(columns={'current_name':'new_name'})"
            )

    def _validate_predict_data(self, pipeline):
        if not isinstance(pipeline, pd.DataFrame):
            raise ValueError("pipeline and curve must be pandas.DataFrame instances")

        if len(set(pipeline.columns)) < len(pipeline.columns):
            raise ValueError(
                "Column names are not unique, please change duplicated column names (in pandas: train_data.rename(columns={'current_name':'new_name'})"
            )

    def _preprocess_fit_data(self, df: pd.DataFrame, array: np.ndarray):
        """
        Process data for fitting the model.
        """
        self._original_features = list(df.columns)

        df, self.types_of_features, self.features_to_drop = get_types_of_features(df)
        self._input_features = list(df.columns)
        continous_features = self.types_of_features["continuous"]
        categorical_features = self.types_of_features["categorical"]
        bool_features = self.types_of_features["bool"]
        self.preprocessor = create_preprocessor(
            continous_features, categorical_features, bool_features
        )
        out = self.preprocessor.fit_transform(df)
        self._feature_mapping = get_feature_mapping(self.preprocessor)
        if out.shape[1] != sum(len(v) for v in self._feature_mapping.values()):
            raise ValueError(
                "Error during one-hot encoding data processing for neural network. "
                "Number of columns in df array does not match feature_mapping."
            )

        self.label_scaler = preprocessing.StandardScaler()  # MaxAbsScaler()
        out_array = self.label_scaler.fit_transform(array)

        return out, out_array

    def _preprocess_predict_data(self, df: pd.DataFrame, fill_missing=True):
        unexpected_columns = set(df.columns) - set(self._original_features)
        if len(unexpected_columns) > 0:
            logger.warning(
                "Data contains columns that were not present during fitting: "
                f"{unexpected_columns}"
            )

        df = df.drop(columns=self.features_to_drop, errors="ignore")

        missing_columns = set(self._input_features) - set(df.columns)
        if len(missing_columns) > 0:
            if fill_missing:
                logger.warning(
                    "Data is missing columns that were present during fitting: "
                    f"{missing_columns}. Trying to fill them with mean values / zeros."
                )
                for col in missing_columns:
                    df[col] = None
            else:
                raise AssertionError(
                    "Data is missing columns that were present during fitting: "
                    f"{missing_columns}. Please fill them with appropriate values."
                )
        X = self.preprocessor.transform(df)
        X = np.array(X)
        X = np.nan_to_num(X)
        return X

    def _fit_model(
        self,
        dataset,
        learning_rate_init,
        batch_size,
        max_iter,
        early_stop,
        patience,
        validation_fraction,
        tol,
    ):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.device = get_torch_device()
        _dev = self.device
        self.model.to(_dev)

        optimizer = torch.optim.AdamW(self.model.parameters(), learning_rate_init)

        patience_counter = 0
        best_iter = 0
        best_val_metric = np.inf

        if patience is not None:
            if early_stop:
                if validation_fraction < 0 or validation_fraction > 1:
                    raise AssertionError(
                        "validation_fraction must be between 0 and 1 when early_stop is True"
                    )
                logger.info(
                    f"Early stopping on validation loss with patience {patience} "
                    f"using {validation_fraction} of the data for validation"
                )
                train_set, val_set = random_split(
                    dataset=dataset,
                    lengths=[1 - validation_fraction, validation_fraction],
                )
            else:
                logger.info(f"Early stopping on training loss with patience {patience}")
                train_set = dataset
                val_set = None
        else:
            train_set = dataset
            val_set = None

        bs = min(batch_size, int(2 ** (3 + np.floor(np.log10(len(train_set))))))
        train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, drop_last=True)
        val_loader = None
        if val_set is not None:
            bs = min(batch_size, int(2 ** (3 + np.floor(np.log10(len(val_set))))))
            val_loader = DataLoader(val_set, batch_size=bs)

        cache_dir = os.path.expanduser("~/.cache")
        cache_dir = os.path.join(cache_dir, "qtt", self.name)
        os.makedirs(cache_dir, exist_ok=True)
        temp_save_file_path = os.path.join(cache_dir, self.temp_file_name)
        for it in range(1, max_iter + 1):
            self.model.train()

            train_loss = []
            header = f"TRAIN: ({it}/{max_iter})"
            metric_logger = MetricLogger(delimiter=" ")
            for batch in metric_logger.log_every(
                train_loader, len(train_loader) // 10, header, logger
            ):
                # forward
                batch = [item.to(_dev) for item in batch]
                X, y = batch
                loss = self.model.train_step(X, y)
                train_loss.append(loss.item())

                # update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                metric_logger.update(loss=loss.item())
            logger.info(f"Averaged stats: {str(metric_logger)}")
            val_metric = np.mean(train_loss)

            if val_loader is not None:
                self.model.eval()

                val_loss = []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = [item.to(_dev) for item in batch]
                        X, y = batch
                        pred = self.model.predict(X)
                        loss = torch.nn.functional.l1_loss(pred, y)
                        val_loss.append(loss.item())
                val_metric = np.mean(val_loss)

            if patience is not None:
                if val_metric + tol < best_val_metric:
                    patience_counter = 0
                    best_val_metric = val_metric
                    best_iter = it
                    torch.save(self.model.state_dict(), temp_save_file_path)
                else:
                    patience_counter += 1
                logger.info(
                    f"VAL: {round(val_metric, 4)}  "
                    f"ITER: {it}/{max_iter}  "
                    f"BEST: {round(best_val_metric, 4)} ({best_iter})"
                )
                if patience_counter >= patience:
                    logger.warning(
                        "Early stopping triggered! "
                        f"No improvement in the last {patience} iterations. "
                        "Stopping training..."
                    )
                    break

        if early_stop:
            self.model.load_state_dict(torch.load(temp_save_file_path, weights_only=True))

    def _fit(
        self,
        X: pd.DataFrame,
        y: ArrayLike,
        **kwargs,
    ):
        if self.is_fit:
            raise AssertionError("Predictor is already fit! Create a new one.")

        y = np.array(y)

        self._validate_fit_data(X, y)
        _X, _y = self._preprocess_fit_data(X, y)

        train_dataset = SimpleTorchTabularDataset(_X, _y)

        self.model = self._get_model()

        self._fit_model(train_dataset, **self.fit_params)

        return self

    def _predict(self, **kwargs) -> np.ndarray:
        """Predict the costs of training a configuration on a new dataset.

        Args:
            X (pd.DataFrame): the configuration to predict.
        """
        if not self.is_fit or self.model is None:
            raise AssertionError("Model is not fitted yet")

        X: pd.DataFrame = kwargs.pop("X", None)
        if X is None:
            raise ValueError("X (pipeline configuration) must be provided")

        self._validate_predict_data(X)
        x = self._preprocess_predict_data(X)

        self.model.eval()
        self.model.to(self.device)
        x_t = torch.tensor(x, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            pred = self.model.predict(x_t)
        out = pred.cpu().squeeze().numpy()
        return out

    def save(self, path: str | None = None, verbose=True) -> str:
        # Save on CPU to ensure the model can be loaded on a box without GPU
        if self.model is not None:
            self.model = self.model.to(torch.device("cpu"))
        path = super().save(path, verbose)
        # Put the model back to the device after the save
        if self.model is not None:
            self.model.to(self.device)
        return path

    @classmethod
    def load(cls, path: str, reset_paths=True, verbose=True):
        """
        Loads the model from disk to memory.
        The loaded model will be on the same device it was trained on (cuda/mps);
        if the device is it's not available (trained on GPU, deployed on CPU),
        then `cpu` will be used.

        Parameters
        ----------
        path : str
            Path to the saved model, minus the file name.
            This should generally be a directory path ending with a '/' character (or appropriate path separator value depending on OS).
            The model file is typically located in os.path.join(path, cls.model_file_name).
        reset_paths : bool, default True
            Whether to reset the self.path value of the loaded model to be equal to path.
            It is highly recommended to keep this value as True unless accessing the original self.path value is important.
            If False, the actual valid path and self.path may differ, leading to strange behaviour and potential exceptions if the model needs to load any other files at a later time.
        verbose : bool, default True
            Whether to log the location of the loaded file.

        Returns
        -------
        model : cls
            Loaded model object.
        """
        model: CostPredictor = super().load(path=path, reset_paths=reset_paths, verbose=verbose)
        return model


class SimpleMLPRegressor(torch.nn.Module):
    def __init__(
        self,
        in_dim: int | list[int],
        enc_out_dim: int = 8,
        enc_nlayers: int = 3,
        enc_hidden_dim: int = 128,
    ):
        super().__init__()
        if isinstance(in_dim, int):
            in_dim = [in_dim]
        self.in_dim = in_dim

        # build config encoder
        encoder = nn.ModuleList()
        for dim in self.in_dim:
            encoder.append(MLP(dim, enc_out_dim, enc_nlayers, enc_hidden_dim))
        self.config_encoder = encoder
        enc_dims = len(self.config_encoder) * enc_out_dim

        self.head = MLP(enc_dims, 1, enc_nlayers, enc_hidden_dim, act_fn=nn.GELU)

    def forward(self, X) -> torch.Tensor:
        x = []
        start = 0
        for i, dim in enumerate(self.in_dim):
            end = start + dim
            output = self.config_encoder[i](X[:, start:end])
            x.append(output)
            start = end
        t = torch.cat(x, dim=1)
        t = self.head(t)
        return t

    def predict(self, X) -> torch.Tensor:
        return self(X)

    def train_step(self, X, y) -> torch.Tensor:
        pred = self(X)
        loss = torch.nn.functional.huber_loss(pred, y)
        return loss
