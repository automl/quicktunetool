import logging
import os
import pickle

import pandas as pd
from numpy.typing import ArrayLike

from qtt.utils import setup_outputdir

logger = logging.getLogger(__name__)


class AbstractPredictor:
    """
    Abstract predictor implementation from which all predictors inherit.

    Parameters
    ----------
    path : str, default = None
        Directory location to store all outputs.
        If None, a new unique time-stamped directory is chosen.
    name : str, default = None
        Name of the subdirectory inside path where model will be saved.
        The final model directory will be os.path.join(path, name)
        If None, defaults to the model's class name: self.__class__.__name__
    """

    model_file_name = "model.pkl"
    model_info_name = "info.pkl"
    model_info_json_name = "info.json"

    def __init__(
        self,
        name: str | None = None,
        path: str | None = None,
    ):
        if name is None:
            self.name = self.__class__.__name__
            logger.info(
                f"No name was specified for model, defaulting to class name: {self.name}",
            )
        else:
            self.name = name

        self.path_root = path
        if self.path_root is None:
            path_suffix = self.name
            path_cur = setup_outputdir(
                path=None, create_dir=True, path_suffix=path_suffix
            )
            self.path_root = path_cur.rsplit(self.path_suffix, 1)[0]
            logger.warning(
                f"No path was specified for model, defaulting to: {self.path_root}",
            )

        self.path = self.create_contexts(os.path.join(self.path_root, self.path_suffix))

        self.model = None
        self.features = None
        self.val_score = None

        self._is_initialized = False

    @property
    def is_fit(self) -> bool:
        """Returns True if the model has been fit."""
        return self.model is not None

    def fit(self, *args, **kwargs):
        """
        Fit model to predict values in y based on X.

        Models should not override the `fit` method, but instead override the `_fit` method which has the same arguments.

        Parameters
        ----------
        X : DataFrame
            The training data features.
        y : ArrayLike
            The training data ground truth labels.
        verbosity : int, default = 2
            Verbosity levels range from 0 to 4 and control how much information is printed.
            Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).
            verbosity 4: logs every training iteration, and logs the most detailed information.
            verbosity 3: logs training iterations periodically, and logs more detailed information.
            verbosity 2: logs only important information.
            verbosity 1: logs only warnings and exceptions.
            verbosity 0: logs only exceptions.
        **kwargs :
            Any additional fit arguments a model supports.
        """
        out = self._fit(*args, **kwargs)
        if out is None:
            out = self
        return out

    def _fit(
        self,
        X: pd.DataFrame,
        y: ArrayLike,
        verbosity: int = 2,
        **kwargs,
    ):
        """
        Fit model to predict values in y based on X.

        Models should override this method with their custom model fit logic.
        X should not be assumed to be in a state ready for fitting to the inner model, and models may require special preprocessing in this method.
        It is very important that `X = self.preprocess(X)` is called within `_fit`, or else `predict` and `predict_proba` may not work as intended.
        It is also important that `_preprocess` is overwritten to properly clean the data.
        Examples of logic that should be handled by a model include missing value handling, rescaling of features (if neural network), etc.
        If implementing a new model, it is recommended to refer to existing model implementations and experiment using toy datasets.

        Refer to `fit` method for documentation.
        """
        raise NotImplementedError

    def _preprocess(self, **kwargs):
        """
        Data transformation logic should be added here.

        Input data should not be trusted to be in a clean and ideal form, while the output should be in an ideal form for training/inference.
        Examples of logic that should be added here include missing value handling, rescaling of features (if neural network), etc.
        If implementing a new model, it is recommended to refer to existing model implementations and experiment using toy datasets.

        In bagged ensembles, preprocessing code that lives in `_preprocess` will be executed on each child model once per inference call.
        If preprocessing code could produce different output depending on the child model that processes the input data, then it must live here.
        When in doubt, put preprocessing code here instead of in `_preprocess_nonadaptive`.
        """
        raise NotImplementedError

    def preprocess(self, **kwargs):
        """
        Preprocesses the input data into internal form ready for fitting or inference.
        """
        return self._preprocess(**kwargs)

    @property
    def path_suffix(self) -> str:
        return self.name

    @staticmethod
    def create_contexts(path_context: str) -> str:
        path = path_context
        return path

    def set_contexts(self, path_context):
        self.path = self.create_contexts(path_context)
        self.path_root = self.path.rsplit(self.path_suffix, 1)[0]

    @classmethod
    def load(cls, path: str, reset_paths: bool = True, verbose: bool = True):
        """
        Loads the model from disk to memory.

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
        file_path = os.path.join(path, cls.model_file_name)
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        if reset_paths:
            model.set_contexts(path)
        return model

    def save(self, path: str | None = None, verbose: bool = True) -> str:
        """
        Saves the model to disk.

        Parameters
        ----------
        path : str, default None
            Path to the saved model, minus the file name.
            This should generally be a directory path ending with a '/' character (or appropriate path separator value depending on OS).
            If None, self.path is used.
            The final model file is typically saved to os.path.join(path, self.model_file_name).
        verbose : bool, default True
            Whether to log the location of the saved file.

        Returns
        -------
        path : str
            Path to the saved model, minus the file name.
            Use this value to load the model from disk via cls.load(path), cls being the class of the model object, such as model = RFModel.load(path)
        """
        if path is None:
            path = self.path
        file_path = os.path.join(path, self.model_file_name)
        with open(file_path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        if verbose:
            logger.info(f"Model saved to: {file_path}")
        return path