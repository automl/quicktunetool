import numpy as np

from ....finetune.image.classification import extract_image_dataset_metafeat, fn
from ....optimizers.quick import QuickOptimizer
from ....utils.pretrained import get_pretrained_optimizer
from ...quick import QuickTuner


class QuickImageCLSTuner(QuickTuner):
    """QuickTuner for image classification.

    Args:
        data_path (str): Path to the dataset.
        path (str, optional): Path to save the optimizer. Defaults to None.
        verbosity (int, optional): Verbosity level. Defaults to 2.
    """

    def __init__(
        self,
        data_path: str,
        n: int = 512,
        path: str | None = None,
        verbosity: int = 2,
    ):
        quick_opt: QuickOptimizer = get_pretrained_optimizer("mtlbm/full")

        trial_info, metafeat = extract_image_dataset_metafeat(data_path)
        quick_opt.setup(n, metafeat=metafeat)

        self.trial_info = trial_info

        super().__init__(quick_opt, fn, path=path, verbosity=verbosity)

    def run(
        self,
        fevals: int | None = None,
        time_budget: float | None = None,
        trial_info: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """

        Args:
            fevals (int, optional): Number of function evaluations to run. Defaults to None.
            time_budget (float, optional): Time budget in seconds. Defaults to None.
            trial_info (dict, optional): Additional information to pass to the objective function. Defaults to None.

        Returns:
            - np.ndarray: Trajectory of the incumbent scores.
            - np.ndarray: Runtime of the incumbent evaluations.
            - np.ndarray: History of all evaluations.
        """
        if trial_info is not None:
            self.trial_info = trial_info
        return super().run(fevals=fevals, time_budget=time_budget, trial_info=self.trial_info)
