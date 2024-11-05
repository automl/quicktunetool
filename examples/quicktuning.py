"""A quick example of using QuickCVCLSTuner to tune vision classifiers on a dataset."""
from qtt import QuickTuner_CVCLS
tuner = QuickTuner_CVCLS("path/to/dataset")
tuner.run(fevals=100, time_budget=3600)