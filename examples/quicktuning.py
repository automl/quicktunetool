"""A quick example of using a special QuickTuner to tune image classifiers on a new dataset."""

from qtt import QuickImageCLSTuner

tuner = QuickImageCLSTuner("path/to/dataset")
tuner.run(fevals=100, time_budget=3600)
