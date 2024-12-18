from qtt import QuickOptimizer, QuickTuner
from qtt.finetune.image.classification import extract_image_dataset_metafeat, fn
import pandas as pd
from ConfigSpace import ConfigurationSpace

pipeline = pd.read_csv("pipeline.csv", index_col=0)
curve = pd.read_csv("curve.csv", index_col=0)
cost = pd.read_csv("cost.csv", index_col=0)
meta = pd.read_csv("meta.csv", index_col=0)
cs = ConfigurationSpace.from_yaml("space.yaml")

config = pd.merge(pipeline, meta, on="dataset")
config.drop(("dataset"), axis=1, inplace=True)
opt = QuickOptimizer(cs, 50, cost_aware=True)

ti, mf = extract_image_dataset_metafeat("path/to/dataset")
opt.setup(128, mf)

qt = QuickTuner(opt, fn)
qt.run(100, trial_info=ti)
