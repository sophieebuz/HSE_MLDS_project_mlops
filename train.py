import os

import pandas as pd
from catboost import CatBoostClassifier


dftrain = pd.read_csv("data/dftrain.csv")

clf = CatBoostClassifier(
    random_state=123, loss_function="MultiClass", eval_metric="TotalF1"
)

clf.fit(dftrain, dftrain["target"])

model_save_file = "Catboost_model.cbm"

if os.path.exists(model_save_file):
    os.remove(model_save_file)

clf.save_model(model_save_file, format="cbm")
