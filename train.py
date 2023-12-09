import os
import pickle

import pandas as pd
from catboost import CatBoostClassifier


dftrain = pd.read_csv("dftrain.csv")

clf = CatBoostClassifier(
    random_state=123, loss_function="MultiClass", eval_metric="TotalF1"
)

clf.fit(dftrain, dftrain["target"])

model_pickle_file = "Catboost_model.pkl"

if os.path.exists(model_pickle_file):
    os.remove(model_pickle_file)

with open(model_pickle_file, "wb") as handle:
    pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
