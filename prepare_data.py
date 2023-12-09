import os

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


wine = load_wine()
X = wine.data
y = wine.target
df = pd.DataFrame(X, columns=wine.feature_names)

dftrain, dftest, ytrain, ytest = train_test_split(
    df, y, test_size=0.3, stratify=y, random_state=123
)

dftrain["target"] = ytrain
dftest["target"] = ytest

if os.path.exists("dftrain.csv"):
    os.remove("dftrain.csv")
if os.path.exists("dftest.csv"):
    os.remove("dftest.csv")

dftrain.to_csv("dftrain.csv", index=False)
dftest.to_csv("dftest.csv", index=False)
