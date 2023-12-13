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

if not os.path.isdir("data"):
     os.mkdir("data")
if os.path.exists("data/dftrain.csv"):
    os.remove("data/dftrain.csv")
if os.path.exists("data/dftest.csv"):
    os.remove("data/dftest.csv")

dftrain.to_csv("data/dftrain.csv", index=False)
dftest.to_csv("data/dftest.csv", index=False)
