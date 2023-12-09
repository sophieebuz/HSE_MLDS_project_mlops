import pickle

import pandas as pd
from sklearn.metrics import classification_report


dftest = pd.read_csv("dftest.csv")

model_pickle_file = "Catboost_model.pkl"
with open(model_pickle_file, "rb") as file:
    clf = pickle.load(file)

y_pred = clf.predict(dftest)
print(classification_report(dftest["target"], y_pred))
