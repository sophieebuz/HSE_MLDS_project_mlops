import pandas as pd
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier


dftest = pd.read_csv("data/dftest.csv")

model_save_file = "Catboost_model.cbm"
from_file = CatBoostClassifier()
clf = from_file.load_model(model_save_file)

y_pred = clf.predict(dftest)
print(classification_report(dftest["target"], y_pred))
