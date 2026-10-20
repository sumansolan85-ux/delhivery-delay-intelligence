import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.shipment_prep import FEATURES, clean_shipments

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["is_delayed"]
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
selector_k = int(config["selector_k"])
n_estimators = int(config["n_estimators"])
max_depth = None if str(config["max_depth"]) == "unlimited" else int(config["max_depth"])
min_samples_leaf = int(config["min_samples_leaf"])
threshold = float(config["threshold"])

model = Pipeline([
    ("selector", SelectKBest(f_classif, k=selector_k)),
    ("model", RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)),
])
model.fit(X_train, y_train)
joblib.dump(model, "models/delay_model.pkl")
pd.DataFrame({"threshold": [threshold]}).to_csv("models/delay_threshold.csv", index=False)
pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)

raw = pd.read_csv("data/raw_shipments.csv")
_, medians = clean_shipments(raw)
pd.DataFrame({"column": list(medians.keys()), "median": list(medians.values())}).to_csv("models/impute_values.csv", index=False)

reloaded = joblib.load("models/delay_model.pkl")
p1 = model.predict_proba(X_test)[:, 1]
p2 = reloaded.predict_proba(X_test)[:, 1]
pd.DataFrame([
    ["test_rows_scored", len(X_test)],
    ["reload_max_abs_probability_diff", float(np.max(np.abs(p1 - p2)))],
    ["threshold", threshold],
], columns=["metric", "value"]).round(4).to_csv("outputs/part_d_reload_check.csv", index=False)
