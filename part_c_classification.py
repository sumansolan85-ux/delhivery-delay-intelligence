import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from src.shipment_prep import FEATURES, NOISE_COLUMNS

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["is_delayed"]

X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

assert len(X_train) == 840 and len(X_val) == 180 and len(X_test) == 180


def row_metrics(y_true, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return [tn, fp, fn, tp, accuracy_score(y_true, pred), precision_score(y_true, pred, zero_division=0), recall_score(y_true, pred, zero_division=0), f1_score(y_true, pred, zero_division=0)]

rows = []
blind = np.zeros(len(y_val), dtype=int)
rows.append(["blind_all_on_time", *row_metrics(y_val, blind)])

logreg = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])
logreg.fit(X_train, y_train)
rows.append(["logistic_regression", *row_metrics(y_val, logreg.predict(X_val))])

tree_candidates = []
for d in [2, 3, 4, 5, 6, 8, 10]:
    m = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
    m.fit(X_train, y_train)
    f1 = f1_score(y_val, m.predict(X_val), zero_division=0)
    tree_candidates.append((d, f1, m))
best_tree_depth = sorted(tree_candidates, key=lambda x: (-x[1], x[0]))[0][0]
best_tree = [x[2] for x in tree_candidates if x[0] == best_tree_depth][0]
rows.append([f"decision_tree_depth_{best_tree_depth}", *row_metrics(y_val, best_tree.predict(X_val))])

rf_default_17 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_default_17.fit(X_train, y_train)
rows.append(["random_forest_default", *row_metrics(y_val, rf_default_17.predict(X_val))])

pd.DataFrame(rows, columns=["model", "TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"]).round(4).to_csv("outputs/part_c_models.csv", index=False)

# Feature signal check on training rows only.
selector_all = SelectKBest(f_classif, k="all").fit(X_train, y_train)
f_scores = selector_all.scores_
rf_imp = rf_default_17.feature_importances_
feature_scores = pd.DataFrame({"feature": FEATURES, "f_score": f_scores, "rf_importance": rf_imp})
feature_scores["f_rank"] = feature_scores["f_score"].rank(method="min", ascending=False).astype(int)
feature_scores["is_noise"] = feature_scores["feature"].isin(NOISE_COLUMNS)
feature_scores = feature_scores[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]].sort_values("feature")
feature_scores.round(4).to_csv("outputs/part_c_feature_scores.csv", index=False)

selectk_rows = []
for k in [5, 7, 10, 12, 17]:
    pipe = Pipeline([("selector", SelectKBest(f_classif, k=k)), ("model", RandomForestClassifier(n_estimators=100, random_state=42))])
    pipe.fit(X_train, y_train)
    selectk_rows.append([k, f1_score(y_val, pipe.predict(X_val), zero_division=0)])
pd.DataFrame(selectk_rows, columns=["k", "val_f1"]).round(4).to_csv("outputs/part_c_selectk.csv", index=False)

selector7 = SelectKBest(f_classif, k=7).fit(X_train, y_train)
Xtr7 = selector7.transform(X_train)
Xv7 = selector7.transform(X_val)
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10],
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5)
grid.fit(Xtr7, y_train)
rand = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions=param_grid, n_iter=9, cv=5, random_state=42)
rand.fit(Xtr7, y_train)

def params_text(p):
    return json.dumps(p, sort_keys=True, separators=(",", ":"))

def val_f1_for(est):
    return f1_score(y_val, est.predict(Xv7), zero_division=0)

default7 = RandomForestClassifier(n_estimators=100, random_state=42)
default7.fit(Xtr7, y_train)

tuning_rows = [
    ["grid_search", 27, 5, 135, params_text(grid.best_params_), grid.best_score_, val_f1_for(grid.best_estimator_)],
    ["randomized_search", 9, 5, 45, params_text(rand.best_params_), rand.best_score_, val_f1_for(rand.best_estimator_)],
    ["default_forest_same_7_features", 1, 0, 1, "defaults", "", val_f1_for(default7)],
]
pd.DataFrame(tuning_rows, columns=["search", "candidates", "cv_folds", "total_fits", "best_params", "best_cv_accuracy", "val_f1"]).round(4).to_csv("outputs/part_c_tuning.csv", index=False)

# Choose by validation F1, with the required tie order.
priority = {"default_forest_same_7_features": 0, "grid_search": 1, "randomized_search": 2}
choice = max(tuning_rows, key=lambda r: (r[6], -priority[r[0]]))
chosen_from = choice[0]
if chosen_from == "grid_search":
    chosen_params = grid.best_params_
elif chosen_from == "randomized_search":
    chosen_params = rand.best_params_
else:
    chosen_params = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}

final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=7)),
    ("model", RandomForestClassifier(random_state=42, **chosen_params)),
])
final_model.fit(X_train, y_train)
val_prob = final_model.predict_proba(X_val)[:, 1]
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
threshold_rows = []
for t in thresholds:
    pred = (val_prob >= t).astype(int)
    threshold_rows.append([t, *row_metrics(y_val, pred)])
th_df = pd.DataFrame(threshold_rows, columns=["threshold", "TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"])
th_df.round(4).to_csv("outputs/part_c_thresholds.csv", index=False)
valid = th_df[th_df["recall"] >= 0.80]
threshold = float(valid["threshold"].max()) if not valid.empty else 0.1

# Exactly one test evaluation.
test_pred = (final_model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
final_test_metrics = row_metrics(y_test, test_pred)
pd.DataFrame([[threshold, *final_test_metrics]], columns=["threshold", "TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"]).round(4).to_csv("outputs/part_c_final_test.csv", index=False)

max_depth_out = "unlimited" if chosen_params.get("max_depth") is None else chosen_params.get("max_depth")
pd.DataFrame([[7, chosen_params.get("n_estimators"), max_depth_out, chosen_params.get("min_samples_leaf"), threshold, chosen_from]],
             columns=["selector_k", "n_estimators", "max_depth", "min_samples_leaf", "threshold", "chosen_from"]).to_csv("outputs/part_c_final_config.csv", index=False)
