import os
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from src.shipment_prep import FEATURES

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["delivery_hours"]

X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

scaler = StandardScaler().fit(X_train)
train_scaled = scaler.transform(X_train)
test_scaled = scaler.transform(X_test)
pd.DataFrame([
    ["train_rows", len(X_train)], ["val_rows", len(X_val)], ["test_rows", len(X_test)],
    ["train_scaled_max_abs_mean", np.abs(train_scaled.mean(axis=0)).max()],
    ["test_scaled_max_abs_mean", np.abs(test_scaled.mean(axis=0)).max()],
], columns=["metric", "value"]).round(4).to_csv("outputs/part_b_checks.csv", index=False)

baseline_pred = np.full(len(y_val), y_train.mean())

def metrics(y_true, pred):
    mse = mean_squared_error(y_true, pred)
    return mean_absolute_error(y_true, pred), mse, np.sqrt(mse), r2_score(y_true, pred)

baseline = metrics(y_val, baseline_pred)

depths = [1, 2, 3, 4, 6, 8, 12, None]
tree_rows = []
for d in depths:
    m = DecisionTreeRegressor(max_depth=d, random_state=42)
    m.fit(X_train, y_train)
    tr = metrics(y_train, m.predict(X_train))[2]
    va = metrics(y_val, m.predict(X_val))[2]
    tree_rows.append(["unlimited" if d is None else d, tr, va])
tree_df = pd.DataFrame(tree_rows, columns=["max_depth", "train_rmse", "val_rmse"])
tree_df.round(4).to_csv("outputs/part_b_tree_depth.csv", index=False)
best_depth = int(tree_df.loc[tree_df["val_rmse"].idxmin(), "max_depth"])

models = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}
rows = []
rows.append(["mean_baseline", "val", *baseline])
val_scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    score = metrics(y_val, model.predict(X_val))
    val_scores[name] = score
    rows.append([name, "val", *score])

lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    m = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
    m.fit(X_train, y_train)
    _, _, rmse, r2 = metrics(y_val, m.predict(X_val))
    lr_rows.append([lr, rmse, r2])
pd.DataFrame(lr_rows, columns=["learning_rate", "val_rmse", "val_r2"]).round(4).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

ranked = sorted(models, key=lambda n: val_scores[n][2])
cv = KFold(n_splits=5, shuffle=True, random_state=42)
X_tv = pd.concat([X_train, X_val])
y_tv = pd.concat([y_train, y_val])
cv_rows = []
for name in ranked[:2]:
    # Rebuild so CV is independent of the already-fitted training-only instance.
    if name == "linear_regression":
        cv_model = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    elif name == "decision_tree":
        cv_model = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
    elif name == "random_forest":
        cv_model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        cv_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    scores = cross_val_score(cv_model, X_tv, y_tv, cv=cv)
    cv_rows.append([name, scores.mean(), scores.std()])
pd.DataFrame(cv_rows, columns=["model", "mean_r2", "std_r2"]).round(4).to_csv("outputs/part_b_cv.csv", index=False)

winner = ranked[0]
final_test = metrics(y_test, models[winner].predict(X_test))
rows.append([winner, "test", *final_test])
pd.DataFrame(rows, columns=["model", "split", "MAE", "MSE", "RMSE", "R2"]).round(4).to_csv("outputs/part_b_metrics.csv", index=False)
