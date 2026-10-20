import os
import numpy as np
import pandas as pd
from src.shipment_prep import PROMISED_HOURS, FEATURES, clean_shipments, build_features

os.makedirs("outputs", exist_ok=True)
raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

raw_rows = len(raw)
dedup = raw.drop_duplicates().copy()
service_before = dedup["service_type"].nunique()
missing_distance_before = int(dedup["distance_km"].isna().sum())
missing_weight_before = int(dedup["parcel_weight_kg"].isna().sum())
missing_declared_before = int(dedup["declared_value"].isna().sum())
missing_payment_before = int(dedup["payment_mode"].isna().sum())
sentinel_rows = int((dedup["parcel_weight_kg"] > 100).sum())

clean, medians = clean_shipments(raw)
service_after = clean["service_type"].nunique()
model = build_features(clean, hubs)
model["is_delayed"] = (model["delivery_hours"] > model["promised_hours"]).astype(int)
assert len(model) == len(clean) == 1200

model.to_csv("data/modelling_table.csv", index=False)

q1, q3 = model["delivery_hours"].quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
filled_value = model["declared_value_inr"]
checks = [
    ("raw_rows", raw_rows),
    ("duplicate_rows_removed", raw_rows - len(dedup)),
    ("rows_after_dedup", len(dedup)),
    ("missing_distance_km_before", missing_distance_before),
    ("missing_parcel_weight_before", missing_weight_before),
    ("sentinel_weight_rows", sentinel_rows),
    ("missing_declared_value_before", missing_declared_before),
    ("missing_payment_mode_before", missing_payment_before),
    ("service_type_spellings_before", service_before),
    ("service_type_spellings_after", service_after),
    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),
    ("rows_after_hub_merge", len(model)),
    ("nulls_in_features_and_target", int(model[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum())),
    ("declared_value_mean_inr", filled_value.mean()),
    ("declared_value_median_inr", filled_value.median()),
    ("log_declared_value_mean", np.log(filled_value).mean()),
    ("log_declared_value_median", np.log(filled_value).median()),
    ("delayed_count", int(model["is_delayed"].sum())),
    ("delay_rate", model["is_delayed"].mean()),
    ("hours_q1", q1),
    ("hours_q3", q3),
    ("hours_lower_bound", lower),
    ("hours_upper_bound", upper),
    ("hours_outlier_count", int(((model["delivery_hours"] < lower) | (model["delivery_hours"] > upper)).sum())),
]
pd.DataFrame(checks, columns=["metric", "value"]).round(4).to_csv("outputs/part_a_checks.csv", index=False)

bands = pd.cut(model["distance_km"], bins=[0, 200, 400, 600, 2000], labels=["0-200", "201-400", "401-600", "601+"], include_lowest=True, right=True)
rows = []
for dim, groups in [("service_type", model["service_type"].unique()), ("hub_tier", sorted(model["hub_tier"].unique())), ("distance_band", ["0-200", "201-400", "401-600", "601+"])]:
    if dim == "distance_band":
        temp = model.assign(distance_band=bands)
        group_col = "distance_band"
        groups = [g for g in groups if g in temp[group_col].astype(str).unique()]
    else:
        temp = model
        group_col = dim
    for g in groups:
        sub = temp[temp[group_col].astype(str) == str(g)]
        rows.append([dim, g, len(sub), int(sub["is_delayed"].sum()), sub["is_delayed"].mean(), sub["delivery_hours"].mean()])
pd.DataFrame(rows, columns=["dimension", "group", "shipments", "delayed", "delay_rate", "mean_hours"]).round(4).to_csv("outputs/part_a_delay_by_group.csv", index=False)

corr = model.select_dtypes(include=np.number).corr()
corr.to_csv("outputs/part_a_correlations.csv", index=True, index_label="column")
