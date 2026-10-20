import joblib
import numpy as np
import pandas as pd
from src.shipment_prep import FEATURES, clean_shipments, build_features

raw = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
model = joblib.load("models/delay_model.pkl")
threshold = float(pd.read_csv("models/delay_threshold.csv").iloc[0, 0])
feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
medians_df = pd.read_csv("models/impute_values.csv")
medians = dict(zip(medians_df["column"], medians_df["median"]))
segments = pd.read_csv("outputs/hub_segments.csv")

clean, _ = clean_shipments(raw, medians=medians)
feat = build_features(clean, hubs)
X = feat[feature_order]
prob = model.predict_proba(X)[:, 1]
out = pd.DataFrame({
    "shipment_id": clean["shipment_id"].values,
    "hub_id": clean["hub_id"].values,
    "delay_probability": np.round(prob, 4),
})
out["flagged"] = (out["delay_probability"] >= threshold).astype(int)
out = out.merge(segments[["hub_id", "segment"]], on="hub_id", how="left", validate="many_to_one")
out["action"] = np.where(
    out["flagged"].eq(0),
    "No action",
    np.where(out["segment"].eq("Escalate"), "Escalate to hub manager", "Proactive customer call")
)
out = out[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
out.to_csv("outputs/new_shipment_actions.csv", index=False)
