import os
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.shipment_prep import FEATURES

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")

profile = (df.groupby("hub_id", as_index=False)
           .agg(shipments=("shipment_id", "count"),
                mean_delivery_hours=("delivery_hours", "mean"),
                delay_rate=("is_delayed", "mean"),
                mean_distance_km=("distance_km", "mean"),
                mean_n_stops=("n_stops", "mean"))
           .sort_values("hub_id"))
profile.round(4).to_csv("outputs/hub_profile.csv", index=False)

profile_cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
scaled = StandardScaler().fit_transform(profile[profile_cols])

sep_rows = []
clusterings = {}
for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(scaled)
    clusterings[k] = labels
    within, across = [], []
    for i, j in combinations(range(len(profile)), 2):
        d = float(np.linalg.norm(scaled[i] - scaled[j]))
        if labels[i] == labels[j]:
            within.append(d)
        else:
            across.append(d)
    ratio = np.mean(across) / np.mean(within)
    sizes = sorted(np.bincount(labels).tolist(), reverse=True)
    sep_rows.append([k, ratio, "-".join(map(str, sizes))])
pd.DataFrame(sep_rows, columns=["K", "across_over_within", "cluster_sizes"]).round(4).to_csv("outputs/part_d_separation.csv", index=False)

labels = clusterings[3]
cluster_delay = {c: profile.loc[labels == c, "delay_rate"].mean() for c in np.unique(labels)}
ordered = sorted(cluster_delay, key=cluster_delay.get)
name_map = {ordered[0]: "Standard", ordered[1]: "Watch", ordered[2]: "Escalate"}
seg = pd.DataFrame({"hub_id": profile["hub_id"], "segment": [name_map[x] for x in labels]})
seg = seg.merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")
seg = seg[["hub_id", "segment", "hub_city", "hub_tier"]]
seg.to_csv("outputs/hub_segments.csv", index=False)

segment_profile = (seg.merge(profile, on="hub_id")
                   .groupby("segment", as_index=False)
                   .agg(hubs=("hub_id", "count"),
                        mean_delivery_hours=("mean_delivery_hours", "mean"),
                        delay_rate=("delay_rate", "mean"),
                        mean_distance_km=("mean_distance_km", "mean"),
                        mean_n_stops=("mean_n_stops", "mean")))
segment_profile["_order"] = segment_profile["segment"].map({"Standard": 0, "Watch": 1, "Escalate": 2})
segment_profile = segment_profile.sort_values("_order").drop(columns="_order")
segment_profile.round(4).to_csv("outputs/segment_profile.csv", index=False)

expected_segment = {1: "Standard", 2: "Watch", 3: "Escalate"}
mismatch = seg[seg.apply(lambda r: r["segment"] != expected_segment[r["hub_tier"]], axis=1)]
mismatch.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)
