# Delhivery Delay Intelligence Capstone

This repository implements one connected, leakage-safe pipeline for delivery-time prediction, proactive delay alerts, and hub operating segments. The data is synthetic and is generated entirely by the committed `generate_data.py` script.

## How to run

From a fresh clone, run these commands from the repository root, in this order:

```bash
pip install -r requirements.txt
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

`generate_data.py` must be committed verbatim and regenerates the three source CSVs in `data/`. The Part A script creates `data/modelling_table.csv`; Parts B, C and D consume that table. `package_model.py` regenerates `models/delay_model.pkl` from the final Part C configuration and also writes the threshold, feature order and imputation files. The committed pickle is reliably loadable with the pinned scikit-learn version in `requirements.txt`; if the installed version differs, the grader should re-run `python package_model.py` rather than rely on the committed pickle.

All CSV outputs are written with `index=False`, and generated numeric output values are rounded to four decimals where applicable.

## Data findings

- **Service type:** Express has the highest delay rate at **0.4897** (119 delayed parcels out of 243), versus 0.2056 for Standard and 0.0857 for Economy. This is the clearest service-level delay hotspot.
- **Hub tier:** Delay rate rises sharply with tier: **Tier 1 = 0.0400**, **Tier 2 = 0.2150**, and **Tier 3 = 0.4400**. The observed operating behaviour therefore broadly worsens as the nominal tier increases.
- **Distance:** Delay rate increases monotonically across the distance bands: **0.0219** for 0-200 km, **0.1368** for 201-400 km, **0.4364** for 401-600 km, and **0.8523** for 601+ km.
- **Strongest correlations with delivery hours:** excluding `delivery_hours` itself, the three largest absolute correlations are `promised_hours` (**0.8372**), `distance_km` (**0.4013**), and `n_stops` (**0.3247**). These are signed correlations from `outputs/part_a_correlations.csv`.
- **Declared value transformation:** the median-filled `declared_value_inr` has mean **1,533.0542 Rupees** and median **1,121.5000 Rupees**, showing right skew. Taking the natural log reduces the influence of large values: the corresponding mean is **7.0277** and median **7.0224**.

## Model results (regression)

The regression target is `delivery_hours`, using the pinned 17-feature order.

- The mean baseline has validation R² of **-0.0212**. A negative R² means the constant mean predictor is worse than using the validation-set mean as a reference; it provides essentially no useful parcel-level discrimination.
- The validation winner is **linear regression**, with validation RMSE **4.8912 hours** and R² **0.9313**. Gradient boosting is extremely close at RMSE **4.9165** and R² **0.9306**.
- The shallow tree is high-bias: depth 1 has train RMSE **11.6334** and validation RMSE **12.1324**, a small gap but both errors are high. Deeper trees reduce training error, while the validation error bottoms out at depth **6** (RMSE **6.2796**) and then rises. The unlimited tree has train RMSE **0.0000** versus validation RMSE **7.8368**, a gap of **7.8368 hours**, which is the high-variance pattern.
- For gradient boosting, learning rate **0.1** is best with validation R² **0.9306**. At **0.01**, R² is only **0.7127**, indicating that 100 estimators are too slow to learn enough signal; at **1.0**, R² falls to **0.8440**, consistent with an overly aggressive/overshooting step size.
- The validation ranking puts linear regression first, while the shared 5-fold cross-validation on the 1,020 training-plus-validation rows puts gradient boosting first: mean R² **0.9353** versus **0.9261** for linear regression. Both CV standard deviations are small (**0.0045** and **0.0048**). For this repository I would ship **linear regression**, because the stated final-model rule selects the lowest validation RMSE, linear regression wins that pinned validation comparison, and its simpler form is easier to audit and maintain while the CV difference is modest.
- The final linear-regression model scores **0.9493 R²** on the untouched test set, with RMSE **4.5235 hours**.

## Model results (classification)

The classifier predicts whether `delivery_hours` will exceed the service promise.

- The blind validation baseline predicts every parcel as on-time: accuracy **0.7667**, with TN **138**, FP **0**, FN **42**, TP **0**. It looks acceptable on accuracy only because roughly three quarters of validation parcels are not delayed; it catches no delayed parcels and therefore cannot support proactive operations.
- The five bookkeeping/noise columns have a combined random-forest importance of **0.2245** in the 17-feature model. All five are outside the top seven by ANOVA F-score, and each has an F-score below 10. The top seven are exactly `distance_km`, `hub_tier`, `hub_daily_capacity`, `n_stops`, `promised_hours`, `rain_flag`, and `sale_event_flag`.
- The SelectKBest table shows why seven features were retained: validation F1 is **0.6988** at k=5, **0.8333** at k=7, **0.7857** at k=10, **0.8049** at k=12, and **0.7848** at k=17. Adding the weaker/noise features does not improve the validation result.
- Hyperparameter tuning did **not** beat the untuned forest on the pinned 180-row validation set. Grid search has validation F1 **0.7750**, randomized search **0.7848**, while the untuned forest on the same seven selected features reaches **0.8333**. This is a useful warning about tuning against a small validation sample: a search can optimize cross-validation accuracy without producing the best held-out validation F1.
- The operations threshold is **0.5**. At this threshold the validation precision is **0.7955** and recall is **0.8333**. It is the highest tested threshold that still reaches the required recall of at least 0.80, so it trades fewer unnecessary calls for still catching at least four-fifths of delayed parcels.
- On the untouched test set, precision is **0.6889**, recall is **0.7561**, and F1 is **0.7209**. Recall is lower than the validation value by about 0.0772, which is plausible with only **41 delayed parcels** in the test set; the test result should not be interpreted as a precise estimate of long-run recall.

## Operations recommendation

### Hub playbooks

The **Escalate** segment contains nine hubs. Their hub-level delay rates are:

- **HUB13 Nagpur — 0.4600**
- **HUB15 Kochi — 0.3600**
- **HUB17 Patna — 0.4600**
- **HUB18 Ranchi — 0.3800**
- **HUB19 Guwahati — 0.6200**
- **HUB20 Raipur — 0.4800**
- **HUB21 Bhubaneswar — 0.4000**
- **HUB23 Siliguri — 0.4400**
- **HUB24 Agartala — 0.5400**

The four tier-versus-segment mismatches are:

- **HUB07 Kolkata — Tier 1, Watch**: its nominal Tier 1 label understates the operating risk; use the Watch playbook.
- **HUB13 Nagpur — Tier 2, Escalate**: the hub behaves like the highest-risk segment; use the Escalate playbook rather than relying on the Tier 2 label.
- **HUB15 Kochi — Tier 2, Escalate**: likewise, its observed behaviour is materially worse than the nominal tier suggests.
- **HUB22 Dehradun — Tier 3, Watch**: its observed behaviour is better than a typical Tier 3 hub; use Watch for operations, while retaining the tier as a separate capacity/structural label.

The alert threshold is **0.5**. On the final test table, **6.9 of every 10 flagged parcels are actually delayed** (precision **0.6889**), and about **7.6 of every 10 delayed parcels are caught** (recall **0.7561**). The test set is small, so these ratios are operational indicators rather than precise long-run estimates.

For the 48 new parcels, **8 were flagged** and **6 of those 8 require a hub-manager escalation**; the other two flagged parcels receive proactive customer calls.

### Why keep K = 3?

K = 3 gives a strong across-to-within distance ratio of **5.3928** with cluster sizes **9-8-7**. The ratio rises to **6.3047** at K=4 and **6.8866** at K=5, but those solutions create a cluster of only **4 hubs** and **2 hubs**, respectively. Three playbooks are fixed by the business, and K=3 produces three substantial operating groups rather than tiny specialist clusters that would be difficult to staff and manage as independent playbooks.

## Production plan

- **Accuracy:** `outputs/part_b_metrics.csv`, `outputs/part_c_final_test.csv`, and the Part A delay tables provide explicit regression and classification performance checks instead of relying on a single headline metric.
- **Reliability:** `outputs/part_d_reload_check.csv` verifies that the saved and reloaded classifier gives a maximum absolute probability difference of **0.0** on the test rows.
- **Reproducibility:** `generate_data.py`, pinned `requirements.txt`, fixed random seeds, fixed feature order, and deterministic output scripts make the pipeline reproducible from a clean clone.
- **Maintainability:** common cleaning and feature engineering live once in `src/shipment_prep.py`; the prediction script reuses those functions instead of duplicating preprocessing.
- **Consistency:** `models/feature_order.csv` fixes the exact 17 input columns and `models/impute_values.csv` carries the training-time medians into prediction-time cleaning.
- **Monitorability:** the repository writes structured CSV diagnostics for data quality, model performance, feature signal, thresholds, hub profiles, segmentation, and model reload behaviour.
- **Performance/speed:** the final classifier is a seven-feature random forest with a small fixed forest size, so the online scoring path only cleans 48-row batches, merges hub metadata, selects seven features, and evaluates one saved model.

Three core production monitors and example numeric triggers are:

1. **Weekly delayed share among parcels whose outcomes are known:** alert if it moves more than **5 percentage points** above the comparable baseline for four consecutive weeks.
2. **Weekly flagged share:** investigate if the share of parcels flagged by the 0.5 threshold rises above **20%** or falls below **10%** for two consecutive weeks, because either shift can indicate prevalence or score-distribution drift.
3. **Observed recall on matured parcels:** trigger model review if rolling four-week recall falls below **0.70**.
4. **Feature drift:** trigger investigation if the PSI-like distribution comparison for any core feature exceeds **0.20**; distance, n_stops, rain_flag, sale_event_flag and promised_hours should be checked first because they carry substantial predictive signal.

Retraining should run **monthly**, with an additional review after major network/process changes. The **15–28 September sale window** should be evaluated against comparable sale-event periods, not ordinary weeks: `sale_event_flag` is an explicit model feature, so the correct comparison is sale-to-sale performance and drift. Data drift here would look like changed distributions of distance, stops, service mix, rain/sale frequency, hub behaviour, or model score probabilities relative to the training period, followed by degraded recall/precision on parcels whose outcomes have matured.

## Limitations

1. **Synthetic data:** the generator is deliberately artificial and does not represent real Delhivery shipment records, customer behaviour, hub processes, or operational costs. External validity therefore has to be demonstrated on real data before deployment.
2. **Small test positives:** the test set contains only **41 delayed parcels**, so precision and recall can move noticeably when a small number of parcels change classification.
3. **Single pinned split for the operational threshold:** the 0.5 alert threshold is selected from one 180-row validation set. A production threshold should be confirmed with larger rolling time-based validation and an explicit cost matrix.
4. **No time-aware holdout:** the supplied modelling workflow uses the required random split rather than a temporal backtest. Production evaluation should add forward-looking validation to test whether performance survives changing traffic and hub conditions.
5. **Bookkeeping noise can change in production:** the synthetic generator intentionally includes five bookkeeping columns as distractors. Real systems may contain additional identifiers or process fields whose leakage risk needs separate review.
