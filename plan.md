# Plan – Telco Customer Churn Prediction

Predict which customers of a telecom provider are likely to churn (leave) within the next month, so the sales & marketing team can offer them retention incentives.

- **Data:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample data, 7043 customers × 21 columns)
- **Target:** `Churn` (Yes/No)

## Requirements

**Jupyter notebook (Python)** covering:

1. **Data cleaning & EDA** – find and fix data-type issues (e.g. `TotalCharges`); analyse the target distribution; show how categorical features (e.g. contract type) influence churn.
2. **Preprocessing & feature engineering** – create ≥ 1 new feature from existing data (e.g. average monthly charges over the customer's tenure); scale numeric features appropriately.
3. **Model training** – train a suitable classifier and justify the choice; use cross-validation to prevent overfitting.
4. **Model evaluation** – churn is an imbalanced class: choose metrics that reflect this; visualise the results.
5. **Code quality** – modular code (functions/classes, clear structure, readable); describe (no implementation needed) how the model would be exposed through a REST API for the team's CRM system.

**Repository:** the notebook; `uv` as package/project manager (`pyproject.toml` + `uv.lock`); `README.md` briefly describing the approach, the metric results and how to use everything.

## Steps

One step = one commit, pushed only after approval. Decision IDs (Dx) refer to the next section.

- [x] **0. Plan & repo hygiene** – `plan.md`, `.gitignore`
- [x] **1. Project scaffold**
  - [x] uv project: Python 3.13, `src/churn` package, runtime deps + dev group (D1–D6, D8)
  - [x] ruff config in `pyproject.toml` (D11)
  - [x] notebook skeleton `churn_prediction.ipynb` (section headings only)
- [x] **2. Data loading & cleaning** (`churn.config`, `churn.data`)
  - [x] download helper: Kaggle public API → `data/`, SHA-256 check, cached (D12, D36)
  - [x] `clean()`: `TotalCharges` → float, blanks → 0; drop `customerID`; `SeniorCitizen` → No/Yes (D16, D19, D37)
  - [x] unit tests (D9)
  - [x] GitHub Actions CI (D10)
  - [x] notebook: data overview, dtype issue, cleaning
- [x] **3. Exploratory data analysis** (`churn.eda`, `churn.plots`, D34, D38)
  - [x] target distribution (counts + %)
  - [x] churn rate per categorical feature + Cramér's V / chi² table
  - [x] numeric distributions by churn + churn-rate heatmap (tenure × monthly charges)
  - [x] churn vs tenure by contract type
  - [x] association heatmap
  - [x] short written findings
  - [x] unit tests for the statistics helpers (D9)
- [ ] **4. Feature engineering & preprocessing** (`churn.features`, `churn.pipeline`)
  - [ ] stateless transformer: level collapse + new features (D17, D20, D21)
  - [ ] `ColumnTransformer`: Yeo-Johnson numerics, one-hot categoricals (D22, D23)
  - [ ] `build_pipeline(model, …)` with switches for the ablation
  - [ ] unit tests (no NaN/inf at tenure 0, expected columns, fit/predict on a sample)
- [ ] **5. Model training & tuning** (`churn.models`)
  - [ ] stratified 80/20 hold-out split (D26)
  - [ ] metrics primer in the notebook: definitions + business meaning (D29, D30)
  - [ ] candidates with search spaces; `RandomizedSearchCV`, 5-fold, PR-AUC (D24, D27)
  - [ ] ensemble of the best models (D24)
- [ ] **6. Model comparison & selection** (`churn.evaluation`)
  - [ ] repeated stratified 5-fold × 3 on train: table + boxplots (D26)
  - [ ] pick the final model + written justification (D28)
- [ ] **7. Preprocessing ablation** (D32)
  - [ ] cumulative variants (scaling, level collapse, new features, class weights)
  - [ ] table + chart + conclusion
- [ ] **8. Threshold tuning & final evaluation**
  - [ ] F2-optimal threshold via CV on train (D31)
  - [ ] hold-out test metrics + confusion matrix (counts + %) (D30)
  - [ ] ROC & PR curves, metrics vs threshold, calibration, cumulative gains (D33)
- [ ] **9. Explainability** – SHAP summary + single-customer waterfall; LR odds ratios (D6, D33)
- [ ] **10. Persistence & REST API** – save pipeline + threshold + metadata to `models/`; API design section (D35)
- [ ] **11. README.md** – approach, metric results, usage
- [ ] **12. Final review** – check the repo against `plan.md`, decisions and the assignment PDF (ask the user to provide the PDF again)
- [ ] **13. Publish** – GitHub "About" description + topics, make the repository public

## Decisions & design

### Tooling

| ID | Decision | Why |
|---|---|---|
| D1 | `uv`, Python **3.13** | uv is required. All dependencies have 3.13 wheels; keeps TensorFlow possible later (no 3.14 wheel yet). |
| D2 | pandas, numpy, scikit-learn, matplotlib + **seaborn** | sklearn `Pipeline`/`ColumnTransformer` keep preprocessing inside CV folds (no leakage); seaborn gives nicer statistical plots with little code. |
| D3 | Static plots only (no plotly) | Interactive plotly output does not render in GitHub's notebook viewer. |
| D4 | **XGBoost** for gradient boosting; CPU-only `xgboost-cpu`, regular `xgboost` on macOS (environment markers) | Industry standard for tabular data, sklearn API. LightGBM is noisy and overfits small data by default; CatBoost is heavier. The default Linux wheel pulls a 305 MB NVIDIA NCCL library (multi-GPU only); the CPU build is 2–6 MB but has no macOS wheel (the regular macOS package is CPU-only anyway). |
| D5 | **sklearn `MLPClassifier`** as neural net | Same Pipeline/CV API, no extra dependency. TensorFlow is overkill for 7k rows (NNs rarely beat GBM/LR on small tabular data) and heavy. |
| D6 | **SHAP** for explainability | Global importance + per-customer explanations ("why is this customer at risk?") for marketing. |
| D7 | Not used: Optuna, imbalanced-learn | Small search spaces → `RandomizedSearchCV` suffices; SMOTE on one-hot data is questionable and unnecessary with threshold tuning (D25, D31). |

### Project & engineering

| ID | Decision | Why |
|---|---|---|
| D8 | Notebook `churn_prediction.ipynb` at repo root; logic in installable `src/churn/` package | Notebook is the first thing a reader sees and stays narrative; helper code lives in documented, testable modules. |
| D9 | pytest for deterministic code only | Cleaning, features and pipeline shapes are where silent bugs hide (e.g. division by tenure 0); model quality is checked by CV, not unit tests. |
| D10 | GitHub Actions: `uv sync --locked` → ruff → pytest → execute notebook (nbmake) | Proves the lockfile installs and the notebook runs end-to-end (incl. data download) on a clean machine. |
| D11 | ruff for linting + formatting (incl. notebooks) | One fast tool instead of black + isort + flake8. |
| D12 | Data not committed; download helper (Kaggle public API → `data/`, gitignored); only the expected CSV is read from the zip, in memory; its SHA-256 is verified on every load | Licence is "© Original Authors" (redistribution unclear); no login needed; the checksum guarantees identical data (CSV, not zip, so re-zipping doesn't matter); reading one member in memory rules out malicious archive paths ("zip slip"). |
| D13 | `RANDOM_STATE = 42` everywhere; notebook committed with outputs | Reproducible results; plots visible directly on GitHub. |
| D14 | Repository text stays generic | Assignment PDF is internal: never committed (`.gitignore`), no organisation named. |
| D15 | Code comments reference decision IDs (e.g. `# D16`) | Links code to the reasoning here. |
| D36 | `churn/config.py` holds paths, data source, `RANDOM_STATE` and column groups | One place for constants shared by the notebook, modules and tests. |

### Data cleaning

| ID | Decision | Why |
|---|---|---|
| D16 | `TotalCharges` text → float; 11 blanks (all `tenure = 0`) → 0 | New customers not billed yet: 0 is the true value; keeps all rows. |
| D17 | "No internet service" / "No phone service" → "No" | Already encoded in `InternetService`/`PhoneService`; removes 7 redundant one-hot columns → cleaner coefficients and SHAP plots. |
| D18 | Keep the 22 rows identical apart from `customerID` | All are tenure-1 basic plans with different IDs → plausible distinct customers. |
| D19 | Drop `customerID`; `SeniorCitizen` 0/1 → No/Yes; keep all other features (incl. `gender`) | ID carries no signal; consistent binary columns; gender shows no effect in EDA, its ~0 importance is itself a finding. |
| D37 | `clean()` returns a new frame, is idempotent and keeps `Churn` as Yes/No; `split_features_target()` returns X and y (1 = churn) | Safe to re-apply (notebook, tests, API input); readable labels in EDA plots; numeric target only where models need it. |

### Features & preprocessing

| ID | Decision | Why |
|---|---|---|
| D20 | New features: `AvgMonthlyCharges` = TotalCharges / tenure (MonthlyCharges if tenure = 0); `AbsChargeDelta` = \|MonthlyCharges − AvgMonthlyCharges\|; `NumAddons` (0–6 internet add-ons); `TenureGroup` (0–6, 7–12, 13–24, 25–48, 49–72 months) | Initial exploration: price changed → 34 % churn vs 23 % stable; 0 add-ons 52 % vs 6 add-ons 5 %; tenure ≤ 6 m 53 % vs > 48 m 10 %. `AvgMonthlyCharges` alone ≈ `MonthlyCharges` (r ≈ 1); kept as the suggested feature and as input for the delta. |
| D21 | Level collapse + feature engineering as a stateless first step of the sklearn `Pipeline` | Saved model accepts raw customer records (no train/serve skew for the API); steps can be switched off for the ablation. |
| D22 | Numerics: Yeo-Johnson `PowerTransformer` (standardised) | Fixes `TotalCharges` skew (0.96 → −0.15; log1p overcorrects to −0.82) and scales in one step; needed for LR/SVM/MLP, harmless for trees. |
| D23 | Categoricals: one-hot (`drop="if_binary"`, `handle_unknown="ignore"`) | Nominal categories; Yes/No becomes one 0/1 column; unseen categories at serving time don't crash. |

### Modelling

| ID | Decision | Why |
|---|---|---|
| D24 | Candidates: dummy (prior), logistic regression, random forest, XGBoost, SVM (RBF), MLP, ensemble (soft voting / stacking) of the best | Linear, bagging, boosting, kernel and neural approaches plus a "mixture"; dummy anchors the comparison, LR is the interpretable reference. |
| D25 | No resampling or class weights in the final model | Keeps probabilities calibrated (a "63 % risk" really means ≈ 63 % in the CRM); imbalance handled by metrics (D29, D30) and threshold (D31); class weights tested in the ablation (D32). |
| D26 | Stratified 80/20 hold-out; tuning with stratified 5-fold CV on train; tuned models compared with repeated stratified 5-fold × 3 (new seed) | **Hold-out:** 20 % of customers (1,409) are set aside before any modelling and scored only once at the end → honest estimate for unseen customers.<br>**5-fold CV (tuning):** the training data is split into 5 parts; each hyperparameter setting is trained on 4 and validated on the 5th, rotating 5× → picks settings that generalise instead of fitting one lucky split.<br>**Repeated 5-fold × 3 (comparison):** tuned models are re-scored on 3 fresh shuffles (15 scores each); the tuning folds favour the settings chosen on them, new folds don't, and the spread shows whether model differences are real or noise.<br>**Stratified:** every split keeps the 26.5 % churn rate. Preprocessing is fitted on training folds only (inside the Pipeline) → no leakage. |
| D27 | `RandomizedSearchCV`, ~20–30 draws per model, small documented search spaces | Cheap, no extra dependency, enough for small spaces. |
| D28 | Final model: best mean CV PR-AUC; prefer a simpler model if within 1 std | Differences on this data are small; simpler = explainable, calibrated, cheap to serve. |

### Evaluation

| ID | Decision | Why |
|---|---|---|
| D29 | Primary metric: **PR-AUC** (average precision) | Focuses on the churn class; no-skill baseline is the churn rate (0.265), not 0.5, so it reflects the imbalance. Threshold-free. |
| D30 | Also reported: ROC-AUC, Brier score, precision, recall, F1, F2, balanced accuracy, confusion matrix (counts + %); accuracy only as a cautionary contrast | Defined in the notebook with business meaning: low precision → incentives wasted on loyal customers (budget); low recall → churners missed (lost revenue). Always predicting "No" already gives 73.5 % accuracy. |
| D31 | Decision threshold maximises **F2** (CV on train, `TunedThresholdClassifierCV`) | Recall weighted 2× precision: a missed churner usually costs more than a wasted incentive; a metrics-vs-threshold plot shows other operating points. |
| D32 | Preprocessing effects via ablation: cumulative variants, CV PR-AUC for LR, XGBoost and a scale-sensitive model | Shows what cleaning, scaling, new features and class weights actually contribute. |
| D33 | Evaluation plots: CV boxplots, ROC & PR curves (all models), confusion matrix, metrics vs threshold, calibration curve, cumulative gains, SHAP summary + waterfall, LR odds ratios | Covers model comparison, operating point, probability quality and explanations for marketing. |
| D34 | EDA plots: target distribution, churn rate per categorical, numeric distributions by churn, tenure × charges heatmap, churn vs tenure by contract, association heatmap. One theme from a validated colour-blind-safe palette: churn = accent blue, retained = gray (emphasis), magnitudes = one-hue blue ramp, contract types = ordinal ramp; values labelled directly; retina PNGs ~11 in wide on an opaque light surface | Palette validator: churn vs gray CVD ΔE 15.9 (target ≥ 8), contrast ≥ 3:1. Static images have no hover, so labels carry the values; the width matches GitHub's notebook view. |
| D38 | Feature–churn association: Cramér's V (numeric features binned into quintiles) + chi² p-value; statistics in `churn/eda.py`, plotting in `churn/plots.py` | One 0–1 scale for categorical and numeric features; p-values show which differences could be chance; the notebook stays free of computation code. |

### Deliverables

| ID | Decision | Why |
|---|---|---|
| D35 | REST API described, not implemented: FastAPI service loads the joblib-saved pipeline (+ threshold, version); `POST /predict` (single + batch) with a pydantic schema of raw customer fields; returns probability, label, model version; Docker; monitoring + retraining. Pipeline saved to `models/` (gitignored, recreated by the notebook) | Implementation not required; the saved pipeline makes the description concrete. |
