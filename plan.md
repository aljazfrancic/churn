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
- [x] **4. Feature engineering & preprocessing** (`churn.features`, `churn.pipeline`)
  - [x] stratified 80/20 train/test split first, before anything is fitted (D26)
  - [x] `FeatureEngineer`: stateless; re-applies `clean()`, level collapse, new features (D17, D20, D21)
  - [x] `ColumnTransformer`: Yeo-Johnson numerics, one-hot categoricals, explicit column lists (D22, D23, D39)
  - [x] `build_pipeline(model, scaler=…, collapse_levels=…, new_features=…)`: features → preprocessing → model; the switches let the ablation (step 7) vary each step
  - [x] notebook: split, churn rate per new feature, skewness before/after scaling, model inputs
  - [x] unit tests: tenure 0 (no NaN/inf), tenure groups, level collapse, switches, input columns, raw records in, unseen categories (D9)
- [x] **5. Model training & tuning** (`churn.models`)
  - [x] metrics primer in the notebook: definitions, business meaning, no-skill values (D29, D30)
  - [x] candidates with search spaces; `RandomizedSearchCV`, 25 draws, 5-fold, PR-AUC (D24, D27)
  - [x] soft-voting ensemble of the 3 best; SVM probabilities via Platt scaling (D24, D40)
  - [x] unit tests: search spaces match their models, every tuned model gives probabilities (D9)
- [x] **6. Model comparison & selection** (`churn.evaluation`)
  - [x] repeated stratified 5-fold × 3 on train, same folds for every model: table incl. the tuning score (D26, D41)
  - [x] boxplots: PR-AUC, same-fold difference to the selected model, Brier score (D33, D41)
  - [x] `select_model()`: D28 as code with a complexity order + written justification → logistic regression (D28, D42)
  - [x] unit tests: selection rule, complexity order covers every candidate, same folds for every model (D9)
- [x] **7. Preprocessing ablation** (`churn.ablation`, D32, D44)
  - [x] cumulative variants: baseline (one-hot, unscaled) → + standardisation → + Yeo-Johnson → + level collapse → + new features (= final) → + class weights
  - [x] LR, XGBoost and SVM with their tuned settings on the 15 comparison folds (D44)
  - [x] table (inputs, PR-AUC, Brier per variant) + chart (same-fold PR-AUC change per step) + conclusion (D45)
  - [x] unit tests: one step per variant ending at the final setup, class weights reach every model, same-fold step effects, final variant = comparison scores (D9)
- [x] **8. Threshold tuning & final evaluation** (`churn.threshold`, `churn.evaluation`, D31, D47, D48)
  - [x] F2-optimal threshold: `TunedThresholdClassifierCV` on the 15 comparison folds, thresholds in steps of 0.01 (D31)
  - [x] hold-out test: threshold-free table (CV mean vs test) + decision table (0.5 vs F2 threshold), bootstrap 95 % CIs (D47, D48)
  - [x] confusion matrices at both thresholds: counts + % of the actual class (D30)
  - [x] ROC & PR curves (other tuned models in gray), metrics vs threshold, calibration (10 equal-size bins), cumulative gains (D33)
  - [x] unit tests: threshold = best mean F2 on the comparison folds, decision metrics match scikit-learn, bootstrap intervals, refitted copies (D9)
- [x] **9. Explainability** (`churn.explain`, D6, D49, D50)
  - [x] exact SHAP values of the final model for the test customers, in log-odds (D49)
  - [x] summary: mean |SHAP| per original feature + beeswarm of the strongest inputs; waterfall of the typical flagged churner (D33, D49)
  - [x] odds ratios vs the most common level, numerics per 1 SD, bootstrap 95 % intervals; forest plot (D50)
  - [x] unit tests: inputs map to their features, SHAP values add up to the log-odds, typical churner, odds-ratio contrasts (D9)
- [x] **10. Persistence & REST API** (`churn.serving`, D35, D51–D56)
  - [x] save the evaluated model with its F2 threshold (`FixedThresholdClassifier`) + metadata JSON to `models/` (D51–D54)
  - [x] `load_model()` checks the SHA-256 before unpickling (D52)
  - [x] `score()`: request body of raw CRM records → response body with risk, flag and model version (D55)
  - [x] notebook section 9: save, request & response of the reloaded model for three test customers, service design (endpoints, validation, deployment, monitoring, retraining) (D35, D55)
  - [x] unit tests: round trip, version, tampered file, raw records scored at the threshold, metadata from the hold-out tables (D9)
  - [x] `joblib` declared as a dependency (D56)
- [x] **11. README.md** – approach, metric results, usage (D57–D60)
  - [x] results: test metrics with 95 % intervals next to CV means and no-skill values; F2 threshold vs 0.5; strongest drivers (D57)
  - [x] approach: one item per notebook section, linking the notebook and `plan.md` as the decision log (D58, D59)
  - [x] usage: setup, notebook, checks, scoring snippet with the saved model; REST API summary, repository layout, data & licence (D57, D60)
- [ ] **12. Final review** – check the repo against `plan.md`, decisions and the assignment PDF (ask the user to provide the PDF again)
  - [ ] recompute every number quoted in notebook markdown, `plan.md` and `README.md`; flag any that no output shows
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
| D6 | **SHAP** for explainability; `ipywidgets` in the dev group | Global importance + per-customer explanations ("why is this customer at risk?") for marketing. shap imports `tqdm.auto`, which prints a warning in Jupyter unless the progress widget is installed. |
| D7 | Not used: Optuna, imbalanced-learn | Small search spaces → `RandomizedSearchCV` suffices; SMOTE on one-hot data is questionable and unnecessary with threshold tuning (D25, D31). |
| D56 | `joblib` declared as a runtime dependency | Imported directly (parallel loops, model files), so declared instead of relying on scikit-learn to install it. |

### Project & engineering

| ID | Decision | Why |
|---|---|---|
| D8 | Notebook `churn_prediction.ipynb` at repo root; logic in installable `src/churn/` package | Notebook is the first thing a reader sees and stays narrative; helper code lives in documented, testable modules. |
| D9 | pytest for deterministic code only | Cleaning, features and pipeline shapes are where silent bugs hide (e.g. division by tenure 0); model quality is checked by CV, not unit tests. |
| D10 | GitHub Actions: `uv sync --locked` → ruff → pytest → execute notebook (nbmake, cell timeout 30 min) | Proves the lockfile installs and the notebook runs end-to-end (incl. data download) on a clean machine. Model tuning takes ≈ 7 min on the GitHub runner, more than nbmake's default 5-min cell limit. |
| D11 | ruff for linting + formatting (incl. notebooks) | One fast tool instead of black + isort + flake8. |
| D12 | Data not committed; download helper (Kaggle public API → `data/`, gitignored); only the expected CSV is read from the zip, in memory; its SHA-256 is verified on every load | Licence is "© Original Authors" (redistribution unclear); no login needed; the checksum guarantees identical data (CSV, not zip, so re-zipping doesn't matter); reading one member in memory rules out malicious archive paths ("zip slip"). |
| D13 | `RANDOM_STATE = 42` everywhere; notebook committed with outputs | Reproducible results; plots visible directly on GitHub. |
| D14 | Repository text stays generic | Assignment PDF is internal: never committed (`.gitignore`), no organisation named. |
| D15 | Code comments reference decision IDs (e.g. `# D16`) | Links code to the reasoning here. |
| D36 | `churn/config.py` holds paths, data source, `RANDOM_STATE` and column groups | One place for constants shared by the notebook, modules and tests. |
| D43 | No `$` in notebook markdown: amounts written as USD (plots keep $) | Notebook renderers read the text between two `$` as LaTeX: GitHub's viewer broke the `AbsChargeDelta` line, JupyterLab 4 every paragraph with two amounts; escaping (`\$`) does not help in JupyterLab. |

### Data cleaning

| ID | Decision | Why |
|---|---|---|
| D16 | `TotalCharges` text → float; 11 blanks (all `tenure = 0`) → 0 | New customers not billed yet: 0 is the true value; keeps all rows. |
| D17 | "No internet service" / "No phone service" → "No" | Already encoded in `InternetService`/`PhoneService`; removes 14 redundant one-hot columns (7 features × 3 → 1, D23) → cleaner coefficients and SHAP plots. |
| D18 | Keep the 22 rows identical apart from `customerID` | All are tenure-1 basic plans with different IDs → plausible distinct customers. |
| D19 | Drop `customerID`; `SeniorCitizen` 0/1 → No/Yes; keep all other features (incl. `gender`) | ID carries no signal; consistent binary columns; gender shows no effect in EDA, its ~0 importance is itself a finding. |
| D37 | `clean()` returns a new frame, is idempotent and keeps `Churn` as Yes/No; `split_features_target()` returns X and y (1 = churn) | Safe to re-apply (notebook, tests, API input); readable labels in EDA plots; numeric target only where models need it. |

### Features & preprocessing

| ID | Decision | Why |
|---|---|---|
| D20 | New features: `AvgMonthlyCharges` = TotalCharges / tenure (MonthlyCharges if tenure = 0); `AbsChargeDelta` = \|MonthlyCharges − AvgMonthlyCharges\|; `NumAddons` (0–6 internet add-ons); `TenureGroup` (0–6, 7–12, 13–24, 25–48, 49+ months; open-ended for customers beyond 72) | Training split: bill changed by > $2 → 34 % churn vs 18 % stable (tenure > 1), increases and decreases alike (hence absolute; increase and decrease as two features: same CV PR-AUC, equal LR weights); 0 add-ons 52 % vs 6 add-ons 5 % (internet customers); tenure ≤ 6 m 52 % vs > 48 m 10 %. `AvgMonthlyCharges` alone ≈ `MonthlyCharges` (Pearson correlation r = 0.996; 1 = perfectly linear); kept as the suggested feature and as input for the delta. |
| D45 | Keep all four new features although the ablation shows no CV gain (LR +0.000, XGBoost −0.005, SVM −0.002 PR-AUC) | Required (≥ 1 new feature); neutral for the final model, while dropping them would mean re-tuning for nothing. Reported as a finding: once Yeo-Johnson fixes the skew, the raw columns already carry the signal. |
| D21 | Type fixes (`clean()`, idempotent), level collapse + feature engineering as the stateless first step of the sklearn `Pipeline` (`FeatureEngineer`) | Saved model accepts raw customer records as exported (blank `TotalCharges`, `SeniorCitizen` 0/1, ID) → no train/serve skew for the API; stateless → nothing to leak between folds; steps can be switched off for the ablation. |
| D22 | Numerics: Yeo-Johnson `PowerTransformer` (standardised) | Learns one exponent λ per feature: fixes skew where needed (`TotalCharges` 0.95 → −0.15, `AbsChargeDelta` 2.49 → 0.07) and leaves near-symmetric features almost unchanged. log1p is a fixed λ = 0: over-corrects most features to ≈ −0.8, under-corrects `AbsChargeDelta` (fitted λ = −0.39). Scales in the same step; needed for LR/SVM/MLP, harmless for trees. |
| D23 | Categoricals: one-hot (`drop="if_binary"`, `handle_unknown="ignore"`); dense DataFrame output with readable names (`Contract_Two year`) | Nominal categories; Yes/No becomes one 0/1 column; unseen categories at serving time are encoded as all zeros (with a warning) instead of crashing; names feed SHAP and coefficient plots. |
| D39 | `ColumnTransformer` gets explicit column lists from `config` (incl. the new features); every other input column is dropped | Fields a caller sends by mistake (an ID, even the target) can never reach the model; `build_pipeline` wires the lists to the switches in one place. |

### Modelling

| ID | Decision | Why |
|---|---|---|
| D24 | Candidates: dummy (prior), logistic regression, random forest, XGBoost, SVM (RBF), MLP; soft-voting ensemble (average probability) of the 3 best by CV PR-AUC | Linear, bagging, boosting, kernel and neural approaches plus a "mixture"; dummy anchors the comparison, LR is the interpretable reference. Stacking would need another nested CV and rarely helps on 7k rows. |
| D25 | No resampling or class weights in the final model | Keeps probabilities calibrated (a "63 % risk" really means ≈ 63 % in the CRM); imbalance handled by metrics (D29, D30) and threshold (D31); class weights tested in the ablation (D32). |
| D26 | Stratified 80/20 hold-out; tuning with stratified 5-fold CV on train; tuned models compared with repeated stratified 5-fold × 3 (new seed) | **Hold-out:** 20 % of customers (1,409) are set aside first (start of notebook section 3, before anything is fitted) and scored only once at the end → honest estimate for unseen customers.<br>**5-fold CV (tuning):** the training data is split into 5 parts; each hyperparameter setting is trained on 4 and validated on the 5th, rotating 5× → picks settings that generalise instead of fitting one lucky split.<br>**Repeated 5-fold × 3 (comparison):** tuned models are re-scored on 3 fresh shuffles (15 scores each); the tuning folds favour the settings chosen on them, new folds don't, and the spread shows whether model differences are real or noise.<br>**Stratified:** every split keeps the 26.5 % churn rate. Preprocessing is fitted on training folds only (inside the Pipeline) → no leakage. |
| D27 | `RandomizedSearchCV`, 25 draws per model on the same stratified 5-fold split, small documented search spaces (`churn.models.candidates`); searches use all cores, models run single-threaded | Cheap, no extra dependency, enough for small spaces (≈ 2 min on 12 cores); identical folds make the tuning scores comparable. |
| D46 | Logistic regression: `max_iter` = 10,000 | lbfgs converges in ≤ 94 iterations on scaled inputs, but needs up to ≈ 5,200 on the unscaled ablation baseline (D32); a solver budget, not a tuned setting, so no score changes and no convergence warning. |
| D40 | SVM tuned on its decision scores; the final SVM is wrapped in `CalibratedClassifierCV(SVC(), ensemble=False)` (Platt scaling) | Ranking metrics need no probabilities (≈ 4× faster tuning); `SVC(probability=True)` is deprecated in scikit-learn 1.9; the sigmoid is monotonic (ranking unchanged) and gives probabilities for soft voting, Brier score and the CRM. |
| D28 | Final model: best mean CV PR-AUC; prefer a simpler model if within 1 std | Differences on this data are small; simpler = explainable, calibrated, cheap to serve. |
| D41 | Comparison: every model on the same 15 folds (`RepeatedStratifiedKFold`, seed `RANDOM_STATE + 1`). Table: PR-AUC mean + std, tuning PR-AUC, ROC-AUC, Brier, fit time. Figure: PR-AUC, PR-AUC difference to the selected model on the same fold, Brier; the dummy only in the table | Some folds are harder for every model: the raw spread (std ≈ 0.02) hides consistent gaps that same-fold differences reveal (std 0.003–0.014); the tuning column shows how optimistic the tuning folds were (D26); the dummy's 0.265 would squash the axes. |
| D42 | `select_model()` applies D28 as code; complexity order: logistic regression < XGBoost < random forest < SVM < MLP < ensemble; the dummy is never selected | The choice follows a documented, tested rule instead of a judgement by eye. Ordered by explainability, calibration and serving cost: depth-2 XGBoost has fast exact TreeSHAP; the random forest has larger trees and less calibrated probabilities; the SVM needs Platt scaling and slow kernel SHAP; the MLP is a black box; the ensemble combines 3 models. |

### Evaluation

| ID | Decision | Why |
|---|---|---|
| D29 | Primary metric: **PR-AUC** (average precision) | Focuses on the churn class; no-skill baseline is the churn rate (0.265), not 0.5, so it reflects the imbalance. Threshold-free. Not recall: flagging everyone already gives 100 %, and any model reaches any recall via the threshold; PR-AUC averages precision over all recall levels, and recall gets its weight in the threshold (D31). |
| D30 | Also reported: ROC-AUC, Brier score, precision, recall, F1, F2, balanced accuracy, share flagged, confusion matrix (counts + % of the actual class); accuracy only as a cautionary contrast | Defined in the notebook with business meaning: low precision → incentives wasted on loyal customers (budget); low recall → churners missed (lost revenue). Always predicting "No" already gives 73.5 % accuracy. Row shares read directly as recall and as the share of loyal customers who get an offer. |
| D31 | Decision threshold maximises **F2**: `TunedThresholdClassifierCV` on the 15 comparison folds (D41), thresholds 0.01–0.99 in steps of 0.01; F2 = 0 when no one is flagged | Recall weighted 2× precision: a missed churner usually costs more than a wasted incentive. 15 folds give a steadier choice than 5; a round threshold is easy to quote in the CRM and is scored exactly on every fold; the zero avoids undefined-precision warnings above the highest probability. A metrics-vs-threshold plot shows other operating points. |
| D47 | Hold-out test: the final model at its F2 threshold and at 0.5; the other tuned models only as gray ROC & PR curves. Two tables: threshold-free metrics (CV mean vs test) and threshold metrics (0.5 vs F2) | Model, settings and threshold are fixed on the training data before the test set is touched, so the other models can only show whether the CV picture holds; 0.5 next to the F2 threshold shows what tuning buys (more churners reached, more offers sent). |
| D48 | Test uncertainty: bootstrap 95 % percentile intervals from 1,000 resamples of the 1,409 test customers, on fixed predictions and thresholds; decision metrics counted from the confusion matrix | With 374 churners, the test PR-AUC is only known to about ±0.05, so a gap to CV can be judged. No refits → seconds; counting by hand keeps the 2,000 evaluations per table fast (tested against scikit-learn). |
| D32 | Preprocessing effects via a cumulative ablation: baseline (one-hot, unscaled) → + standardisation → + Yeo-Johnson → + level collapse → + new features (= final) → + class weights (`class_weight="balanced"`; XGBoost: `scale_pos_weight` = stayed / churned); CV PR-AUC + Brier for LR, XGBoost and the SVM | Shows what each step actually contributes. Two scaling steps separate "same scale" from "less skew" (tests D22); XGBoost is the control (tree splits ignore monotonic transforms); the SVM is the most scale-sensitive candidate (its RBF kernel compares distances); Brier shows what class weights do to the probabilities (D25). |
| D44 | Ablation: each model keeps its tuned setting; every variant is scored on the 15 comparison folds (D41); a step's effect = same-fold PR-AUC change vs the previous variant. Table: inputs, PR-AUC, Brier; chart: PR-AUC change per step, own x-scale per model | Only the preprocessing changes. Re-tuning every variant would add ≈ 30 min to CI, and the SVM's gamma range cannot fit unscaled data anyway. The settings were tuned on the final setup, which slightly favours it; that variant reproduces section 5's scores. The unscaled SVM is so far behind that a shared axis would flatten every other step. |
| D33 | Evaluation plots: CV boxplots; on the test set ROC & PR curves (final model, other models in gray, threshold markers), confusion matrices, metrics vs threshold (precision, recall, F2, share flagged), calibration curve (10 equal-size bins), cumulative gains; SHAP summary (per feature + beeswarm) and waterfall, odds-ratio forest plot (D49, D50) | Covers model comparison, operating point, probability quality and explanations for marketing. Equal-size bins average ≈ 140 test customers each; most predictions are low, so equal-width bins would leave the high-risk ones nearly empty. |
| D34 | EDA plots: target distribution, churn rate per categorical, numeric distributions by churn, tenure × charges heatmap, churn vs tenure by contract, association heatmap; section 3: churn rate per new feature. One theme from a validated colour-blind-safe palette: churn = accent blue, retained = gray (emphasis), magnitudes = one-hue blue ramp, contract types = ordinal ramp; values labelled directly; retina PNGs ~11 in wide on an opaque light surface | Palette validator: churn vs gray CVD ΔE 15.9 (target ≥ 8), contrast ≥ 3:1. Static images have no hover, so labels carry the values; the width matches GitHub's notebook view. |
| D38 | Feature–churn association: Cramér's V (numeric features binned into quintiles) + chi² p-value; statistics in `churn/eda.py`, plotting in `churn/plots.py` | One 0–1 scale for categorical and numeric features; p-values show which differences could be chance; the notebook stays free of computation code. |

### Explainability

| ID | Decision | Why |
|---|---|---|
| D49 | SHAP: exact `LinearExplainer` on the final logistic regression, in log-odds, training set as background, for the test customers. Summary: mean \|SHAP\| per original feature (one-hot columns summed) + beeswarm of the 15 strongest inputs; waterfall: the flagged churner with the median risk; custom plots in the notebook's look | Exact and instant for a linear model, and the contributions add up to the prediction (contributions in probability would need an approximate explainer). Summing one-hot columns is exact and gives one number per feature; the beeswarm adds the direction; the median flagged churner is the typical case marketing would see. |
| D50 | Odds ratios: levels of multi-level features vs the most common level (exp of the coefficient difference); yes/no inputs Yes vs No; numeric inputs per +1 SD after Yeo-Johnson. Bootstrap 95 % intervals from 1,000 refits of the regression on resampled training customers, preprocessing fixed; forest plot on a log scale, blue where the interval excludes 1 | Every level of a multi-level feature is an input, so a single coefficient has no reference; per-SD effects are comparable across inputs. Refitting only the regression keeps every coefficient in the same units and takes ≈ 20 s on 12 cores (the whole pipeline: 76 s). Correlated inputs (tenure, charges, tenure groups, phone service) split one effect among them, so the notebook reads them as a group. |

### Deliverables

| ID | Decision | Why |
|---|---|---|
| D35 | REST API described, not implemented: a FastAPI service loads the saved model at start-up; `POST /v1/predictions` (one or many customers), validated by a pydantic schema of the raw fields, returns risk, flag and model version; `GET /v1/model`, `GET /health`; Docker image from `uv.lock`; monitoring + retraining. Model saved to `models/` (gitignored, recreated by the notebook) | Implementation not required; the saved model and `churn.serving.score` make the description concrete (D55). |
| D57 | README: results first (test metrics with 95 % intervals, CV means and no-skill values; threshold trade-off; strongest drivers), then approach, usage, REST API, repository layout, data & licence; every result copied from a notebook output | The requirements ask for the approach, the metric results and how to use everything; a reader who only skims sees the results; traceable numbers for the final review. |
| D58 | No figures in the README; it links the notebook | GitHub renders the notebook with every chart; text and tables cannot drift from it, and no image files are added. |
| D59 | The README links `plan.md` as the decision log | Shows the reasoning behind each choice and makes the Dx references in the code resolvable. |
| D60 | Scoring new customers: a Python snippet with `load_model()` and `score()` | Uses the tested serving code; no CLI to maintain. |

### Serving

| ID | Decision | Why |
|---|---|---|
| D51 | Saved model: the evaluated one (logistic regression trained on the training set, F2 threshold), not refitted on all 7,043 customers | Test metrics, threshold and explanations describe exactly this file, and the test set stays untouched for comparing future versions (champion/challenger). Refitting would add 25 % more data, but the saved file would never have been evaluated. |
| D52 | `churn_model.joblib`: scikit-learn's `FixedThresholdClassifier` (fitted pipeline + threshold); `churn_model.json`: metadata incl. the joblib file's SHA-256, checked before unpickling | `predict()` flags at the F2 threshold by construction (the bare pipeline's `predict()` would use 0.5); the metadata is readable without unpickling; unpickling runs code, so a corrupted or swapped file is rejected first (as D12). skops would avoid pickle, but adds a dependency and needs every custom class listed as trusted. |
| D53 | Model version: UTC save date + first 8 hex characters of the file's SHA-256 (e.g. `20261020-e148cbe7`) | Sorts by date, is readable and changes whenever the file does; the CRM stores it with every score (audit, rollback). |
| D54 | Metadata: version, creation time, SHA-256, model name, threshold; training data (customers, churn rate, data SHA-256); expected metrics (CV PR-AUC; test PR-AUC, ROC-AUC, Brier, and precision, recall and share flagged at the threshold, with 95 % intervals). No library versions or input fields | Lineage, and the baseline that monitoring compares live results with. Library versions are pinned by `uv.lock` (and scikit-learn warns on a mismatch); the input fields are in `churn.config` and the request schema. |
| D55 | `churn.serving.score`: request body `{"customers": [raw records]}` → response body `{"model_version", "threshold", "predictions": [{"customerID", "churn_probability", "flagged"}]}`; notebook demo with the reloaded file on three test customers (typical flagged churner of section 8.2, lowest-risk loyal customer, brand-new customer with a blank `TotalCharges`); per-customer reasons only described | The endpoint's exact request and response without a web framework; proves that the saved file scores raw CRM records (D21). Reasons (exact SHAP for the regression) would need the training means of the inputs saved as well: an extension. |
