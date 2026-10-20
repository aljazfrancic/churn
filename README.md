# Telco Customer Churn Prediction

[![CI](https://github.com/aljazfrancic/churn/actions/workflows/ci.yml/badge.svg)](https://github.com/aljazfrancic/churn/actions/workflows/ci.yml)

Predicts which customers of a telecom provider are likely to leave within the next month, so that the sales & marketing team can offer them a retention incentive in time. Built on the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) data (IBM sample data: 7,043 customers, 26.5 % churn). The full analysis, with every chart, is in [`churn_prediction.ipynb`](churn_prediction.ipynb).

## Results

The final model, a **logistic regression**, scored once on 1,409 held-out test customers (20 % of the data, set aside before any model was fitted):

| Metric | Test | 95 % interval | CV mean | No skill |
|---|---:|---:|---:|---:|
| PR-AUC (primary) | 0.670 | 0.618–0.717 | 0.672 | 0.265 |
| ROC-AUC | 0.847 | 0.825–0.868 | 0.850 | 0.5 |
| Brier score (lower = better) | 0.134 | 0.124–0.144 | 0.133 | 0.195 |

- **It holds up on new customers:** every cross-validation mean lies inside its test interval, and the predicted risks are calibrated.
- **Decision threshold 0.19**, which maximises F2 (recall weighted 2× precision, as a missed churner costs more than a wasted offer): offers to 49 % of the customers reach 87 % of the churners, with a precision of 47 %. The default threshold of 0.5 would reach 52 % of them (precision 67 %, 21 % of customers flagged).
- **Strongest drivers:** contract type, internet service and tenure; e.g. a two-year contract cuts the odds of churning to a fifth of a month-to-month contract's (odds ratio 0.21).

## Key takeaways for sales & marketing

- **Who leaves:** month-to-month customers (43 % churn vs 3 % with a two-year contract), customers in their first six months (over half), fiber-optic customers (42 %), customers without online security or tech support (42 %) and those paying by electronic check (45 %). New customers with a bill of USD 90 or more churn most (77 %).
- **What to offer:** a longer contract is the strongest lever: other things equal, a one-year contract halves the odds of churning and a two-year contract cuts them to a fifth. Online security, tech support and automatic payment instead of an electronic check each go with roughly 25–30 % lower odds.
- **Whom to contact:** the customers the model flags (risk ≥ 0.19) are about half of all customers and include 87 % of those who would leave; 47 % of the offers reach one of them. On a smaller budget, start with the highest risks: the top 21 % include 52 % of those who leave, and 67 % of their offers reach one.
- **Read the risk as a probability:** of 10 customers with a 60 % risk, about 6 leave, so the risk can be weighed against a customer's value and the cost of an offer.
- **Measure before rolling out:** these are patterns in past data, not proven effects. Compare each offer against a control group that gets none.

## Approach

Each item is a section of the notebook; the logic lives in the tested package [`src/churn/`](src/churn/), and every design decision is recorded with its reason in [`plan.md`](plan.md) (numbered Dx and cited in the code).

1. **Data cleaning:** `TotalCharges` is text because of 11 blanks; all belong to brand-new customers (tenure 0) and become 0, their true amount; `SeniorCitizen` 0/1 becomes No/Yes; `customerID` is dropped.
2. **Exploratory analysis:** churn rate per feature with chi² tests and Cramér's V. Month-to-month contracts (43 % churn vs 3 % for two-year), short tenure, fiber optic, no security or support add-ons and payment by electronic check go with churn; gender doesn't, and phone service only matters among DSL customers.
3. **Features & preprocessing:** a stratified 80/20 train/test split comes first. Four new features: average monthly charges over the tenure, its gap to the current bill, number of add-ons and tenure groups. Numeric features are scaled with Yeo-Johnson, categorical ones one-hot encoded, all inside one scikit-learn pipeline that takes raw customer records.
4. **Training:** five models (logistic regression, random forest, XGBoost, SVM, MLP), each tuned with a randomized search over 25 settings and scored by stratified 5-fold cross-validation on the training set with PR-AUC, which reflects the class imbalance; plus a no-skill baseline and a soft-voting ensemble of the three best.
5. **Selection:** re-scored on 15 fresh folds (5-fold × 3), all tuned models are within 0.015 PR-AUC of each other. The rule prefers a simpler model within one standard deviation of the best; logistic regression is both the best (0.672) and the simplest, and its probabilities are the most accurate (Brier score 0.133, tied with the ensemble).
6. **Ablation:** adding the preprocessing steps one at a time shows that only scaling with skew correction clearly helps the models that need scaling; the new features are neutral, and class weights would distort the final model's probabilities.
7. **Threshold & test:** the F2-optimal threshold is chosen by cross-validation on the training set; then the model is scored once on the test set, with bootstrap 95 % intervals.
8. **Explanations:** exact SHAP values per customer and odds ratios with bootstrap intervals.
9. **Serving:** the model is saved with its threshold and metadata; the REST API is described in the notebook (summary below).

## Usage

Requires [uv](https://docs.astral.sh/uv/), which installs Python 3.13 and the locked dependencies. On macOS, XGBoost also needs the OpenMP runtime: `brew install libomp`.

```bash
git clone https://github.com/aljazfrancic/churn.git && cd churn
uv sync                                    # Python 3.13 + dependencies from uv.lock
uv run jupyter lab churn_prediction.ipynb  # explore the notebook
```

Re-running the whole notebook downloads the data from Kaggle (no login; checked by SHA-256) into `data/` and saves the model to `models/`, both git-ignored. It takes about 5 minutes on 12 CPU cores:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace churn_prediction.ipynb
```

Checks, also run by CI on every push (unit tests on Linux, macOS and Windows; lint and the full notebook on Linux):

```bash
uv run pytest                                        # unit tests
uv run ruff check . && uv run ruff format --check .  # lint and formatting
```

**Scoring new customers** with the saved model (after a notebook run), from records in the raw export format, e.g. in `uv run python`:

```python
import pandas as pd
from churn.serving import load_model, score

model, metadata = load_model()  # checks the file's SHA-256 before unpickling
customers = pd.read_csv("customers.csv")  # same columns as the Kaggle CSV
response = score(model, {"customers": customers.to_dict("records")}, metadata["version"])
# per customer: customerID, churn_probability and flagged (at the threshold)
```

## REST API

Described in section 9 of the notebook, not implemented: a FastAPI service loads the saved model at start-up and offers `POST /v1/predictions` (one or many raw customer records in; churn probability, flag and model version out), `GET /v1/model` and `GET /health`. Requests are validated with pydantic; the service runs in a Docker image built from `uv.lock`. Monitoring covers input drift, the share of customers flagged and, a month later, the realised precision and recall; retraining uses champion/challenger tests and a control group without offers.

## Repository layout

```
churn_prediction.ipynb    the analysis, with all outputs
src/churn/                the package behind the notebook
  config.py               paths, data source, random seed, column groups
  data.py                 download (SHA-256 check), loading, cleaning
  eda.py                  churn rates, Cramér's V, chi² tests, service mix
  features.py             level collapse and the new features
  pipeline.py             preprocessing and the full model pipeline
  models.py               candidate models, tuning, ensemble
  evaluation.py           model comparison, selection, test metrics with bootstrap intervals
  ablation.py             preprocessing ablation
  threshold.py            F2 threshold
  explain.py              SHAP values and odds ratios
  plots.py                every chart, in one style
  serving.py              saving and loading the model, scoring like the API
tests/                    unit tests (pytest)
plan.md                   requirements, steps and every design decision with its reason
pyproject.toml, uv.lock   project definition and locked dependencies (uv)
.github/workflows/ci.yml  unit tests on 3 OSes, lint and a full notebook run on every push
```

## Data & licence

The data is IBM's Telco Customer Churn sample, downloaded from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) on first use and not redistributed here. The code is licensed under [MIT](LICENSE).
