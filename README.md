# ML4HL Responsible Artificial Intelligence Toolbox: Retrospective Opioid Use Disorder Risk Stratification

This project provides a teaching workflow for retrospective opioid use disorder (OUD) risk stratification with a calibrated classification model and Responsible Artificial Intelligence (RAI) tools. The dataset column `OD` records whether OUD was documented during the same 2-year source period used for the predictors. The workflow is not validated for prospective incident prediction, diagnosis or clinical deployment
- Goal: help students translate model outputs into explicit, auditable clinical-policy interpretations
- Notebook: `ML4HL_OD_RAI_toolbox.ipynb`
- Utilities: `utils.py` (AUC reporting, threshold selection policies, plots)
- Data: `Data/opiod_raw_data.csv` (1,000 rows, 20+ features)


**Table of Contents**
- Introduction
- Project Structure
- Install
- Quickstart: Run the Notebook
- Data Description
- Utilities: Evaluation, Thresholds, and Plots
- Optional: Responsible AI Dashboard
- Reproducibility and Notes
- Troubleshooting
- License and Acknowledgements


**Introduction**
- Purpose: analyse retrospective OUD risk scores, understand errors, evaluate performance and explore validation-only threshold policies based on workload, recall and illustrative harm points
- Methods: scikit‑learn pipeline (preprocessing + logistic regression), probability calibration, transparent model reporting (ROC/PR AUC), threshold trade‑offs, and Responsible AI (RAI) tooling for interpretability, error analysis, counterfactuals, and causal insights.
- Notebook origin: adapted from Microsoft’s Responsible AI toolkit examples with additional didactic commentary for healthcare use cases.


**Project Structure**
- `ML4HL_OD_RAI_toolbox.ipynb`: step‑by‑step walkthrough covering data cleaning, model development, calibration, threshold selection, and RAI dashboard configuration.
- `utils.py`: helper functions used by the notebook for evaluation, threshold policies, and visualizations.
- `Data/opiod_raw_data.csv`: sample dataset used by the notebook (1,000 rows).
- `environment.yml`: Conda environment for reproducibility, including the validated Python 3.10 scientific stack and RAI packages
- `LICENSE`: license for this repository.
- `README.md`: this file.


**Install**
- Prereqs: conda/mamba, Git, and a working Jupyter setup.
- Create the environment (recommended via conda/mamba):
  - `conda env create -f environment.yml`
  - `conda activate od_rai_mamba`
- Verify Jupyter + widgets:
  - `python -c "import IPython, ipywidgets; print('Jupyter OK')"`
  - If using JupyterLab and widgets don’t render, ensure Lab ≥ 3.x. No manual widget install is typically required with this env.


**Quickstart: Run the Notebook**
- Launch Jupyter:
  - `jupyter lab` (or `jupyter notebook`)
- Open `ML4HL_OD_RAI_toolbox.ipynb` and run cells top‑to‑bottom.
- What the notebook does:
  - Introduces clinical motivation, the learning objectives, and the three RAI themes explored: interpretability, counterfactual reasoning, and causal analysis.
  - Loads `Data/opiod_raw_data.csv`, performs schema alignment, and renames columns for code friendliness (e.g., `rx ds → rx_ds`, `SURG → Surgery`).
  - Summarizes the dataset (1,000 patient rows, 20+ features) and explains each attribute in a healthcare context.
  - Splits data into training, validation and test sets (70/15/15) with a fixed random seed, keeping test outcomes closed until final evaluation
  - Builds a preprocessing pipeline: median imputation + scaling for numeric variables and binary imputation for categorical/binary flags.
  - Establishes baseline discrimination (majority class vs unweighted logistic regression), then calibrates the logistic model using `CalibratedClassifierCV`.
  - Reports discrimination (ROC/PR AUC), calibration diagnostics, prevalence, and lift to ground discussions of model quality.
  - Keeps workload-only, recall-floor-only and harm-point sensitivity analyses separate on validation data
  - Freezes the threshold using the joint validation rule before one final test evaluation
  - Configures the Responsible AI dashboard (interpretability, error analysis, counterfactuals, causal inference) to inspect model behavior beyond global metrics.


**Data Description**
- Rows: patient‑level records.
- Target: `OD` (1 = recorded OUD diagnosis in the source 2-year period, 0 = no recorded diagnosis)
- Key predictors in the raw CSV:
  - `Low_inc`: low income flag.
  - `SURG`: surgery within 2 years (renamed to `Surgery` in notebook).
  - `rx ds`: cumulative opioid prescription days-supply filled during 2 years, renamed to `rx_ds`; it is not confirmed consumption or unique exposure days
  - `A .. V`: binary flags (e.g., infectious diseases, circulatory, respiratory, injuries, trauma, etc.).
- Example prevalence in the notebook: ~0.18 on validation/test.
- The dataset is synthetic and supports classroom analysis only. The source does not establish that all predictors preceded the OUD diagnosis


**Utilities: Evaluation, Thresholds, and Plots**
`utils.py` exposes small, composable helpers used in the notebook. Import them directly:

```python
from utils import (
    positive_scores, auc_report, tradeoff_table, wilson_interval,
    pick_threshold_cost, pick_threshold_recall_floor, pick_threshold_workload,
    summary_at_threshold,
    plot_recall_floor_curves, plot_cumulative_recall_at_threshold, plot_topk_at_threshold,
)
```

- Evaluation
  - `positive_scores(estimator, X)`: returns positive‑class scores for classifiers supporting `predict_proba` or `decision_function`.
  - `auc_report(y_true, y_score, name="model", plot=True)`: prints ROC/PR AUC, prevalence, and lift; plots ROC/PR curves.
- Threshold trade‑offs
  - `tradeoff_table(y_true, y_score, thresholds=None)`: precision, recall, confusion counts, alerts/1k, and TP/1k across 0, each distinct score, and 1 when thresholds are omitted
  - `pick_threshold_workload(y_true, y_score, alerts_per_1000_max)`: best TP/1k under an alert budget (returns summary + table).
  - `pick_threshold_recall_floor(y_true, y_score, recall_floor)`: max precision subject to minimum recall (returns summary + table).
  - `pick_threshold_cost(y_true, y_score, C_FP, C_FN)`: minimizes expected cost (Bayes formula vs empirical minimum).
- Visualizations at a threshold
  - `summary_at_threshold(y_true, y_score, thr)`: one‑row summary at a specific threshold.
  - `plot_recall_floor_curves(...)`: precision/recall vs threshold with chosen recall floor and threshold highlighted.
  - `plot_cumulative_recall_at_threshold(...)`: cumulative capture vs number of alerts with vertical line at the implied alerts.
  - `plot_topk_at_threshold(...)`: bar chart of top‑K highest‑risk patients, coloring TPs/FPs with threshold line.

Minimal example (outside the notebook):

```python
# Given a fitted sklearn classifier `clf` and arrays y_val, X_val
y_score = positive_scores(clf, X_val)
auc_report(y_val, y_score, name="My Model", plot=True)

# Choose threshold under an illustrative alert budget of 300 per 1,000
res = pick_threshold_workload(y_val, y_score, alerts_per_1000_max=300.0)
print(res["summary"])   # chosen threshold and metrics

# Visualize at the chosen threshold
thr = res["threshold"]
plot_recall_floor_curves(y_val, y_score, recall_floor=0.30, chosen_threshold=thr)
plot_cumulative_recall_at_threshold(y_val, y_score, chosen_threshold=thr)
plot_topk_at_threshold(y_val, y_score, chosen_threshold=thr, top_k=30)
```


**Responsible AI Dashboard**
The environment includes `responsibleai` and `raiwidgets`. The notebook imports them; you can optionally create a dashboard for error analysis and explanations. Example pattern:

```python
from responsibleai import RAIInsights
from raiwidgets import ResponsibleAIDashboard

# X_train, y_train, X_test, y_test, and a fitted `calibrated_clf` exist from the notebook
features = X_train.columns.tolist()
rai = RAIInsights(
    model=calibrated_clf,
    train=X_train, test=X_test,
    target_column="OD",
    task_type="classification",
    categorical_features=[c for c in features if X_train[c].nunique() <= 10],
)

rai.explainer.add()
rai.error_analysis.add()
rai.compute()
ResponsibleAIDashboard(rai)
```

Notes:
- Dashboard components:
  - **Interpretability**: global feature importance summaries (e.g., opioid prescription days, income status) that explain why the classifier flags patients.
  - **Error Analysis**: heatmaps and decision trees that highlight segments (such as surgery patients) where the model underperforms.
  - **Counterfactuals**: individual patient “what‑if” scenarios (e.g., fewer opioid days) to explore actionable interventions.
  - **Causal Inference**: uplift estimates to reason about policy changes and their potential impact on OD incidence.
- Usage tips:
  - If the widget does not render, trust the notebook (File → Trust Notebook) and prefer JupyterLab ≥ 3.x.
  - The dashboard can be heavy; run after the core analysis completes and save outputs for later review in class discussions.


**Reproducibility and Notes**
- Random seed: the notebook sets `RANDOM_STATE = 42` for splits and modeling.
- Calibration: uses `CalibratedClassifierCV` over a logistic baseline pipeline (with imputation, scaling, and variance filtering).
- Recalibration: compares average raw and calibrated probabilities with observed validation prevalence; it does not guarantee an individual outcome
- Threshold selection: validation recall must be at least 60% and alerts must not exceed 300 per 1,000; the feasible threshold with highest precision is locked before test evaluation
- Executed split: 700 training records with 127 recorded OUD cases, 150 validation records with 27 cases, and 150 final test records with 27 cases
- Calibration-in-the-large on validation: observed prevalence 0.180, mean raw predicted probability 0.186, mean calibrated predicted probability 0.185, absolute difference 0.005
- Executed validation result: threshold 0.219932; recall 0.667 (18 of 27, 95% confidence interval 0.478 to 0.814), precision 0.439 (18 of 41, 95% confidence interval 0.299 to 0.590), and 273.3 alerts per 1,000
- Final locked test result from 150 synthetic records: recall 0.741 (20 of 27, 95% confidence interval 0.553 to 0.868), precision 0.370 (20 of 54, 95% confidence interval 0.254 to 0.504), 360 alerts per 1,000, precision-recall area under the curve 0.405 and receiver operating characteristic area under the curve 0.731
- Data stewardship: this is a synthetic teaching dataset. In clinical settings, ensure governance, privacy, bias auditing, and alignment with institutional review processes before deployment.


**Troubleshooting**
- Widget/dashboard not showing:
  - Trust the notebook. Try JupyterLab instead of classic Notebook.
  - Ensure the conda env is active where Jupyter runs (`which jupyter`).
- Import errors (e.g., `fairlearn`, `responsibleai`):
  - Recreate the environment: `conda env remove -n od_rai_mamba && conda env create -f environment.yml`
- Plots not appearing:
  - Ensure cells aren’t in skipped state and that Matplotlib backend is interactive (`%matplotlib inline` or default in Jupyter).


**License and Acknowledgements**
- License: see `LICENSE`.
- Based on/reference: Microsoft Responsible AI toolbox notebooks and standard scikit‑learn documentation.
