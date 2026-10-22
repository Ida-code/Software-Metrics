# RiskLens: scoring a team's repository

RiskLens classifies Python files into **Low, Medium, or High historical
change-proneness**. The class is based on revision-count terciles in the
training dataset. It is not a bug label, defect probability, or guarantee of
future changes.

## Train and save the scoring models

From this project folder, run:

```powershell
python master_analysis.py
```

The script saves:

- `risklens_model.joblib`: Random Forest using the nine model features.
- `risklens_static_model.joblib`: Random Forest using only complexity,
  size/function-count, and maintainability features for files with little
  history.

Both artifacts contain the fitted estimator, required feature order, and the
training revision-count cutoffs. **The cutoffs are the terciles calculated from
the training dataset; they are not recalculated from the team's repository.**
Predictions and High-class scores are therefore relative to the training
repositories and model, not to the team's own file distribution.

Only load model files created by a trusted run of `master_analysis.py`.

## Score a local repository

The target repository must be available locally and have Git history. Scoring
does not clone repositories or make network calls. For example:

```powershell
python score_repo.py --repo "C:\path\to\team-repo" --out risk_report.csv
```

To omit test code:

```powershell
python score_repo.py --repo "C:\path\to\team-repo" --out risk_report.csv --exclude-tests
```

The scorer includes current tracked files and untracked, non-ignored Python
files. It uses Lizard's per-function cyclomatic complexity and NLOC, Radon's
maintainability index, and PyDriller's local Git history. Author count is the
number of distinct commit author names; code age is measured from the first
touching commit to the scoring date. A revision is a commit that changes the
file. Coupling counts other files changed in the same commits and averages the
number of shared commits per partner. These history metrics cover the local
repository's available history.

Files touched by fewer than five commits use the static-only model and are
marked in the report. Each CSV row includes the predicted class, the model's
relative probability for the training **High** class, and the three largest
SHAP contributions to that class score. Positive contributions move the score
up; negative contributions move it down. These are model explanations, not
causal effects.

The dashboard's **Score a New Repo** tab reads `risk_report.csv` from this
folder by default; it can also load an uploaded report. To have the dashboard
find the report automatically, write it here:

```powershell
python score_repo.py --repo "C:\path\to\team-repo" --out risk_report.csv
```

Then run or refresh Streamlit:

```powershell
streamlit run dashboard.py
```

## Dependencies and local tests

The scorer uses the project's existing Python environment with pandas,
NumPy, scikit-learn, joblib, Lizard, Radon, PyDriller, and SHAP installed.
Run the focused tests with:

```powershell
python -m pytest risklens_tests
```

The tests create tiny local Git repositories in temporary directories; they do
not contact a remote service.
