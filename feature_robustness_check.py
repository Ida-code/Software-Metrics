"""
Feature robustness check: which features associate with historical file changes?

The Random Forest's built-in importance can be misleading (it is biased when
features are correlated with each other, and it is measured on the training data).
This script tests the same question in four independent ways:

  1. Plain correlation of every feature with the revision count (no model at all)
  2. Permutation importance on files the model has NOT seen (held-out test set)
  3. Drop-one-feature test: how much does cross-validated F1 fall when a feature is removed?
  4. Feature-group test: static code metrics only vs adding code age vs adding authors

HOW TO USE (from the folder that contains full_merged_dataset.csv):
    python feature_robustness_check.py
This also writes feature_robustness_results.json for the Streamlit dashboard.
"""

import warnings
warnings.filterwarnings("ignore")

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURES = ["avg_complexity", "max_complexity", "total_nloc", "function_count",
            "maintainability_index", "num_authors", "code_age_days",
            "coupling_partner_count", "avg_coupling_degree"]
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR / "full_merged_dataset.csv"
RESULTS_PATH = SCRIPT_DIR / "feature_robustness_results.json"

NAMES = {
    "avg_complexity": "Average Complexity",
    "max_complexity": "Max Complexity",
    "total_nloc": "Total Lines of Code",
    "function_count": "Number of Functions",
    "maintainability_index": "Maintainability Index",
    "num_authors": "Number of Authors",
    "code_age_days": "Code Age (days)",
    "coupling_partner_count": "Coupling Partner Count",
    "avg_coupling_degree": "Avg Coupling Degree",
}

# ---------- Load data and rebuild the same historical-change label as master_analysis.py ----------

df = pd.read_csv(DATASET_PATH)
for col in ["coupling_partner_count", "avg_coupling_degree"]:
    if col not in df.columns:
        df[col] = 0.0
df = df.dropna(subset=FEATURES + ["n-revs"]).copy()

q1 = df["n-revs"].quantile(0.33)
q2 = df["n-revs"].quantile(0.66)
df["risk_label"] = df["n-revs"].apply(lambda r: "Low" if r <= q1 else ("Medium" if r <= q2 else "High"))
y = df["risk_label"]
print(
    f"Files: {len(df)}   Historical change cut-offs: "
    f"Low <= {q1:.1f}, Medium <= {q2:.1f}, High above\n"
)


def make_model():
    return RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced")


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def cv_f1(cols):
    return cross_val_score(make_model(), df[cols], y, cv=skf, scoring="f1_macro").mean()


# ---------- CHECK 1: plain correlation with revisions, no model involved ----------

print("=" * 72)
print("CHECK 1: Spearman correlation of each feature with revision count (no model)")
print("=" * 72)
rows = []
for f in FEATURES:
    r, p = spearmanr(df[f], df["n-revs"])
    rows.append((NAMES[f], r, p))
corr_df = pd.DataFrame(rows, columns=["Feature", "Spearman_r", "p_value"])
corr_df = corr_df.reindex(corr_df["Spearman_r"].abs().sort_values(ascending=False).index)
print(corr_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
correlation_results = [
    {
        "feature": next(key for key, value in NAMES.items() if value == row.Feature),
        "label": row.Feature,
        "spearman_r": float(row.Spearman_r),
        "p_value": float(row.p_value),
    }
    for row in corr_df.itertuples(index=False)
]

# ---------- CHECK 2: permutation importance on UNSEEN files ----------

print("\n" + "=" * 72)
print("CHECK 2: Permutation importance on held-out files (drop in F1 when a feature is shuffled)")
print("=" * 72)
X_train, X_test, y_train, y_test = train_test_split(
    df[FEATURES], y, test_size=0.25, random_state=42, stratify=y
)
model = make_model().fit(X_train, y_train)
perm = permutation_importance(model, X_test, y_test, scoring="f1_macro",
                              n_repeats=30, random_state=42)
perm_df = pd.DataFrame({
    "Feature": [NAMES[f] for f in FEATURES],
    "F1_drop_mean": perm.importances_mean,
    "F1_drop_std": perm.importances_std,
}).sort_values("F1_drop_mean", ascending=False)
print(perm_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("A feature whose drop is no bigger than its std is not reliably important.")

plt.figure(figsize=(8, 5))
order = perm_df.sort_values("F1_drop_mean")
plt.barh(order["Feature"], order["F1_drop_mean"], xerr=order["F1_drop_std"], color="#2E74B5")
plt.xlabel("Drop in F1-macro when the feature is shuffled (held-out files)")
plt.title("Permutation importance on unseen files")
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "permutation_importance.png", dpi=150)
permutation_results = [
    {
        "feature": feature,
        "label": NAMES[feature],
        "f1_drop_mean": float(perm.importances_mean[FEATURES.index(feature)]),
        "f1_drop_std": float(perm.importances_std[FEATURES.index(feature)]),
        "reliably_above_noise": bool(
            perm.importances_mean[FEATURES.index(feature)]
            > perm.importances_std[FEATURES.index(feature)]
        ),
    }
    for feature in FEATURES
]

# ---------- CHECK 3: remove one feature at a time ----------

print("\n" + "=" * 72)
print("CHECK 3: Cross-validated F1 when ONE feature is removed")
print("=" * 72)
base = cv_f1(FEATURES)
print(f"All 9 features: F1-macro = {base:.3f}\n")
drop_rows = []
for f in FEATURES:
    remaining = [c for c in FEATURES if c != f]
    score = cv_f1(remaining)
    drop_rows.append((NAMES[f], score, score - base))
drop_df = pd.DataFrame(drop_rows, columns=["Removed feature", "F1_without_it", "Change_vs_all"])
drop_df = drop_df.sort_values("Change_vs_all")
print(drop_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("A feature can matter yet show a small drop if other features carry the same information.")
drop_results = [
    {
        "feature": next(key for key, value in NAMES.items() if value == f),
        "label": f,
        "f1_without_feature": float(score),
        "change_vs_all": float(change),
    }
    for f, score, change in drop_rows
]

# ---------- CHECK 4: groups of features ----------

print("\n" + "=" * 72)
print("CHECK 4: What does each kind of information add?")
print("=" * 72)
static = ["avg_complexity", "max_complexity", "total_nloc", "function_count", "maintainability_index"]
groups = [
    ("Static code metrics only (complexity, size, maintainability)", static),
    ("Static + code age", static + ["code_age_days"]),
    ("Static + number of authors", static + ["num_authors"]),
    ("Static + authors + code age", static + ["num_authors", "code_age_days"]),
    ("All 9 features", FEATURES),
]
group_rows = []
for label, cols in groups:
    score = cv_f1(cols)
    group_rows.append({"label": label, "features": cols, "f1_macro": float(score)})
    print(f"{label:<62} F1-macro = {score:.3f}")

# ---------- Context: how tied together are authors, size, age and revisions? ----------

print("\n" + "=" * 72)
print("CONTEXT: Spearman correlation of Number of Authors with other columns")
print("=" * 72)
author_context = []
for other in ["n-revs", "total_nloc", "code_age_days", "function_count", "max_complexity"]:
    r, p = spearmanr(df["num_authors"], df[other])
    author_context.append({
        "feature": other,
        "label": "Revision count" if other == "n-revs" else NAMES.get(other, other),
        "spearman_r": float(r),
        "p_value": float(p),
    })
    print(f"Number of Authors vs {other:<16} r = {r:.3f}")
print("\nIf Authors vs n-revs is very high, authors and revisions are two views of the same")
print("commit history, and 'authors predicts revisions' is partly built into the data.")

results = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "dataset": DATASET_PATH.name,
    "rows": int(len(df)),
    "risk_cutoffs": {"low_max": float(q1), "medium_max": float(q2)},
    "cross_validated_f1_all_features": float(base),
    "correlations": correlation_results,
    "permutation_importance": permutation_results,
    "drop_one_feature": drop_results,
    "feature_groups": group_rows,
    "author_context": author_context,
    "permutation_test_size": 0.25,
    "permutation_repeats": 30,
    "cross_validation_folds": 5,
}
RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\nSaved chart: {SCRIPT_DIR / 'permutation_importance.png'}")
print(f"Saved dashboard results: {RESULTS_PATH}")