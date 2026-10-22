"""
MASTER ANALYSIS SCRIPT - Explainable Historical Change-Proneness Analysis

This script:
1. Loads complexity, revisions, churn, maintainability, and ownership/age data
   from FIVE repos (requests, flask, click, httpx, rich)
2. Cleans and merges them all into one big dataset
3. Labels historical file change-proneness as Low / Medium / High using revision
   count quantiles. These are historical change categories, not bug labels.
4. Trains a Random Forest classifier with cross-validation
5. Explains predictions using SHAP
6. Compares this new model against your ORIGINAL simple linear regression baseline

HOW TO USE:
1. Place this script directly inside:
   the main folder that contains the requests, flask, click, httpx, and rich subfolders
2. Install any missing packages once:
   pip install pandas numpy scipy scikit-learn shap matplotlib
3. Run:
   python master_analysis.py
"""

import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPOS = ["requests", "flask", "click", "httpx", "rich"]
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = SCRIPT_DIR / "master_results.json"
GENERALIZATION_RESULTS_PATH = SCRIPT_DIR / "generalization_results.json"
SCORING_MODEL_PATH = SCRIPT_DIR / "risklens_model.joblib"
STATIC_SCORING_MODEL_PATH = SCRIPT_DIR / "risklens_static_model.joblib"

EXCLUDE_KEYWORDS = ["cacert", ".ai", ".lock", "idnadata", "uts46data",
                     ".rst", ".md", ".yml", ".yaml", ".toml"]

def clean_path(path):
    path = str(path).replace("\\", "/")
    path = path.replace("./", "")
    path = path.strip()
    if path.startswith("src/"):
        path = path[len("src/"):]
    return path

def is_source_file(path):
    if not path.endswith(".py"):
        return False
    return not any(bad in path.lower() for bad in EXCLUDE_KEYWORDS)


def read_csv_flexible_encoding(path, **kwargs):
    """Reads a CSV that might be UTF-8 or UTF-16 (common Windows PowerShell issue)."""
    try:
        return pd.read_csv(path, encoding="utf-8", **kwargs)
    except (UnicodeDecodeError, UnicodeError):
        return pd.read_csv(path, encoding="utf-16", **kwargs)


all_repo_data = []

for repo in REPOS:
    print(f"\n{'='*60}\nProcessing repo: {repo}\n{'='*60}")
    folder = SCRIPT_DIR / repo

    #  Load complexity.csv (Lizard)
    complexity = read_csv_flexible_encoding(
        folder / "complexity.csv",
        header=None,
        names=["NLOC", "CCN", "Tokens", "Params", "Length",
               "LongName", "File", "FunctionName", "Signature", "StartLine", "EndLine"],
    )
    for col in ["NLOC", "CCN", "Tokens", "Params", "Length"]:
        complexity[col] = pd.to_numeric(complexity[col], errors="coerce")
    complexity = complexity.dropna(subset=["NLOC", "CCN"])
    complexity["File"] = complexity["File"].apply(clean_path)
    complexity = complexity[complexity["File"].apply(is_source_file)]

    file_complexity = complexity.groupby("File").agg(
        avg_complexity=("CCN", "mean"),
        max_complexity=("CCN", "max"),
        total_nloc=("NLOC", "sum"),
        function_count=("CCN", "count"),
    ).reset_index()

    # ---------- Load revisions.csv (CodeMaat) ----------
    revisions = read_csv_flexible_encoding(folder / "revisions.csv")
    revisions["n-revs"] = pd.to_numeric(revisions["n-revs"], errors="coerce")
    revisions = revisions.dropna(subset=["n-revs"])
    revisions["entity"] = revisions["entity"].apply(clean_path)
    revisions = revisions[revisions["entity"].apply(is_source_file)]

    # ---------- Load churn.csv (CodeMaat) ----------
    churn = read_csv_flexible_encoding(folder / "churn.csv")
    for col in ["added", "deleted"]:
        if col in churn.columns:
            churn[col] = pd.to_numeric(churn[col], errors="coerce")
    churn = churn.dropna(subset=["added", "deleted"])
    churn["entity"] = churn["entity"].apply(clean_path)
    churn = churn[churn["entity"].apply(is_source_file)]

    # ---------- Load maintainability.json (Radon) ----------
    mi_path = folder / "maintainability.json"
    # Windows PowerShell redirects (>) often save files as UTF-16 with a BOM,
    # which breaks plain UTF-8 reading. Try UTF-8 first, fall back to UTF-16 automatically.
    try:
        with open(mi_path, "r", encoding="utf-8") as f:
            mi_data = json.load(f)
    except UnicodeDecodeError:
        with open(mi_path, "r", encoding="utf-16") as f:
            mi_data = json.load(f)
    mi_rows = []
    for filepath, values in mi_data.items():
        mi_rows.append({"File": clean_path(filepath), "maintainability_index": values.get("mi")})
    maintainability = pd.DataFrame(mi_rows)
    maintainability = maintainability[maintainability["File"].apply(is_source_file)]

    #  Load ownership_age.csv (PyDriller) 
    ownership = read_csv_flexible_encoding(folder / "ownership_age.csv")
    ownership["file"] = ownership["file"].apply(clean_path)
    ownership = ownership.rename(columns={"file": "File"})
    ownership = ownership[ownership["File"].apply(is_source_file)]

    #  Load bugfix_counts.csv (PyDriller + keyword detection) 
    bugfix_path = folder / "bugfix_counts.csv"
    if bugfix_path.exists():
        bugfix = read_csv_flexible_encoding(bugfix_path)
        bugfix["file"] = bugfix["file"].apply(clean_path)
        bugfix = bugfix.rename(columns={"file": "File"})
        bugfix = bugfix[bugfix["File"].apply(is_source_file)]
        bugfix = bugfix[["File", "bugfix_commits", "bugfix_ratio"]]
    else:
        print(f"  (no bugfix_counts.csv found for {repo} - skipping bug-fix features for this repo)")
        bugfix = pd.DataFrame(columns=["File", "bugfix_commits", "bugfix_ratio"])

    # ---------- Load coupling.csv (CodeMaat) ----------
    # CodeMaat's coupling output has one row PER PAIR of coupled files:
    # columns: entity, coupled, degree, average-revs
    # We aggregate this down to one row PER FILE: how many other files it's
    # coupled with (coupling_partner_count) and its average coupling degree.
    coupling_path = folder / "coupling.csv"
    if coupling_path.exists():
        coupling_raw = read_csv_flexible_encoding(coupling_path)
        coupling_raw["entity"] = coupling_raw["entity"].apply(clean_path)
        coupling_raw["coupled"] = coupling_raw["coupled"].apply(clean_path)
        coupling_raw["degree"] = pd.to_numeric(coupling_raw["degree"], errors="coerce")

        # Each file can appear as either "entity" or "coupled" in a given pair,
        # so we count/average from both directions and combine them.
        side_a = coupling_raw.groupby("entity").agg(
            partners_a=("coupled", "count"), avg_degree_a=("degree", "mean")
        ).reset_index().rename(columns={"entity": "File"})
        side_b = coupling_raw.groupby("coupled").agg(
            partners_b=("entity", "count"), avg_degree_b=("degree", "mean")
        ).reset_index().rename(columns={"coupled": "File"})

        coupling = pd.merge(side_a, side_b, on="File", how="outer").fillna(0)
        coupling["coupling_partner_count"] = coupling["partners_a"] + coupling["partners_b"]
        coupling["avg_coupling_degree"] = coupling[["avg_degree_a", "avg_degree_b"]].mean(axis=1)
        coupling = coupling[coupling["File"].apply(is_source_file)]
        coupling = coupling[["File", "coupling_partner_count", "avg_coupling_degree"]]
    else:
        print(f"  (no coupling.csv found for {repo} - skipping coupling features for this repo)")
        coupling = pd.DataFrame(columns=["File", "coupling_partner_count", "avg_coupling_degree"])

    # ---------- Merge everything for this repo ----------
    merged = pd.merge(file_complexity, revisions, left_on="File", right_on="entity", how="inner")
    merged = pd.merge(merged, churn, left_on="File", right_on="entity", how="inner", suffixes=("", "_churn"))
    merged = pd.merge(merged, maintainability, on="File", how="left")
    merged = pd.merge(merged, ownership, on="File", how="left")
    merged = pd.merge(merged, bugfix, on="File", how="left")
    merged = pd.merge(merged, coupling, on="File", how="left")

    # Deduplicate (same issue as before - renamed files can create duplicate rows)
    merged = merged.sort_values("n-revs", ascending=False).drop_duplicates(subset=["File"], keep="first")

    merged["repo"] = repo
    print(f"  -> {len(merged)} files merged for {repo}")
    all_repo_data.append(merged)

# ---------- Combine all 5 repos into one big dataset ----------

full_data = pd.concat(all_repo_data, ignore_index=True)
full_data["maintainability_index"] = full_data["maintainability_index"].fillna(full_data["maintainability_index"].median())
full_data["num_authors"] = full_data["num_authors"].fillna(1)
full_data["code_age_days"] = full_data["code_age_days"].fillna(full_data["code_age_days"].median())
full_data["pydriller_commit_count"] = full_data["pydriller_commit_count"].fillna(0)
full_data["bugfix_commits"] = full_data["bugfix_commits"].fillna(0)
full_data["bugfix_ratio"] = full_data["bugfix_ratio"].fillna(0)
full_data["coupling_partner_count"] = full_data["coupling_partner_count"].fillna(0)
full_data["avg_coupling_degree"] = full_data["avg_coupling_degree"].fillna(0)

print(f"\n\nTOTAL COMBINED DATASET: {len(full_data)} files across {len(REPOS)} repos")
print(full_data["repo"].value_counts())

# ---------- Correlation check (Pearson AND Spearman, as recommended) ----------

print("\n--- Correlation: Max Complexity vs Revisions (whole combined dataset) ---")
pearson_corr, pearson_p = pearsonr(full_data["max_complexity"], full_data["n-revs"])
spearman_corr, spearman_p = spearmanr(full_data["max_complexity"], full_data["n-revs"])
print(f"Pearson:  r = {pearson_corr:.3f}, p = {pearson_p:.5f}")
print(f"Spearman: r = {spearman_corr:.3f}, p = {spearman_p:.5f}  (more robust to outliers)")

# ---------- NEW: Correlation with bug-fix-specific commits (more precise than raw revisions) ----------

print("\n--- Correlation: Max Complexity vs BUG-FIX-SPECIFIC Commits ---")
bugfix_pearson, bugfix_p = pearsonr(full_data["max_complexity"], full_data["bugfix_commits"])
bugfix_spearman, bugfix_sp = spearmanr(full_data["max_complexity"], full_data["bugfix_commits"])
print(f"Pearson:  r = {bugfix_pearson:.3f}, p = {bugfix_p:.5f}")
print(f"Spearman: r = {bugfix_spearman:.3f}, p = {bugfix_sp:.5f}")
print("(This isolates commits whose message suggests a bug fix, rather than counting ALL changes)")

print("\n--- Correlation: Coupling Partner Count vs Revisions ---")
if full_data["coupling_partner_count"].sum() > 0:
    coupling_pearson, coupling_p = pearsonr(full_data["coupling_partner_count"], full_data["n-revs"])
    print(f"Pearson:  r = {coupling_pearson:.3f}, p = {coupling_p:.5f}")
    print("(Tests whether files with many 'hidden' coupled dependencies also change more often)")
else:
    print("No coupling data available - skipping this correlation.")

# ---------- Create historical change-proneness labels ----------

q1 = full_data["n-revs"].quantile(0.33)
q2 = full_data["n-revs"].quantile(0.66)
print(
    f"\nHistorical change-proneness cutoffs derived from all files: "
    f"Low <= {q1:.1f}, Medium <= {q2:.1f}, High > {q2:.1f}"
)

def label_risk(revs):
    if revs <= q1:
        return "Low"
    elif revs <= q2:
        return "Medium"
    else:
        return "High"

full_data["risk_label"] = full_data["n-revs"].apply(label_risk)
print("\nHistorical change-proneness label distribution:")
print(full_data["risk_label"].value_counts())

# ---------- Build feature set for the model ----------

FEATURES = ["avg_complexity", "max_complexity", "total_nloc", "function_count",
            "maintainability_index", "num_authors", "code_age_days",
            "coupling_partner_count", "avg_coupling_degree"]
STATIC_FEATURES = [
    "avg_complexity",
    "max_complexity",
    "total_nloc",
    "function_count",
    "maintainability_index",
]
# Note: bugfix_commits/bugfix_ratio are deliberately NOT included as model features,
# since they are extremely close to (derived from) the same commit history used to
# build the risk label itself - including them would let the model "cheat" by nearly
# seeing the answer. They are used only as an independent correlation check above.

model_data = full_data.dropna(subset=FEATURES + ["risk_label"])
X = model_data[FEATURES]
y = model_data["risk_label"]

print(f"\nFinal modeling dataset: {len(model_data)} files, {len(FEATURES)} features")
print(
    "Target: historical change-proneness based on revision counts; "
    "this is not a confirmed-bug or future-defect label."
)

# Persist scoring models trained on all labeled historical files. The static
# model deliberately uses only source-code measurements for files with little
# repository history.
training_cutoffs = {
    "low_max_revisions": float(q1),
    "medium_max_revisions": float(q2),
}
scoring_models = [
    (
        SCORING_MODEL_PATH,
        FEATURES,
        "full",
        RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42,
            class_weight="balanced",
        ),
    ),
    (
        STATIC_SCORING_MODEL_PATH,
        STATIC_FEATURES,
        "static",
        RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42,
            class_weight="balanced",
        ),
    ),
]
for artifact_path, feature_order, model_kind, scoring_model in scoring_models:
    scoring_model.fit(model_data[feature_order], model_data["risk_label"])
    joblib.dump(
        {
            "model": scoring_model,
            "feature_order": feature_order,
            "cutoffs": training_cutoffs,
            "classes": ["Low", "Medium", "High"],
            "model_kind": model_kind,
            "label_definition": (
                "Historical change-proneness from training revision-count "
                "terciles; not a defect or future-risk label."
            ),
        },
        artifact_path,
    )
    print(f"Saved {model_kind} scoring model: {artifact_path.name}")

# ---------- Train Random Forest with cross-validation (not just one split) ----------

rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=skf, scoring="f1_macro")
print(f"\n--- Cross-Validated Model Performance (5-fold) ---")
print(f"F1-macro scores per fold: {[round(s,3) for s in cv_scores]}")
print(f"Mean F1-macro: {cv_scores.mean():.3f}  (std: {cv_scores.std():.3f})")

# ---------- Also do one train/test split for a detailed report + confusion matrix ----------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("\n--- Detailed classification report (held-out 25% test set) ---")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=["Low", "Medium", "High"])
print("Confusion matrix (rows=actual, cols=predicted), order = [Low, Medium, High]:")
print(cm)

# ---------- Evaluate generalization to an unseen repository ----------

print("\n--- Leave-One-Repository-Out Generalization ---")
print(
    "Each fold trains on four repositories and predicts the fifth. "
    "For each fold, revision-count class cutoffs are computed from its training "
    "repositories and applied to both train and held-out files."
)
holdout_records = []
repository_scores = []
risk_classes = ["Low", "Medium", "High"]

for held_out_repo in sorted(model_data["repo"].unique()):
    train_mask = model_data["repo"] != held_out_repo
    test_mask = ~train_mask
    train_rows = model_data.loc[train_mask]
    test_rows = model_data.loc[test_mask]

    train_q1 = train_rows["n-revs"].quantile(0.33)
    train_q2 = train_rows["n-revs"].quantile(0.66)

    def label_with_cutoffs(revisions):
        if revisions <= train_q1:
            return "Low"
        if revisions <= train_q2:
            return "Medium"
        return "High"

    y_train_repo = train_rows["n-revs"].apply(label_with_cutoffs)
    y_test_repo = test_rows["n-revs"].apply(label_with_cutoffs)
    repo_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    )
    repo_model.fit(train_rows[FEATURES], y_train_repo)
    repo_predictions = repo_model.predict(test_rows[FEATURES])
    repo_probabilities = repo_model.predict_proba(test_rows[FEATURES])
    high_class_index = list(repo_model.classes_).index("High")

    repo_f1 = f1_score(
        y_test_repo, repo_predictions, labels=risk_classes,
        average="macro", zero_division=0,
    )
    repo_accuracy = accuracy_score(y_test_repo, repo_predictions)
    repository_scores.append({
        "repository": held_out_repo,
        "rows": int(len(test_rows)),
        "f1_macro": float(repo_f1),
        "accuracy": float(repo_accuracy),
        "train_low_max_revisions": float(train_q1),
        "train_medium_max_revisions": float(train_q2),
    })
    print(
        f"  {held_out_repo}: n={len(test_rows)}, "
        f"macro F1={repo_f1:.3f}, accuracy={repo_accuracy:.3f}"
    )

    for row_index, actual, predicted, probabilities in zip(
        test_rows.index,
        y_test_repo,
        repo_predictions,
        repo_probabilities,
    ):
        holdout_records.append({
            "row_index": row_index,
            "repo_holdout_actual_label": actual,
            "repo_holdout_prediction": predicted,
            "repo_holdout_high_probability": float(
                probabilities[high_class_index]
            ),
            "repo_holdout_low_max_revisions": float(train_q1),
            "repo_holdout_medium_max_revisions": float(train_q2),
        })

holdout_predictions = pd.DataFrame(holdout_records).set_index("row_index")
full_data = full_data.join(holdout_predictions, how="left")

all_holdout_actual = holdout_predictions["repo_holdout_actual_label"]
all_holdout_predicted = holdout_predictions["repo_holdout_prediction"]
overall_holdout_f1 = f1_score(
    all_holdout_actual, all_holdout_predicted, labels=risk_classes,
    average="macro", zero_division=0,
)
overall_holdout_accuracy = accuracy_score(
    all_holdout_actual, all_holdout_predicted
)
generalization_results = {
    "evaluation_method": "leave-one-repository-out cross-validation",
    "target": "historical change-proneness from revision-count quantiles",
    "feature_set": FEATURES,
    "rows": int(len(holdout_predictions)),
    "repositories": repository_scores,
    "overall_oof_f1_macro": float(overall_holdout_f1),
    "overall_oof_accuracy": float(overall_holdout_accuracy),
    "mean_repository_f1_macro": float(np.mean(
        [row["f1_macro"] for row in repository_scores]
    )),
    "note": (
        "Each held-out repository is labeled using revision-count cutoffs "
        "calculated only from the other repositories. Features are whole-history "
        "aggregates, so this is cross-repository generalization, not a future-time "
        "prediction evaluation."
    ),
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
}
GENERALIZATION_RESULTS_PATH.write_text(
    json.dumps(generalization_results, indent=2), encoding="utf-8"
)
full_data.to_csv(SCRIPT_DIR / "full_merged_dataset.csv", index=False)
print(
    f"Overall repository-held-out macro F1: {overall_holdout_f1:.3f}; "
    f"accuracy: {overall_holdout_accuracy:.3f}"
)
print(f"Saved held-out predictions in full_merged_dataset.csv")
print(f"Saved repository evaluation: {GENERALIZATION_RESULTS_PATH.name}")

# ---------- Feature importance (built-in, quick view before SHAP) ----------

importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\n--- Feature Importance (Random Forest built-in) ---")
print(importances)

master_results = {
    "f1": float(cv_scores.mean()),
    "evaluation_method": "5-fold stratified cross-validation",
    "folds": 5,
    "rows": int(len(model_data)),
    "random_seed": 42,
    "feature_importance": [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in importances.items()
    ],
    "repository_holdout_f1_macro": float(overall_holdout_f1),
    "repository_holdout_accuracy": float(overall_holdout_accuracy),
    "repository_holdout_method": "leave-one-repository-out cross-validation",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "library_versions": {
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": version("scikit-learn"),
    },
}
RESULTS_PATH.write_text(json.dumps(master_results, indent=2), encoding="utf-8")
print(f"Saved results: {RESULTS_PATH.name}")

plt.figure(figsize=(8, 5))
importances.plot(kind="barh", color="#2E74B5")
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Which Metrics Matter Most for Predicting Risk")
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "feature_importance.png")
print("Saved chart: feature_importance.png")

# ---------- SHAP explainability ----------

try:
    import shap
    print("\n--- Generating SHAP explanations (this may take a minute) ---")
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)

    # Summary plot for the "High" risk class
    class_index = list(rf.classes_).index("High")
    plt.figure()
    shap.summary_plot(shap_values[:, :, class_index], X_test, show=False)
    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / "shap_summary_high_risk.png")
    print("Saved chart: shap_summary_high_risk.png")
except Exception as e:
    print(f"\nSHAP step skipped due to an error: {e}")
    print("(Everything else above is still valid - SHAP is an optional add-on.)")

# ---------- Compare against your ORIGINAL baseline (single repo, linear regression) ----------

print("\n\n=== COMPARISON: New Model vs Your Original Baseline ===")
print("Original baseline (Requests only, linear regression, 29 files):")
print("   Correlation r = 0.733 | R-squared = 0.538 | MAE = 84.13 revisions")
print("\nNew model (5 repos, Random Forest classifier, cross-validated):")
print(f"   Combined dataset size = {len(full_data)} files")
print(f"   Pearson correlation r = {pearson_corr:.3f}")
print(f"   Spearman correlation r = {spearman_corr:.3f}")
print(f"   Cross-validated F1-macro = {cv_scores.mean():.3f}")

print(
    "\nDone. Files saved: full_merged_dataset.csv, master_results.json, "
    "generalization_results.json, feature_importance.png, "
    "shap_summary_high_risk.png"
)
