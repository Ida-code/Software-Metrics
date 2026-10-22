"""
Software Complexity vs Bug-Proneness Analysis
Merges Lizard complexity data with CodeMaat change/churn data,
runs correlation + regression, and builds a tested prediction model.

HOW TO USE:
1. Place this script inside your cloned project folder (e.g. C:\\Users\\idash\\SoftwareMetricsProject\\requests)
2. Make sure complexity.csv, revisions.csv, churn.csv are in the same folder
3. Install required packages once:
   pip install pandas scipy statsmodels scikit-learn matplotlib
4. Run:
   python analyze.py
"""

import pandas as pd
from scipy.stats import pearsonr
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# ---------- STEP 1: Load the raw files ----------

complexity = pd.read_csv(
    "complexity.csv",
    header=None,
    names=["NLOC", "CCN", "Tokens", "Params", "Length",
           "LongName", "File", "FunctionName", "Signature", "StartLine", "EndLine"],
)

# ---------- STEP 1b: Force numeric columns to actually be numbers ----------
# Lizard's CSV sometimes has a trailing summary row or stray header text.
# This converts bad values to NaN and then drops those rows.

numeric_cols = ["NLOC", "CCN", "Tokens", "Params", "Length"]
for col in numeric_cols:
    complexity[col] = pd.to_numeric(complexity[col], errors="coerce")

before_count = len(complexity)
complexity = complexity.dropna(subset=numeric_cols)
after_count = len(complexity)
if before_count != after_count:
    print(f"Removed {before_count - after_count} non-numeric row(s) from complexity.csv (likely a summary/header row).")

complexity["CCN"] = complexity["CCN"].astype(int)
complexity["NLOC"] = complexity["NLOC"].astype(int)

revisions = pd.read_csv("revisions.csv")   # columns: entity, n-revs
churn = pd.read_csv("churn.csv")           # columns: entity, added, deleted, commits

revisions["n-revs"] = pd.to_numeric(revisions["n-revs"], errors="coerce")
revisions = revisions.dropna(subset=["n-revs"])

for col in ["added", "deleted", "commits"]:
    if col in churn.columns:
        churn[col] = pd.to_numeric(churn[col], errors="coerce")
churn = churn.dropna(subset=["added", "deleted"])

# ---------- STEP 2: Clean file paths so they match across tools ----------

def clean_path(path):
    path = str(path).replace("\\", "/")
    path = path.replace("./", "")
    path = path.strip()
    # Normalize so paths match regardless of a leading "src/" folder,
    # since Lizard and CodeMaat sometimes report paths differently.
    if path.startswith("src/"):
        path = path[len("src/"):]
    return path

complexity["File"] = complexity["File"].apply(clean_path)
revisions["entity"] = revisions["entity"].apply(clean_path)
churn["entity"] = churn["entity"].apply(clean_path)

# ---------- STEP 3: Keep only real source code files ----------
# Filters out certs, images, lock files, docs, and generated data files.

EXCLUDE_KEYWORDS = ["cacert", ".ai", ".lock", "idnadata", "uts46data",
                     ".rst", ".md", ".yml", ".yaml", ".toml"]

def is_source_file(path):
    if not path.endswith(".py"):
        return False
    return not any(bad in path.lower() for bad in EXCLUDE_KEYWORDS)

complexity = complexity[complexity["File"].apply(is_source_file)]
revisions = revisions[revisions["entity"].apply(is_source_file)]
churn = churn[churn["entity"].apply(is_source_file)]

# ---------- STEP 4: Aggregate complexity per file ----------
# Lizard gives one row per FUNCTION - we need one row per FILE.

file_complexity = complexity.groupby("File").agg(
    avg_complexity=("CCN", "mean"),
    max_complexity=("CCN", "max"),
    total_nloc=("NLOC", "sum"),
    function_count=("CCN", "count"),
).reset_index()

# ---------- STEP 5: Merge complexity + revisions + churn ----------

merged = pd.merge(file_complexity, revisions, left_on="File", right_on="entity", how="inner")
merged = pd.merge(merged, churn, left_on="File", right_on="entity", how="inner", suffixes=("", "_churn"))

if len(merged) == 0:
    print("\nFull path match found 0 rows - retrying by matching on file name only...")
    file_complexity["_basename"] = file_complexity["File"].apply(lambda p: p.split("/")[-1])
    revisions["_basename"] = revisions["entity"].apply(lambda p: p.split("/")[-1])
    churn["_basename"] = churn["entity"].apply(lambda p: p.split("/")[-1])

    merged = pd.merge(file_complexity, revisions, on="_basename", how="inner", suffixes=("", "_rev"))
    merged = pd.merge(merged, churn, on="_basename", how="inner", suffixes=("", "_churn"))

merged = merged[["File", "avg_complexity", "max_complexity", "total_nloc",
                  "function_count", "n-revs", "added", "deleted", "commits"]]

if len(merged) == 0:
    print("\nWARNING: The merge still produced 0 rows even by file name only. This usually means the file paths")
    print("in complexity.csv don't match the paths in revisions.csv/churn.csv at all.")
    print("Sample complexity file paths:", file_complexity['File'].head(5).tolist())
    print("Sample revisions entity paths:", revisions['entity'].head(5).tolist())
    raise SystemExit("Fix file path mismatch above, then re-run.")

# ---------- STEP 5b: Remove duplicate file entries ----------
# Can happen if a file was renamed/moved in git history and appears under
# more than one historical path, or if both merge attempts matched it.
# We keep the row with the HIGHEST n-revs for each file (the most complete history).

before_dedup = len(merged)
merged = merged.sort_values("n-revs", ascending=False).drop_duplicates(subset=["File"], keep="first")
after_dedup = len(merged)
if before_dedup != after_dedup:
    print(f"\nRemoved {before_dedup - after_dedup} duplicate file row(s) - kept the entry with the most complete history for each file.")
merged = merged.reset_index(drop=True)

merged.to_csv("merged_data.csv", index=False)
print(f"\nMerged dataset created: merged_data.csv  ({len(merged)} files)\n")
print(merged.sort_values("max_complexity", ascending=False).head(10))

# ---------- STEP 6: Correlation ----------

print("\n--- CORRELATION: Max Complexity vs Number of Revisions ---")
corr, pvalue = pearsonr(merged["max_complexity"], merged["n-revs"])
print(f"Correlation coefficient: {corr:.3f}")
print(f"P-value: {pvalue:.5f}")
if pvalue < 0.05:
    print("=> Statistically significant relationship found.")
else:
    print("=> No statistically significant relationship found (this is still a valid, honest result).")

# ---------- STEP 7: Regression ----------

print("\n--- REGRESSION: Predicting Revisions from Max Complexity ---")
X = sm.add_constant(merged["max_complexity"])
model = sm.OLS(merged["n-revs"], X).fit()
print(model.summary())

# ---------- STEP 8: Train/test split - honest prediction test ----------

print("\n--- PREDICTION MODEL: Trained on 70% of files, tested on the other 30% ---")
train, test = train_test_split(merged, test_size=0.3, random_state=42)

lr = LinearRegression()
lr.fit(train[["max_complexity"]], train["n-revs"])

predictions = lr.predict(test[["max_complexity"]])
mae = mean_absolute_error(test["n-revs"], predictions)

print(f"Mean Absolute Error on UNSEEN files: {mae:.2f} revisions")
print("(This means, on average, predictions were off by this many revisions on files the model never saw.)")

comparison = test[["File", "max_complexity", "n-revs"]].copy()
comparison["predicted_revs"] = predictions.round(1)
comparison.to_csv("prediction_results.csv", index=False)
print("\nSaved prediction_results.csv - shows actual vs predicted for held-out files.")
print(comparison.sort_values("n-revs", ascending=False).head(10))

# ---------- STEP 8b: Log-transformed regression and prediction ----------
# Your data likely has a few extreme outlier files (e.g. one file changed
# hundreds of times while most are changed only a handful of times).
# Taking the log of n-revs compresses that scale, usually gives a better-fitting
# and more honest model, and is standard practice in this kind of analysis.

import numpy as np

print("\n\n=== LOG-TRANSFORMED VERSION (recommended for reporting) ===")
merged["log_n_revs"] = np.log1p(merged["n-revs"])  # log(1 + x) handles files with 0 revisions safely

print("\n--- CORRELATION (log scale): Max Complexity vs log(Revisions) ---")
corr_log, pvalue_log = pearsonr(merged["max_complexity"], merged["log_n_revs"])
print(f"Correlation coefficient: {corr_log:.3f}")
print(f"P-value: {pvalue_log:.5f}")

print("\n--- REGRESSION (log scale) ---")
X_log = sm.add_constant(merged["max_complexity"])
model_log = sm.OLS(merged["log_n_revs"], X_log).fit()
print(model_log.summary())

print("\n--- PREDICTION MODEL (log scale): Trained on 70%, tested on 30% ---")
train_log, test_log = train_test_split(merged, test_size=0.3, random_state=42)

lr_log = LinearRegression()
lr_log.fit(train_log[["max_complexity"]], train_log["log_n_revs"])

log_predictions = lr_log.predict(test_log[["max_complexity"]])
# Convert predictions back from log scale to real revision counts for interpretability
real_predictions = np.expm1(log_predictions)
mae_log = mean_absolute_error(test_log["n-revs"], real_predictions)

print(f"Mean Absolute Error on UNSEEN files (log model, converted back to real scale): {mae_log:.2f} revisions")
print("Compare this to the earlier plain regression's MAE - the log model is usually")
print("more accurate for datasets with a few extreme outlier files.")

comparison_log = test_log[["File", "max_complexity", "n-revs"]].copy()
comparison_log["predicted_revs_log_model"] = real_predictions.round(1)
comparison_log.to_csv("prediction_results_log_model.csv", index=False)
print("\nSaved prediction_results_log_model.csv")
print(comparison_log.sort_values("n-revs", ascending=False).head(10))

# Log-scale scatter plot - usually much clearer to read than the raw scale
plt.figure(figsize=(8, 6))
plt.scatter(merged["max_complexity"], merged["log_n_revs"], alpha=0.6, color="darkorange")
plt.xlabel("Max Cyclomatic Complexity (per file)")
plt.ylabel("log(1 + Number of Revisions)")
plt.title("Code Complexity vs log(Change Frequency)")
plt.tight_layout()
plt.savefig("scatter_complexity_vs_log_revisions.png")
print("Saved chart: scatter_complexity_vs_log_revisions.png")

# ---------- STEP 9: Visualizations ----------

plt.figure(figsize=(8, 6))
plt.scatter(merged["max_complexity"], merged["n-revs"], alpha=0.6)
plt.xlabel("Max Cyclomatic Complexity (per file)")
plt.ylabel("Number of Revisions")
plt.title("Code Complexity vs Change Frequency")
plt.tight_layout()
plt.savefig("scatter_complexity_vs_revisions.png")
print("\nSaved chart: scatter_complexity_vs_revisions.png")

top10 = merged.sort_values("n-revs", ascending=False).head(10)
plt.figure(figsize=(10, 6))
plt.barh(top10["File"], top10["n-revs"])
plt.xlabel("Number of Revisions")
plt.title("Top 10 Most-Changed Files")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("top10_riskiest_files.png")
print("Saved chart: top10_riskiest_files.png")

print("\nDone. Open merged_data.csv, prediction_results.csv, and the two PNG charts for your report/demo.")
