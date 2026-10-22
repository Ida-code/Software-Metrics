"""
File-Level SonarQube Comparison

Pulls per-file metrics (complexity, cognitive complexity, code smells, bugs,
duplications) directly from SonarQube's Web API for all 5 projects, merges
them against your existing full_merged_dataset.csv by filename, and compares
SonarQube's file-level scores against your model's file-level risk predictions.

HOW TO USE:
1. Make sure your SonarQube server is running (StartSonar.bat) and all 5
   projects have been scanned (requests-analysis, flask-analysis,
   click-analysis, httpx-analysis, rich-analysis).
2. Place this script in your main folder (same as full_merged_dataset.csv).
3. Install requests once if needed:
   pip install requests
4. Set a SonarQube user token with Browse permission for all five projects:
   $env:SONAR_TOKEN = "<your-token>"
5. Run:
   python sonarqube_file_comparison.py
   Use --use-cached to reuse sonarqube_file_level.csv without contacting SonarQube.

The token is read from the environment and is not written to disk.
"""

import argparse
import os
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

SONAR_URL = "http://localhost:9000"
SONAR_TOKEN = os.getenv("SONAR_TOKEN")
SONAR_FILE_LEVEL_PATH = "sonarqube_file_level.csv"

PROJECTS = {
    "requests-analysis": "requests",
    "flask-analysis": "flask",
    "click-analysis": "click",
    "httpx-analysis": "httpx",
    "rich-analysis": "rich",
}

METRICS = "complexity,cognitive_complexity,code_smells,bugs,duplicated_lines_density,ncloc"


def fetch_project_files(project_key):
    """Fetch per-file metrics for one SonarQube project, handling pagination."""
    all_components = []
    page = 1
    while True:
        url = (
            f"{SONAR_URL}/api/measures/component_tree"
            f"?component={project_key}&metricKeys={METRICS}&qualifiers=FIL"
            f"&ps=500&p={page}"
        )
        headers = {"Authorization": f"Bearer {SONAR_TOKEN}"} if SONAR_TOKEN else {}
        resp = requests.get(url, headers=headers, timeout=(5, 30))
        resp.raise_for_status()
        data = resp.json()

        for comp in data.get("components", []):
            row = {"sonar_path": comp.get("path", ""), "project": project_key}
            for m in comp.get("measures", []):
                # Prefix every SonarQube metric with "sonar_" so it's never
                # ambiguous, in the merged table or in a demo, which tool
                # a given column came from.
                row[f"sonar_{m['metric']}"] = m.get("value")
            all_components.append(row)

        total = data["paging"]["total"]
        page_size = data["paging"]["pageSize"]
        if page * page_size >= total:
            break
        page += 1

    return pd.DataFrame(all_components)


def clean_path(path):
    path = str(path).replace("\\", "/").strip()
    if path.startswith("src/"):
        path = path[len("src/"):]
    return path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--use-cached",
    action="store_true",
    help="Reuse sonarqube_file_level.csv instead of requesting SonarQube again.",
)
args = parser.parse_args()

# ---------- STEP 1: Fetch or load SonarQube file-level metrics ----------

if args.use_cached:
    if not os.path.exists(SONAR_FILE_LEVEL_PATH):
        raise SystemExit(
            f"{SONAR_FILE_LEVEL_PATH} does not exist. Run without --use-cached "
            "while SonarQube is available to download file-level metrics."
        )
    print(f"Loading cached SonarQube data from {SONAR_FILE_LEVEL_PATH}...")
    sonar_df = pd.read_csv(SONAR_FILE_LEVEL_PATH)
else:
    print("Fetching per-file data from SonarQube for all 5 projects...")
    all_sonar_data = []
    failed_projects = []
    for project_key, repo_name in PROJECTS.items():
        print(f"  Fetching {project_key} ...")
        try:
            df = fetch_project_files(project_key)
            df["repo"] = repo_name
            all_sonar_data.append(df)
            print(f"    -> {len(df)} files")
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else "unknown"
            if status == 401:
                reason = "authentication required or token is invalid"
            elif status == 403:
                reason = "token lacks permission to browse this project"
            elif status == 404:
                reason = "project not found, not accessible, or API endpoint unavailable"
            else:
                reason = f"HTTP {status}"
            failed_projects.append(f"{project_key}: {reason}")
            print(f"    FAILED for {project_key}: {reason}")
        except requests.RequestException as e:
            failed_projects.append(f"{project_key}: {e}")
            print(f"    FAILED for {project_key}: {e}")

    if not all_sonar_data:
        details = "; ".join(failed_projects)
        if not SONAR_TOKEN:
            details += (
                "\nSet a SonarQube user token in the SONAR_TOKEN environment variable "
                "and ensure it has Browse permission for these projects."
            )
        raise SystemExit(
            "Could not fetch data from any SonarQube project. "
            f"Check that SonarQube is running at {SONAR_URL} and verify access. "
            f"Details: {details}"
        )

    if failed_projects:
        print("\nWARNING: Continuing with partial SonarQube results.")
        for failure in failed_projects:
            print(f"  - {failure}")

    sonar_df = pd.concat(all_sonar_data, ignore_index=True)
    if sonar_df.empty:
        raise SystemExit(
            "SonarQube returned no file-level components for the accessible projects. "
            "Confirm the projects have completed scans and contain Python files."
        )

# Convert numeric columns
for col in ["sonar_complexity", "sonar_cognitive_complexity", "sonar_code_smells", "sonar_bugs", "sonar_duplicated_lines_density", "sonar_ncloc"]:
    if col in sonar_df.columns:
        sonar_df[col] = pd.to_numeric(sonar_df[col], errors="coerce")

sonar_df["clean_path"] = sonar_df["sonar_path"].apply(clean_path)
sonar_df = sonar_df[sonar_df["clean_path"].str.endswith(".py")]

for issue_metric in [
    "sonar_code_smells",
    "sonar_bugs",
    "sonar_duplicated_lines_density",
]:
    if issue_metric in sonar_df.columns:
        sonar_df[issue_metric] = sonar_df[issue_metric].fillna(0)

if not args.use_cached:
    sonar_df.to_csv(SONAR_FILE_LEVEL_PATH, index=False)
    print(
        f"\nSaved {SONAR_FILE_LEVEL_PATH} with {len(sonar_df)} "
        "total files across scanned repos."
    )

# ---------- STEP 2: Load your existing model dataset ----------

print("\nLoading your full_merged_dataset.csv ...")
model_df = pd.read_csv("full_merged_dataset.csv")
if "risk_label" not in model_df.columns:
    if "n-revs" not in model_df.columns:
        raise SystemExit(
            "The merged dataset has no risk_label or n-revs column, so historical "
            "risk categories cannot be created."
        )
    q1 = model_df["n-revs"].quantile(0.33)
    q2 = model_df["n-revs"].quantile(0.66)
    model_df["risk_label"] = model_df["n-revs"].apply(
        lambda revisions: "Low" if revisions <= q1
        else ("Medium" if revisions <= q2 else "High")
    )
    print(
        "Created risk_label from revision-count quantiles: "
        f"Low <= {q1:.1f}, Medium <= {q2:.1f}, High > {q2:.1f}"
    )
required_predictions = [
    "repo_holdout_actual_label",
    "repo_holdout_prediction",
    "repo_holdout_high_probability",
]
missing_predictions = [
    column for column in required_predictions if column not in model_df.columns
]
if missing_predictions:
    raise SystemExit(
        "Repository-held-out model predictions are missing from "
        "full_merged_dataset.csv. Run master_analysis.py first to generate "
        "out-of-repository predictions before comparing with SonarQube."
    )
model_df["clean_path"] = model_df["File"].apply(clean_path)

# ---------- STEP 3: Merge on repo + filename ----------

merged = pd.merge(
    model_df, sonar_df,
    left_on=["repo", "clean_path"], right_on=["repo", "clean_path"],
    how="inner", suffixes=("", "_sonar")
)

print(f"\nSuccessfully matched {len(merged)} files between your dataset and SonarQube (out of {len(model_df)} total files).")

if len(merged) == 0:
    print("\nWARNING: 0 files matched. Check that SonarQube's file paths match your dataset's paths.")
    print("Sample SonarQube paths:", sonar_df["clean_path"].head(5).tolist())
    print("Sample your dataset paths:", model_df["clean_path"].head(5).tolist())
else:
    merged.to_csv("file_level_comparison.csv", index=False)
    print("Saved file_level_comparison.csv")
    print("\nColumn naming convention: every SonarQube metric is prefixed 'sonar_' "
          "(e.g., sonar_complexity, sonar_code_smells). Model predictions use "
          "leave-one-repository-out evaluation: each file is predicted by a model "
          "that did not train on its repository.")

    # ---------- STEP 4: Compare SonarQube complexity with Lizard complexity ----------

    print(
        "\n--- SonarQube file complexity vs Lizard maximum function complexity ---"
    )
    print(
        "These related measures use different scopes: SonarQube reports file-level "
        "complexity while Lizard's max_complexity is the largest function value."
    )
    valid = merged.dropna(subset=["sonar_complexity", "max_complexity"])
    if len(valid) > 2:
        r, p = pearsonr(valid["sonar_complexity"], valid["max_complexity"])
        print(f"Pearson r = {r:.3f}, p = {p:.5f}  (n={len(valid)} files)")

    # ---------- STEP 5: Compare SonarQube findings with held-out model predictions ----------

    print("\n--- SonarQube findings vs repository-held-out model predictions ---")
    predicted_order = {"Low": 0, "Medium": 1, "High": 2}
    merged["predicted_risk_numeric"] = merged[
        "repo_holdout_prediction"
    ].map(predicted_order)
    for metric in [
        "sonar_code_smells",
        "sonar_bugs",
        "sonar_complexity",
        "sonar_duplicated_lines_density",
    ]:
        if metric not in merged.columns:
            continue
        valid = merged.dropna(
            subset=[metric, "repo_holdout_high_probability"]
        )
        if len(valid) > 2:
            correlation, p_value = spearmanr(
                valid[metric], valid["repo_holdout_high_probability"]
            )
            print(
                f"{metric} vs held-out high-change probability: "
                f"Spearman r = {correlation:.3f}, p = {p_value:.5f} "
                f"(n={len(valid)} files)"
            )

    print(
        "\nInterpretation: these correlations compare SonarQube findings with "
        "predicted historical change-proneness. They do not establish causation "
        "or prove future defects. SonarQube complexity is related to complexity "
        "features used by the model, so that comparison is not an independent "
        "validation; issue counts such as code smells and bugs are separate signals."
    )

    # ---------- STEP 6: Inspect agreement and disagreement examples ----------

    print(
        "\n--- High model-predicted change-proneness with SonarQube findings ---"
    )
    issue_columns = [
        column for column in ["sonar_code_smells", "sonar_bugs"]
        if column in merged.columns
    ]
    if issue_columns:
        merged["sonar_issue_count"] = merged[issue_columns].fillna(0).sum(axis=1)
        agreements = merged[
            (merged["repo_holdout_prediction"] == "High")
            & (merged["sonar_issue_count"] > 0)
        ].sort_values(
            ["sonar_issue_count", "repo_holdout_high_probability"],
            ascending=False,
        ).head(5)
        agreement_columns = [
            column for column in [
                "File", "repo", "repo_holdout_prediction",
                "repo_holdout_high_probability", "risk_label",
                "sonar_code_smells", "sonar_bugs", "sonar_complexity",
            ]
            if column in agreements.columns
        ]
        print(agreements[agreement_columns].to_string(index=False))

        print(
            "\n--- High model-predicted change-proneness with no reported SonarQube issues ---"
        )
        disagreements = merged[
            (merged["repo_holdout_prediction"] == "High")
            & (merged["sonar_issue_count"] == 0)
        ].sort_values("repo_holdout_high_probability", ascending=False).head(5)
        disagreement_columns = [
            column for column in [
                "File", "repo", "repo_holdout_prediction",
                "repo_holdout_high_probability", "risk_label",
                "sonar_code_smells", "sonar_bugs", "sonar_complexity",
            ]
            if column in disagreements.columns
        ]
        print(disagreements[disagreement_columns].to_string(index=False))

    # ---------- STEP 7: Visualize ----------

    plt.figure(figsize=(8, 6))
    colors = merged["repo_holdout_prediction"].map(
        {"Low": "#2E7D32", "Medium": "#F9A825", "High": "#C62828"}
    )
    plt.scatter(merged["sonar_code_smells"], merged["sonar_complexity"], c=colors, alpha=0.6, s=40)
    plt.xlabel("SonarQube: Code Smells (per file)")
    plt.ylabel("SonarQube: Complexity (per file)")
    plt.title("SonarQube Metrics Colored by Held-Out Change-Proneness Prediction")
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2E7D32", label="Low predicted change-proneness"),
        Patch(facecolor="#F9A825", label="Medium predicted change-proneness"),
        Patch(facecolor="#C62828", label="High predicted change-proneness"),
    ]
    plt.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig("file_level_sonarqube_comparison.png", dpi=150)
    print("\nSaved chart: file_level_sonarqube_comparison.png")

print("\nDone.")
