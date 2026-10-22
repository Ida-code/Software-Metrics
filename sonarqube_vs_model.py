"""
Compare SonarQube features with repository metrics for historical change classification.

Loads full_merged_dataset.csv and sonarqube_file_level.csv from this folder,
matches files by repository and normalized path, and evaluates five feature sets
using identical stratified folds. The target is Low/Medium/High historical
revision activity, not bug status or future defect risk.

Run from the project folder:
    python sonarqube_vs_model.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DATA_PATH = SCRIPT_DIR / "full_merged_dataset.csv"
SONAR_DATA_PATH = SCRIPT_DIR / "sonarqube_file_level.csv"
RESULTS_PATH = SCRIPT_DIR / "sonarqube_vs_model_results.csv"
CHART_PATH = SCRIPT_DIR / "sonarqube_vs_model_chart.png"

LABELS = ["Low", "Medium", "High"]
SONAR_METRICS = [
    "sonar_complexity",
    "sonar_cognitive_complexity",
    "sonar_code_smells",
    "sonar_bugs",
    "sonar_duplicated_lines_density",
    "sonar_ncloc",
    "sonar_smells_per_100_lines",
]
STATIC_METRICS = [
    "avg_complexity",
    "max_complexity",
    "total_nloc",
    "function_count",
    "maintainability_index",
]
HISTORY_METRICS = [
    "num_authors",
    "code_age_days",
    "coupling_partner_count",
    "avg_coupling_degree",
]
FEATURE_SETS = {
    "a_sonarqube_only": SONAR_METRICS,
    "b_static_only": STATIC_METRICS,
    "c_history_only": HISTORY_METRICS,
    "d_existing_nine_features": STATIC_METRICS + HISTORY_METRICS,
    "e_existing_plus_sonarqube": STATIC_METRICS + HISTORY_METRICS + SONAR_METRICS,
}
FEATURE_SET_LABELS = {
    "a_sonarqube_only": "A: SonarQube only",
    "b_static_only": "B: Static code metrics",
    "c_history_only": "C: Repository history",
    "d_existing_nine_features": "D: Existing nine features",
    "e_existing_plus_sonarqube": "E: Nine features + SonarQube",
}


def normalize_path(value: object) -> str:
    path = str(value).replace("\\", "/").strip()
    while path.startswith("./"):
        path = path[2:]
    if path.startswith("src/"):
        path = path[len("src/"):]
    return path


def load_matched_data() -> pd.DataFrame:
    for path in [MODEL_DATA_PATH, SONAR_DATA_PATH]:
        if not path.is_file():
            raise FileNotFoundError(f"Required input file not found: {path}")

    model_data = pd.read_csv(MODEL_DATA_PATH)
    sonar_data = pd.read_csv(SONAR_DATA_PATH)
    required_model = {
        "File",
        "repo",
        "n-revs",
        *STATIC_METRICS,
        *HISTORY_METRICS,
    }
    required_sonar = {
        "repo",
        "sonar_complexity",
        "sonar_cognitive_complexity",
        "sonar_code_smells",
        "sonar_bugs",
        "sonar_duplicated_lines_density",
        "sonar_ncloc",
    }
    missing_model = sorted(required_model - set(model_data.columns))
    missing_sonar = sorted(required_sonar - set(sonar_data.columns))
    if missing_model or missing_sonar:
        details = []
        if missing_model:
            details.append(f"full_merged_dataset.csv missing {missing_model}")
        if missing_sonar:
            details.append(f"sonarqube_file_level.csv missing {missing_sonar}")
        raise ValueError("; ".join(details))

    model_data["_match_path"] = model_data["File"].map(normalize_path)
    sonar_path_column = "clean_path" if "clean_path" in sonar_data else "sonar_path"
    if sonar_path_column not in sonar_data:
        raise ValueError(
            "sonarqube_file_level.csv needs either clean_path or sonar_path."
        )
    sonar_data["_match_path"] = sonar_data[sonar_path_column].map(normalize_path)

    key_columns = ["repo", "_match_path"]
    if model_data.duplicated(key_columns).any():
        raise ValueError(
            "full_merged_dataset.csv has duplicate repo/path keys; refusing "
            "a many-to-many file merge."
        )
    if sonar_data.duplicated(key_columns).any():
        raise ValueError(
            "sonarqube_file_level.csv has duplicate repo/path keys; refusing "
            "a many-to-many file merge."
        )

    merged = model_data.merge(
        sonar_data,
        on=key_columns,
        how="inner",
        suffixes=("", "_sonar"),
        validate="one_to_one",
    )
    print(
        f"Matched {len(merged)} files of {len(model_data)} model files "
        f"and {len(sonar_data)} SonarQube files."
    )
    if merged.empty:
        raise ValueError(
            "No files matched. Check repo names and normalized file paths."
        )

    numeric_columns = set(STATIC_METRICS + HISTORY_METRICS + [
        "n-revs",
        "sonar_complexity",
        "sonar_cognitive_complexity",
        "sonar_code_smells",
        "sonar_bugs",
        "sonar_duplicated_lines_density",
        "sonar_ncloc",
    ])
    for column in numeric_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    # SonarQube omits issue-count values when no issues are reported.
    for column in ["sonar_code_smells", "sonar_bugs"]:
        merged[column] = merged[column].fillna(0)

    merged["sonar_smells_per_100_lines"] = np.where(
        merged["sonar_ncloc"] > 0,
        merged["sonar_code_smells"] / merged["sonar_ncloc"] * 100,
        np.nan,
    )
    lower = merged["n-revs"].quantile(0.33)
    upper = merged["n-revs"].quantile(0.66)
    merged["historical_change_label"] = np.select(
        [
            merged["n-revs"] <= lower,
            merged["n-revs"] <= upper,
        ],
        ["Low", "Medium"],
        default="High",
    )
    print(
        "Historical revision-count cutoffs: "
        f"Low <= {lower:.1f}, Medium <= {upper:.1f}, High > {upper:.1f}"
    )
    print(
        "Risk labels here mean historical revision activity; "
        "they are not bug labels."
    )
    return merged


def build_model() -> object:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            class_weight="balanced",
            random_state=42,
        ),
    )


def evaluate(data: pd.DataFrame) -> pd.DataFrame:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    target = data["historical_change_label"]
    fold_results: dict[str, list[dict[str, object]]] = {
        model_name: [] for model_name in FEATURE_SETS
    }

    for fold_index, (train_indices, test_indices) in enumerate(
        splitter.split(data, target), start=1
    ):
        train = data.iloc[train_indices]
        test = data.iloc[test_indices]
        y_train = target.iloc[train_indices]
        y_test = target.iloc[test_indices]

        for model_name, features in FEATURE_SETS.items():
            model = build_model()
            model.fit(train[features], y_train)
            predictions = model.predict(test[features])
            scores = f1_score(
                y_test,
                predictions,
                labels=LABELS,
                average=None,
                zero_division=0,
            )
            fold_results[model_name].append({
                "fold": fold_index,
                "f1_macro": float(np.mean(scores)),
                "f1_low": float(scores[0]),
                "f1_medium": float(scores[1]),
                "f1_high": float(scores[2]),
            })

    paired_deltas = np.array([
        full["f1_macro"] - base["f1_macro"]
        for base, full in zip(
            fold_results["d_existing_nine_features"],
            fold_results["e_existing_plus_sonarqube"],
        )
    ])
    mean_paired_delta = float(np.mean(paired_deltas))
    std_paired_delta = float(np.std(paired_deltas, ddof=1))
    improves_over_one_fold_std = mean_paired_delta > std_paired_delta

    rows: list[dict[str, object]] = []
    for model_name, folds in fold_results.items():
        for fold in folds:
            rows.append({
                "record_type": "fold",
                "feature_set": model_name,
                **fold,
                "f1_macro_mean": np.nan,
                "f1_macro_std": np.nan,
                "f1_low_mean": np.nan,
                "f1_low_std": np.nan,
                "f1_medium_mean": np.nan,
                "f1_medium_std": np.nan,
                "f1_high_mean": np.nan,
                "f1_high_std": np.nan,
                "paired_e_minus_d_f1_macro": (
                    paired_deltas[fold["fold"] - 1]
                    if model_name == "e_existing_plus_sonarqube" else np.nan
                ),
                "mean_paired_e_minus_d_f1_macro": np.nan,
                "std_paired_e_minus_d_f1_macro": np.nan,
                "improved_d_by_more_than_one_fold_std": np.nan,
            })

        fold_frame = pd.DataFrame(folds)
        rows.append({
            "record_type": "summary",
            "feature_set": model_name,
            "fold": 0,
            "f1_macro": np.nan,
            "f1_low": np.nan,
            "f1_medium": np.nan,
            "f1_high": np.nan,
            "f1_macro_mean": float(fold_frame["f1_macro"].mean()),
            "f1_macro_std": float(fold_frame["f1_macro"].std(ddof=1)),
            "f1_low_mean": float(fold_frame["f1_low"].mean()),
            "f1_low_std": float(fold_frame["f1_low"].std(ddof=1)),
            "f1_medium_mean": float(fold_frame["f1_medium"].mean()),
            "f1_medium_std": float(fold_frame["f1_medium"].std(ddof=1)),
            "f1_high_mean": float(fold_frame["f1_high"].mean()),
            "f1_high_std": float(fold_frame["f1_high"].std(ddof=1)),
            "paired_e_minus_d_f1_macro": np.nan,
            "mean_paired_e_minus_d_f1_macro": (
                mean_paired_delta
                if model_name == "e_existing_plus_sonarqube" else np.nan
            ),
            "std_paired_e_minus_d_f1_macro": (
                std_paired_delta
                if model_name == "e_existing_plus_sonarqube" else np.nan
            ),
            "improved_d_by_more_than_one_fold_std": (
                improves_over_one_fold_std
                if model_name == "e_existing_plus_sonarqube" else np.nan
            ),
        })

    results = pd.DataFrame(rows)
    summary = results[results["record_type"] == "summary"].set_index("feature_set")
    best_set = str(summary["f1_macro_mean"].idxmax())
    results["best_feature_set_by_mean_f1"] = best_set
    results["best_feature_set_label"] = FEATURE_SET_LABELS[best_set]
    results["feature_set_label"] = results["feature_set"].map(FEATURE_SET_LABELS)
    results["sonarqube_improved_d_by_more_than_one_fold_std"] = (
        improves_over_one_fold_std
    )
    return results


def save_chart(results: pd.DataFrame) -> None:
    summaries = results[results["record_type"] == "summary"].set_index(
        "feature_set"
    )
    ordered_names = list(FEATURE_SETS)
    means = summaries.loc[ordered_names, "f1_macro_mean"]
    deviations = summaries.loc[ordered_names, "f1_macro_std"]
    colors = [
        "#C62828" if name == summaries["f1_macro_mean"].idxmax() else "#2E74B5"
        for name in ordered_names
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(len(ordered_names))
    ax.bar(
        positions,
        means,
        yerr=deviations,
        capsize=5,
        color=colors,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [FEATURE_SET_LABELS[name] for name in ordered_names],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("Mean F1-macro (5 stratified folds)")
    ax.set_ylim(0, 1)
    ax.set_title("Historical Change Classification by Feature Set")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=160)
    plt.close(fig)


def main() -> None:
    data = load_matched_data()
    results = evaluate(data)
    results.to_csv(RESULTS_PATH, index=False)
    save_chart(results)

    summary = results[results["record_type"] == "summary"].sort_values(
        "f1_macro_mean", ascending=False
    )
    print("\nFive-fold summary (mean +/- fold standard deviation):")
    for row in summary.itertuples(index=False):
        print(
            f"{FEATURE_SET_LABELS[row.feature_set]}: "
            f"macro F1={row.f1_macro_mean:.3f} +/- {row.f1_macro_std:.3f}; "
            f"Low={row.f1_low_mean:.3f} +/- {row.f1_low_std:.3f}; "
            f"Medium={row.f1_medium_mean:.3f} +/- {row.f1_medium_std:.3f}; "
            f"High={row.f1_high_mean:.3f} +/- {row.f1_high_std:.3f}"
        )

    best = summary.iloc[0]
    comparison = results[
        (results["record_type"] == "summary")
        & (results["feature_set"] == "e_existing_plus_sonarqube")
    ].iloc[0]
    print(
        f"\nBest by mean macro F1: {FEATURE_SET_LABELS[best['feature_set']]} "
        f"({best['f1_macro_mean']:.3f})."
    )
    print(
        "SonarQube improved set D by more than one standard deviation of "
        "the paired fold-by-fold F1 improvements: "
        f"{'YES' if comparison['improved_d_by_more_than_one_fold_std'] else 'NO'} "
        f"(mean paired change "
        f"{comparison['mean_paired_e_minus_d_f1_macro']:+.3f}; "
        "paired-fold SD "
        f"{comparison['std_paired_e_minus_d_f1_macro']:.3f})."
    )
    print(f"\nSaved results: {RESULTS_PATH}")
    print(f"Saved chart: {CHART_PATH}")
    print(
        "Interpretation: labels are terciles of historical revision counts. "
        "This evaluates association/classification of historical activity, not "
        "future bugs. Missing SonarQube code-smell and bug counts are treated "
        "as zero reported issues; other missing metrics are median-imputed "
        "within each training fold."
    )


if __name__ == "__main__":
    main()
