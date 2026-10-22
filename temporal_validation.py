"""
Temporal validation for historical file change-proneness.

For each repository, choose the commit at 70% of its first-parent history as
the cutoff. Build features from the repository state and history available at
that cutoff, then predict each existing file's revision count after the cutoff.

The script evaluates five feature sets with leave-one-repository-out GroupKFold
and a reproducible within-repository file split. It writes
forecast_validation_results.csv and forecast_validation_chart.png next to this
script. Existing project scripts and datasets are not modified.

Run from the project root:
    python temporal_validation.py
"""

from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lizard
import matplotlib
import numpy as np
import pandas as pd
from pydriller import Repository
from radon.metrics import mi_visit
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import make_pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tokenize


REPOSITORIES = ["requests", "flask", "click", "httpx", "rich"]
STATIC_FEATURES = [
    "avg_complexity_at_T",
    "max_complexity_at_T",
    "total_nloc_at_T",
    "function_count_at_T",
    "maintainability_index_at_T",
]
PRE_CUTOFF_FEATURES = [
    "revisions_before",
    "num_authors_before",
    "age_at_T",
    "churn_before",
]
FEATURE_SETS = {
    "naive_revisions_before": ["revisions_before"],
    "static_metrics_at_T": STATIC_FEATURES,
    "num_authors_before": ["num_authors_before"],
    "all_pre_T_features": PRE_CUTOFF_FEATURES + STATIC_FEATURES,
    "all_without_revisions_before": [
        feature for feature in PRE_CUTOFF_FEATURES + STATIC_FEATURES
        if feature != "revisions_before"
    ],
}
RESULTS_PATH = Path(__file__).resolve().parent / "forecast_validation_results.csv"
CHART_PATH = Path(__file__).resolve().parent / "forecast_validation_chart.png"


@dataclass(frozen=True)
class FileChange:
    old_path: str | None
    new_path: str | None
    added_lines: int
    deleted_lines: int


@dataclass(frozen=True)
class CommitRecord:
    commit_hash: str
    author: str
    author_date: datetime
    changes: tuple[FileChange, ...]


def run_git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.strip()


def normalize_path(path: str | None) -> str | None:
    if path is None:
        return None
    return path.replace("\\", "/").removeprefix("./")


def as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def get_git_history(repository: Path) -> tuple[list[str], set[str]]:
    first_parent = run_git(
        repository, "rev-list", "--first-parent", "--reverse", "HEAD"
    ).splitlines()
    all_commits = set(run_git(repository, "rev-list", "HEAD").splitlines())
    if len(first_parent) < 2:
        raise ValueError(
            f"{repository.name} has fewer than two first-parent commits; "
            "a 70/30 temporal split cannot be made."
        )
    return first_parent, all_commits


def load_pydriller_commits(repository: Path) -> dict[str, CommitRecord]:
    records: dict[str, CommitRecord] = {}
    for commit in Repository(str(repository)).traverse_commits():
        if commit.hash in records:
            continue
        changes = tuple(
            FileChange(
                old_path=normalize_path(modified.old_path),
                new_path=normalize_path(modified.new_path),
                added_lines=int(modified.added_lines or 0),
                deleted_lines=int(modified.deleted_lines or 0),
            )
            for modified in commit.modified_files
        )
        author = commit.author.email or commit.author.name or "unknown"
        records[commit.hash] = CommitRecord(
            commit_hash=commit.hash,
            author=author,
            author_date=as_utc(commit.author_date),
            changes=changes,
        )
    return records


def source_paths_at_commit(worktree: Path) -> list[str]:
    tracked = run_git(worktree, "ls-files", "--cached", "--full-name").splitlines()
    return sorted(
        path.replace("\\", "/")
        for path in tracked
        if path.lower().endswith(".py")
    )


def extract_static_metrics(worktree: Path, source_paths: list[str]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for relative_path in source_paths:
        file_path = worktree / Path(relative_path)
        try:
            with tokenize.open(file_path) as source_file:
                source = source_file.read()
        except (OSError, SyntaxError, UnicodeError) as error:
            raise RuntimeError(
                f"Cannot read Python file at cutoff: {file_path}: {error}"
            ) from error

        analysis = lizard.analyze_file(str(file_path))
        functions = analysis.function_list
        complexities = [
            float(function.cyclomatic_complexity) for function in functions
        ]
        metrics[relative_path] = {
            "avg_complexity_at_T": (
                float(np.mean(complexities)) if complexities else 0.0
            ),
            "max_complexity_at_T": (
                float(max(complexities)) if complexities else 0.0
            ),
            "total_nloc_at_T": float(
                sum(int(function.nloc) for function in functions)
            ),
            "function_count_at_T": float(len(functions)),
            "maintainability_index_at_T": float(mi_visit(source, True)),
        }
    return metrics


def compute_temporal_features(
    repository: Path,
    cutoff_hash: str,
    all_commits: set[str],
    commit_records: dict[str, CommitRecord],
) -> tuple[list[dict[str, Any]], datetime]:
    before_order = run_git(
        repository, "rev-list", "--reverse", "--topo-order", cutoff_hash
    ).splitlines()
    before_ids = set(before_order)
    future_order = [
        commit_hash
        for commit_hash in run_git(
            repository, "rev-list", "--reverse", "--topo-order", "HEAD"
        ).splitlines()
        if commit_hash not in before_ids
    ]
    if not before_ids.issubset(all_commits):
        raise RuntimeError(
            f"Git history for {repository.name} changed while validation ran."
        )
    missing_commit_data = (before_ids | set(future_order)) - commit_records.keys()
    if missing_commit_data:
        raise RuntimeError(
            f"PyDriller could not read {len(missing_commit_data)} commits "
            f"from {repository.name}; refusing an incomplete temporal analysis."
        )

    cutoff_date = commit_records[cutoff_hash].author_date
    file_stats: dict[str, dict[str, Any]] = {}
    path_owner: dict[str, str] = {}
    next_identity = 0

    def new_identity(path: str) -> str:
        nonlocal next_identity
        identity = f"{next_identity}:{path}"
        next_identity += 1
        file_stats[identity] = {
            "revisions_before": 0,
            "authors": set(),
            "first_date": None,
            "churn_before": 0,
            "future_revisions": 0,
        }
        return identity

    for commit_hash in before_order:
        commit = commit_records[commit_hash]
        changed_identities: set[str] = set()
        for change in commit.changes:
            old_path = change.old_path
            new_path = change.new_path
            candidate = old_path or new_path
            if candidate is None or not candidate.lower().endswith(".py"):
                continue

            identity = (
                path_owner.get(old_path) if old_path else None
            ) or (
                path_owner.get(new_path) if new_path else None
            )
            if identity is None:
                identity = new_identity(new_path or old_path or candidate)

            if identity not in changed_identities:
                stats = file_stats[identity]
                stats["revisions_before"] += 1
                stats["authors"].add(commit.author)
                stats["churn_before"] += change.added_lines + change.deleted_lines
                if stats["first_date"] is None:
                    stats["first_date"] = commit.author_date
                changed_identities.add(identity)

            if new_path:
                path_owner[new_path] = identity
            if old_path and old_path != new_path:
                path_owner.pop(old_path, None)

    with tempfile.TemporaryDirectory(
        prefix=f"temporal-validation-{repository.name}-"
    ) as temporary_directory:
        worktree = Path(temporary_directory) / "cutoff-worktree"
        run_git(
            repository,
            "worktree",
            "add",
            "--detach",
            "--quiet",
            str(worktree),
            cutoff_hash,
        )
        try:
            paths = source_paths_at_commit(worktree)
            static_metrics = extract_static_metrics(worktree, paths)
        finally:
            run_git(
                repository,
                "worktree",
                "remove",
                "--force",
                str(worktree),
            )

    snapshot_identity: dict[str, str] = {}
    for path in paths:
        identity = path_owner.get(path)
        if identity is None:
            identity = new_identity(path)
            path_owner[path] = identity
        snapshot_identity[path] = identity

    snapshot_identities = set(snapshot_identity.values())
    for commit_hash in future_order:
        commit = commit_records[commit_hash]
        changed_identities = set()
        for change in commit.changes:
            old_path = change.old_path
            new_path = change.new_path
            candidate = old_path or new_path
            if candidate is None or not candidate.lower().endswith(".py"):
                continue

            identity = (
                path_owner.get(old_path) if old_path else None
            ) or (
                path_owner.get(new_path) if new_path else None
            )
            if identity is not None and identity in snapshot_identities:
                if identity not in changed_identities:
                    file_stats[identity]["future_revisions"] += 1
                    changed_identities.add(identity)

            if identity is not None and new_path:
                path_owner[new_path] = identity
            if old_path and old_path != new_path:
                path_owner.pop(old_path, None)

    rows: list[dict[str, Any]] = []
    for path in paths:
        identity = snapshot_identity[path]
        stats = file_stats[identity]
        first_date = stats["first_date"] or cutoff_date
        rows.append({
            "repo": repository.name,
            "file": path,
            "cutoff_commit": cutoff_hash,
            "cutoff_date": cutoff_date.isoformat(),
            "revisions_before": int(stats["revisions_before"]),
            "num_authors_before": int(len(stats["authors"])),
            "age_at_T": float(
                max(0.0, (cutoff_date - first_date).total_seconds() / 86400)
            ),
            "churn_before": int(stats["churn_before"]),
            "future_revisions": int(stats["future_revisions"]),
            **static_metrics[path],
        })
    return rows, cutoff_date


def collect_repository_data(root: Path, cutoff_fraction: float) -> pd.DataFrame:
    repository_rows: list[dict[str, Any]] = []
    for repo_name in REPOSITORIES:
        repository = root / repo_name
        if not repository.is_dir():
            raise FileNotFoundError(f"Repository directory not found: {repository}")
        if run_git(repository, "rev-parse", "--is-inside-work-tree") != "true":
            raise ValueError(f"{repository} is not a Git working tree.")

        first_parent, all_commits = get_git_history(repository)
        cutoff_index = min(
            len(first_parent) - 1,
            max(1, math.floor(cutoff_fraction * (len(first_parent) - 1))),
        )
        cutoff_hash = first_parent[cutoff_index]
        commit_records = load_pydriller_commits(repository)
        missing = all_commits - commit_records.keys()
        if missing:
            raise RuntimeError(
                f"PyDriller missed {len(missing)} reachable commits in "
                f"{repo_name}; no temporal scores were written."
            )

        rows, cutoff_date = compute_temporal_features(
            repository, cutoff_hash, all_commits, commit_records
        )
        repository_rows.extend(rows)
        cutoff_ancestors = set(
            run_git(repository, "rev-list", cutoff_hash).splitlines()
        )
        print(
            f"{repo_name}: {len(rows)} Python files at T={cutoff_date.date()} "
            f"(first-parent commit {cutoff_index + 1}/{len(first_parent)}); "
            f"{len(all_commits - cutoff_ancestors)} commits after T"
        )
    data = pd.DataFrame(repository_rows)
    if data.empty:
        raise ValueError("No Python files were found at the temporal cutoffs.")
    return data


def assign_training_terciles(
    train_future_revisions: pd.Series,
    test_future_revisions: pd.Series,
) -> tuple[pd.Series, pd.Series, float, float]:
    lower = float(train_future_revisions.quantile(1 / 3))
    upper = float(train_future_revisions.quantile(2 / 3))

    def assign(values: pd.Series) -> pd.Series:
        return pd.Series(
            np.where(
                values <= lower,
                "Low",
                np.where(values <= upper, "Medium", "High"),
            ),
            index=values.index,
            dtype="object",
        )

    return assign(train_future_revisions), assign(test_future_revisions), lower, upper


def make_model(features: list[str]) -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            random_state=42,
            class_weight="balanced",
        ),
    )


def score_predictions(
    actual: pd.Series,
    predicted: np.ndarray,
    high_probability: np.ndarray,
    future_revisions: pd.Series,
) -> dict[str, float | int]:
    future_high = actual.eq("High").astype(int)
    ranking = pd.DataFrame({
        "probability": high_probability,
        "future_high": future_high.to_numpy(),
    }).sort_values("probability", ascending=False)
    top_count = max(1, math.ceil(0.10 * len(ranking)))
    rho, _ = spearmanr(high_probability, future_revisions)
    return {
        "files": int(len(actual)),
        "f1_macro": float(f1_score(
            actual, predicted, labels=["Low", "Medium", "High"],
            average="macro", zero_division=0,
        )),
        "spearman_high_probability_future_revisions": (
            float(rho) if pd.notna(rho) else float("nan")
        ),
        "precision_at_10_percent": float(
            ranking.head(top_count)["future_high"].mean()
        ),
        "accuracy": float(accuracy_score(actual, predicted)),
    }


def fit_and_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    y_train, y_test, lower, upper = assign_training_terciles(
        train["future_revisions"], test["future_revisions"]
    )
    model = make_model(features)
    model.fit(train[features], y_train)
    predictions = model.predict(test[features])
    probabilities = model.predict_proba(test[features])
    classes = list(model.named_steps["randomforestclassifier"].classes_)
    high_probability = (
        probabilities[:, classes.index("High")]
        if "High" in classes else np.zeros(len(test))
    )
    return predictions, high_probability, lower, upper


def evaluate_group_kfold(data: pd.DataFrame) -> list[dict[str, Any]]:
    n_groups = data["repo"].nunique()
    if n_groups < 2:
        raise ValueError("GroupKFold requires at least two repositories.")
    splitter = GroupKFold(n_splits=n_groups)
    result_rows: list[dict[str, Any]] = []
    prediction_cache: dict[
        str, dict[int, tuple[str, str, float, float, float]]
    ] = {
        name: {} for name in FEATURE_SETS
    }

    for fold_index, (train_indices, test_indices) in enumerate(
        splitter.split(data, groups=data["repo"]), start=1
    ):
        train = data.iloc[train_indices]
        test = data.iloc[test_indices]
        held_out_repos = sorted(test["repo"].unique())
        fold_name = ",".join(held_out_repos)
        for model_name, features in FEATURE_SETS.items():
            predictions, high_probability, lower, upper = fit_and_predict(
                train, test, features
            )
            _, actual, _, _ = assign_training_terciles(
                train["future_revisions"], test["future_revisions"]
            )
            metrics = score_predictions(
                actual, predictions, high_probability, test["future_revisions"]
            )
            result_rows.append({
                "validation": "group_kfold_by_repository",
                "fold": fold_name,
                "fold_index": fold_index,
                "model": model_name,
                **metrics,
                "train_future_revision_q33": lower,
                "train_future_revision_q66": upper,
            })
            for row_index, prediction, probability in zip(
                test_indices, predictions, high_probability
            ):
                prediction_cache[model_name][int(row_index)] = (
                    str(actual.loc[row_index]),
                    str(prediction),
                    float(probability),
                    lower,
                    upper,
                )

    for model_name in FEATURE_SETS:
        cached = prediction_cache[model_name]
        indices = sorted(cached)
        actual = pd.Series(
            [cached[index][0] for index in indices], index=indices
        )
        predictions = np.array([cached[index][1] for index in indices])
        probabilities = np.array([cached[index][2] for index in indices])
        metrics = score_predictions(
            actual, predictions, probabilities, data.loc[indices, "future_revisions"]
        )
        result_rows.append({
            "validation": "group_kfold_by_repository",
            "fold": "all_out_of_fold_predictions",
            "fold_index": 0,
            "model": model_name,
            **metrics,
            "train_future_revision_q33": np.nan,
            "train_future_revision_q66": np.nan,
        })
    return result_rows


def evaluate_within_repository(data: pd.DataFrame, seed: int) -> list[dict[str, Any]]:
    rng = np.random.RandomState(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for _, group in data.groupby("repo", sort=True):
        indices = group.index.to_numpy()
        shuffled = rng.permutation(indices)
        test_count = max(1, math.ceil(0.25 * len(shuffled)))
        test_indices.extend(int(index) for index in shuffled[:test_count])
        train_indices.extend(int(index) for index in shuffled[test_count:])

    train = data.loc[train_indices]
    test = data.loc[test_indices]
    result_rows = []
    for model_name, features in FEATURE_SETS.items():
        predictions, high_probability, lower, upper = fit_and_predict(
            train, test, features
        )
        _, actual, _, _ = assign_training_terciles(
            train["future_revisions"], test["future_revisions"]
        )
        metrics = score_predictions(
            actual, predictions, high_probability, test["future_revisions"]
        )
        result_rows.append({
            "validation": "within_repository_file_split",
            "fold": "all_repositories",
            "fold_index": 0,
            "model": model_name,
            **metrics,
            "train_future_revision_q33": lower,
            "train_future_revision_q66": upper,
        })
    return result_rows


def add_naive_comparison(results: pd.DataFrame) -> pd.DataFrame:
    results["naive_baseline_beats_full_model"] = False
    summary_mask = results["fold"].isin(
        ["all_out_of_fold_predictions", "all_repositories"]
    )
    summaries = results.loc[summary_mask]
    for validation, group in summaries.groupby("validation"):
        values = group.set_index("model")["f1_macro"]
        naive = values["naive_revisions_before"]
        full = values["all_pre_T_features"]
        results.loc[
            (results["validation"] == validation) & summary_mask,
            "naive_baseline_beats_full_model",
        ] = naive > full
    return results


def create_chart(results: pd.DataFrame) -> None:
    summaries = results[
        results["fold"].isin(["all_out_of_fold_predictions", "all_repositories"])
    ].copy()
    validations = [
        "group_kfold_by_repository",
        "within_repository_file_split",
    ]
    model_names = list(FEATURE_SETS)
    labels = {
        "naive_revisions_before": "Revisions before T",
        "static_metrics_at_T": "Static metrics at T",
        "num_authors_before": "Authors before T",
        "all_pre_T_features": "All pre-T features",
        "all_without_revisions_before": "All except revisions",
    }

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    x = np.arange(len(model_names))
    width = 0.36
    group_summary = summaries[
        summaries["validation"] == "group_kfold_by_repository"
    ].set_index("model")["f1_macro"]
    naive_wins = (
        group_summary["naive_revisions_before"]
        > group_summary["all_pre_T_features"]
    )
    metric_specs = [
        ("f1_macro", "Macro F1", (0, 1)),
        (
            "spearman_high_probability_future_revisions",
            "Spearman: High probability vs future revisions",
            (-1, 1),
        ),
        ("precision_at_10_percent", "Precision@10%", (0, 1)),
    ]
    for ax, (metric, title, limits) in zip(axes, metric_specs):
        for offset, validation in zip([-width / 2, width / 2], validations):
            values = (
                summaries[summaries["validation"] == validation]
                .set_index("model")
                .reindex(model_names)[metric]
            )
            ax.bar(
                x + offset,
                values,
                width,
                label=(
                    "Leave-one-repository-out"
                    if validation == "group_kfold_by_repository"
                    else "Within-repository split"
                ),
            )
        ax.set_ylabel(title)
        ax.set_ylim(*limits)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [labels[name] for name in model_names], rotation=22, ha="right"
        )
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Temporal Change-Proneness Forecast Validation — "
        "Naive revisions-only baseline beats full model: "
        f"{'YES' if naive_wins else 'NO'} (repository-held-out F1)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(CHART_PATH, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Folder containing the requests, flask, click, httpx, and rich clones.",
    )
    parser.add_argument(
        "--cutoff-fraction",
        type=float,
        default=0.70,
        help="Position of T on each repository's first-parent history (default: 0.70).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the within-repository split.",
    )
    args = parser.parse_args()
    if not 0.5 < args.cutoff_fraction < 1:
        parser.error("--cutoff-fraction must be greater than 0.5 and less than 1.")

    data = collect_repository_data(args.root.resolve(), args.cutoff_fraction)
    print(
        f"\nBuilt {len(data)} cutoff-snapshot file rows. "
        f"Future revision counts range from {data['future_revisions'].min()} "
        f"to {data['future_revisions'].max()}."
    )
    results = evaluate_group_kfold(data)
    results.extend(evaluate_within_repository(data, args.seed))
    results_frame = add_naive_comparison(pd.DataFrame(results))
    cutoff_dates = "; ".join(
        f"{repo}={group['cutoff_date'].iloc[0]}"
        for repo, group in data.groupby("repo", sort=True)
    )
    results_frame["cutoff_dates_utc"] = cutoff_dates
    results_frame.to_csv(RESULTS_PATH, index=False)
    create_chart(results_frame)

    summaries = results_frame[
        results_frame["fold"].isin(["all_out_of_fold_predictions", "all_repositories"])
    ]
    print("\nSummary results:")
    print(
        summaries[
            [
                "validation", "model", "files", "f1_macro",
                "spearman_high_probability_future_revisions",
                "precision_at_10_percent",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    for validation, group in summaries.groupby("validation"):
        scores = group.set_index("model")["f1_macro"]
        naive_beats_full = (
            scores["naive_revisions_before"] > scores["all_pre_T_features"]
        )
        print(
            f"\nNaive revisions-before-T baseline beats the full pre-T model "
            f"for {validation}: {'YES' if naive_beats_full else 'NO'} "
            f"(F1 {scores['naive_revisions_before']:.3f} vs "
            f"{scores['all_pre_T_features']:.3f})."
        )
    print(f"\nSaved results: {RESULTS_PATH}")
    print(f"Saved chart: {CHART_PATH}")
    print(
        "\nInterpretation: this forecasts post-cutoff revision activity, not bugs. "
        "Tercile thresholds are learned from the training fold only. GroupKFold "
        "tests transfer to a repository absent from training; within-repository "
        "testing measures interpolation among files from repositories represented "
        "in training."
    )


if __name__ == "__main__":
    main()
