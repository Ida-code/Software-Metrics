"""Score Python files in a local Git repository with trained RiskLens models."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import subprocess
import tokenize
from typing import Any

import joblib
import lizard
import numpy as np
import pandas as pd
import shap
from radon.metrics import mi_visit

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "risklens_model.joblib"
STATIC_MODEL_PATH = SCRIPT_DIR / "risklens_static_model.joblib"

FEATURES = [
    "avg_complexity",
    "max_complexity",
    "total_nloc",
    "function_count",
    "maintainability_index",
    "num_authors",
    "code_age_days",
    "coupling_partner_count",
    "avg_coupling_degree",
]
STATIC_FEATURES = [
    "avg_complexity",
    "max_complexity",
    "total_nloc",
    "function_count",
    "maintainability_index",
]
EXCLUDE_KEYWORDS = [
    "cacert", ".ai", ".lock", "idnadata", "uts46data",
    ".rst", ".md", ".yml", ".yaml", ".toml",
]
FEATURE_LABELS = {
    "avg_complexity": "Average Complexity",
    "max_complexity": "Maximum Complexity",
    "total_nloc": "Total Lines of Code",
    "function_count": "Function Count",
    "maintainability_index": "Maintainability Index",
    "num_authors": "Number of Authors",
    "code_age_days": "Code Age in Days",
    "coupling_partner_count": "Coupling Partner Count",
    "avg_coupling_degree": "Average Coupling Degree",
}


def normalize_path(path: str) -> str:
    return path.replace("\\", "/").removeprefix("./")


def is_source_file(path: str, exclude_tests: bool = False) -> bool:
    normalized = normalize_path(path)
    if not normalized.lower().endswith(".py"):
        return False
    if any(keyword in normalized.lower() for keyword in EXCLUDE_KEYWORDS):
        return False
    if exclude_tests:
        parts = PurePosixPath(normalized).parts
        filename = parts[-1].lower()
        if (
            any(part.lower() in {"test", "tests", "testing"} for part in parts[:-1])
            or filename.startswith("test_")
            or filename.endswith("_test.py")
        ):
            return False
    return True


def run_git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode:
        details = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"Git command failed in {repository}: {' '.join(arguments)}"
            + (f"\n{details}" if details else "")
        )
    return result.stdout


def tracked_python_files(repository: Path, exclude_tests: bool) -> list[str]:
    listing = run_git(
        repository, "ls-files", "--cached", "--others", "--exclude-standard"
    )
    files = sorted({
        normalize_path(path.strip())
        for path in listing.splitlines()
        if path.strip() and is_source_file(path.strip(), exclude_tests)
    })
    if not files:
        raise ValueError(
            f"No eligible Python source files found in repository {repository}."
        )
    return files


def extract_static_metrics(
    repository: Path, source_paths: list[str]
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for relative_path in source_paths:
        file_path = repository / Path(relative_path)
        try:
            with tokenize.open(file_path) as source_file:
                source = source_file.read()
        except (OSError, SyntaxError, UnicodeError) as error:
            raise RuntimeError(
                f"Cannot read Python source file {file_path}: {error}"
            ) from error

        analysis = lizard.analyze_file(str(file_path))
        complexities = [
            float(function.cyclomatic_complexity)
            for function in analysis.function_list
        ]
        metrics[relative_path] = {
            "avg_complexity": (
                float(np.mean(complexities)) if complexities else 0.0
            ),
            "max_complexity": (
                float(max(complexities)) if complexities else 0.0
            ),
            "total_nloc": float(
                sum(int(function.nloc) for function in analysis.function_list)
            ),
            "function_count": float(len(analysis.function_list)),
            "maintainability_index": float(mi_visit(source, True)),
        }
    return metrics


def extract_history_metrics(
    repository: Path, source_paths: list[str]
) -> dict[str, dict[str, float]]:
    from pydriller import Repository

    file_identity: dict[str, int] = {}
    file_stats: dict[int, dict[str, Any]] = {}
    pair_commit_counts: dict[tuple[int, int], int] = defaultdict(int)
    next_identity = 0

    def identity_for(path: str) -> int:
        nonlocal next_identity
        normalized = normalize_path(path)
        if normalized not in file_identity:
            file_identity[normalized] = next_identity
            file_stats[next_identity] = {
                "authors": set(),
                "first_date": None,
                "revisions": 0,
            }
            next_identity += 1
        return file_identity[normalized]

    current_file_identities = {
        path: identity_for(path) for path in source_paths
    }
    target_identities = set(current_file_identities.values())
    for commit in Repository(str(repository)).traverse_commits():
        commit_author = commit.author.name or commit.author.email or "unknown"
        commit_identities: set[int] = set()
        renames: list[tuple[str, str, int]] = []
        for modified_file in commit.modified_files:
            old_path = (
                normalize_path(modified_file.old_path)
                if modified_file.old_path else None
            )
            new_path = (
                normalize_path(modified_file.new_path)
                if modified_file.new_path else None
            )
            if old_path is None and new_path is None:
                continue
            resolved = file_identity.get(old_path) if old_path else None
            if resolved is None and new_path:
                resolved = file_identity.get(new_path)
            if resolved is None:
                resolved = identity_for(new_path or old_path or "")

            commit_identities.add(resolved)
            if old_path and new_path and old_path != new_path:
                renames.append((old_path, new_path, resolved))

        commit_date = commit.author_date
        if commit_date.tzinfo is None:
            commit_date = commit_date.replace(tzinfo=timezone.utc)
        for identity in commit_identities:
            stats = file_stats[identity]
            stats["authors"].add(commit_author)
            stats["revisions"] += 1
            if stats["first_date"] is None or commit_date < stats["first_date"]:
                stats["first_date"] = commit_date

        for old_path, new_path, identity in renames:
            file_identity[old_path] = identity
            file_identity[new_path] = identity

        pairs_in_commit: set[tuple[int, int]] = set()
        for target_identity in target_identities.intersection(commit_identities):
            for other_identity in commit_identities:
                if target_identity != other_identity:
                    pairs_in_commit.add(tuple(sorted((target_identity, other_identity))))
        for pair in pairs_in_commit:
            pair_commit_counts[pair] += 1

    now = datetime.now(timezone.utc)
    partner_counts: dict[int, list[int]] = defaultdict(list)
    for (left, right), cochange_commits in pair_commit_counts.items():
        partner_counts[left].append(cochange_commits)
        partner_counts[right].append(cochange_commits)

    history: dict[str, dict[str, float]] = {}
    for path, identity in current_file_identities.items():
        stats = file_stats[identity]
        first_date = stats["first_date"]
        age_days = (
            max(0.0, (now - first_date).total_seconds() / 86400)
            if first_date is not None else 0.0
        )
        partner_degrees = partner_counts[identity]
        history[path] = {
            "num_authors": float(len(stats["authors"])),
            "code_age_days": float(age_days),
            "revisions_before": float(stats["revisions"]),
            "coupling_partner_count": float(len(partner_degrees)),
            "avg_coupling_degree": (
                float(np.mean(partner_degrees)) if partner_degrees else 0.0
            ),
        }

    return history


def extract_repo_features(
    repository: Path, exclude_tests: bool = False
) -> pd.DataFrame:
    repository = repository.resolve()
    if not repository.is_dir():
        raise NotADirectoryError(f"Repository directory does not exist: {repository}")
    repository = Path(
        run_git(repository, "rev-parse", "--show-toplevel").strip()
    ).resolve()
    files = tracked_python_files(repository, exclude_tests)
    static_metrics = extract_static_metrics(repository, files)
    history_metrics = extract_history_metrics(repository, files)

    rows = []
    for path in files:
        row = {"File": path}
        row.update(static_metrics[path])
        row.update(history_metrics[path])
        rows.append(row)
    return pd.DataFrame(rows, columns=["File", *FEATURES, "revisions_before"])


def load_model_artifact(path: Path, expected_features: list[str]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {path}. Run `python master_analysis.py` first."
        )
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise ValueError(f"Model artifact has an unexpected format: {path}")
    if artifact.get("feature_order") != expected_features:
        raise ValueError(
            f"Feature order in {path.name} does not match the scorer. "
            "Regenerate model artifacts with `python master_analysis.py`."
        )
    required_cutoffs = {"low_max_revisions", "medium_max_revisions"}
    if not required_cutoffs.issubset(artifact.get("cutoffs", {})):
        raise ValueError(f"Training revision cutoffs are missing from {path.name}.")
    if not {"Low", "Medium", "High"}.issubset(set(artifact["model"].classes_)):
        raise ValueError(f"Model in {path.name} is missing one or more risk classes.")
    return artifact


def shap_for_high_class(model: Any, rows: pd.DataFrame) -> np.ndarray:
    values = shap.TreeExplainer(model).shap_values(rows)
    classes = list(model.classes_)
    high_index = classes.index("High")
    if isinstance(values, list):
        high_values = np.asarray(values[high_index])
    else:
        values = np.asarray(values)
        if values.ndim == 3:
            high_values = values[:, :, high_index]
        elif values.ndim == 2:
            high_values = (
                values if high_index == 1 else -values
            )
        else:
            raise ValueError(
                f"Unexpected SHAP output shape for High class: {values.shape}"
            )
    if high_values.shape != (len(rows), len(rows.columns)):
        raise ValueError(
            f"Unexpected High-class SHAP shape: {high_values.shape}"
        )
    return high_values


def score_repository(
    repository: Path,
    model_path: Path = MODEL_PATH,
    static_model_path: Path = STATIC_MODEL_PATH,
    exclude_tests: bool = False,
) -> pd.DataFrame:
    features = extract_repo_features(repository, exclude_tests)
    full_artifact = load_model_artifact(model_path, FEATURES)
    static_artifact = load_model_artifact(static_model_path, STATIC_FEATURES)
    full_model = full_artifact["model"]
    static_model = static_artifact["model"]
    cutoff = full_artifact["cutoffs"]

    full_mask = features["revisions_before"] >= 5
    used_full = features.loc[full_mask]
    used_static = features.loc[~full_mask]
    scored = features.copy()
    scored["predicted_class"] = ""
    scored["probability_high"] = 0.0
    scored["model_used"] = ""
    scored["used_static_model"] = ~full_mask
    for feature in FEATURES:
        scored[f"shap_{feature}"] = np.nan
    for rank in range(1, 4):
        scored[f"top_reason_{rank}"] = ""

    for subset, artifact, selected_model, feature_order in (
        (used_full, full_artifact, full_model, FEATURES),
        (used_static, static_artifact, static_model, STATIC_FEATURES),
    ):
        if subset.empty:
            continue
        model_rows = subset[feature_order].astype(float)
        predictions = selected_model.predict(model_rows)
        probabilities = selected_model.predict_proba(model_rows)
        high_index = list(selected_model.classes_).index("High")
        high_shap = shap_for_high_class(selected_model, model_rows)

        for position, index in enumerate(subset.index):
            scored.at[index, "predicted_class"] = str(predictions[position])
            scored.at[index, "probability_high"] = float(
                probabilities[position, high_index]
            )
            scored.at[index, "model_used"] = (
                "Full nine-feature model"
                if artifact["model_kind"] == "full"
                else "Static-only model (fewer than 5 commits)"
            )
            contributions = []
            for feature_position, feature in enumerate(feature_order):
                contribution = float(high_shap[position, feature_position])
                scored.at[index, f"shap_{feature}"] = contribution
                contributions.append(
                    (
                        abs(contribution),
                        feature,
                        float(model_rows.iloc[position][feature]),
                        contribution,
                    )
                )
            contributions.sort(reverse=True)
            for rank, (_, feature, value, contribution) in enumerate(
                contributions[:3], start=1
            ):
                direction = "up" if contribution >= 0 else "down"
                scored.at[index, f"top_reason_{rank}"] = (
                    f"{FEATURE_LABELS[feature]} (value {value:.3g}) "
                    f"pushes the High score {direction} "
                    f"({contribution:+.3f})."
                )

    scored["training_low_cutoff_revisions"] = float(
        cutoff["low_max_revisions"]
    )
    scored["training_medium_cutoff_revisions"] = float(
        cutoff["medium_max_revisions"]
    )
    return scored.sort_values(
        ["probability_high", "File"], ascending=[False, True]
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score Python files in a local Git repository for relative historical "
            "change-proneness."
        )
    )
    parser.add_argument("--repo", required=True, type=Path, help="Path to a local Git repository")
    parser.add_argument("--out", required=True, type=Path, help="Output CSV report path")
    parser.add_argument(
        "--exclude-tests",
        action="store_true",
        help="Exclude Python files under test directories and test_*/ *_test.py files.",
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help=argparse.SUPPRESS)
    parser.add_argument(
        "--static-model", type=Path, default=STATIC_MODEL_PATH, help=argparse.SUPPRESS
    )
    arguments = parser.parse_args()

    report = score_repository(
        arguments.repo,
        model_path=arguments.model,
        static_model_path=arguments.static_model,
        exclude_tests=arguments.exclude_tests,
    )
    output_path = arguments.out.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    static_count = int(report["used_static_model"].sum())
    print(
        f"Scored {len(report)} Python files ({static_count} with the static-only "
        f"model). Saved report: {output_path}"
    )
    print(
        "Probability scores are relative to the saved training models and "
        "cutoffs; they are not probabilities of bugs."
    )


if __name__ == "__main__":
    main()
