import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from score_repo import FEATURES, STATIC_FEATURES, score_repository


def git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def commit_file(repository: Path, path: str, content: str, message: str) -> None:
    commit_files(repository, {path: content}, message)


def commit_files(repository: Path, files: dict[str, str], message: str) -> None:
    for path, content in files.items():
        target = repository / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    git(repository, "add", *files)
    git(repository, "commit", "-m", message)


def make_repository(path: Path) -> Path:
    path.mkdir()
    git(path, "init", "-q")
    git(path, "config", "user.name", "RiskLens Test Author")
    git(path, "config", "user.email", "risklens-test@example.invalid")
    commit_file(
        path,
        "app.py",
        "def calculate(value):\n    return value + 1\n",
        "add app",
    )
    commit_file(
        path,
        "tests/test_app.py",
        "def test_calculate():\n    assert True\n",
        "add test",
    )
    for revision in range(5):
        changes = {
            "app.py": (
                "def calculate(value):\n"
                f"    return value + {revision + 2}\n"
            ),
        }
        if revision == 0:
            changes["tests/test_app.py"] = (
                "def test_calculate():\n"
                "    assert True\n"
                "    assert calculate(1) == 2\n"
            )
        commit_files(path, changes, f"update app {revision}")
    commit_file(
        path,
        "quiet.py",
        "def quiet(value):\n    return value\n",
        "add quiet file",
    )
    return path


def write_model_artifacts(directory: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(42)
    sample_count = 60
    full_x = pd.DataFrame(
        rng.normal(size=(sample_count, len(FEATURES))),
        columns=FEATURES,
    )
    labels = np.tile(["Low", "Medium", "High"], sample_count // 3)
    full_model = RandomForestClassifier(
        n_estimators=12, max_depth=4, random_state=42
    ).fit(full_x, labels)
    static_model = RandomForestClassifier(
        n_estimators=12, max_depth=4, random_state=42
    ).fit(full_x[STATIC_FEATURES], labels)
    shared = {
        "cutoffs": {"low_max_revisions": 3.0, "medium_max_revisions": 8.0},
        "classes": ["Low", "Medium", "High"],
        "label_definition": "test fixture",
    }
    full_path = directory / "full.joblib"
    static_path = directory / "static.joblib"
    joblib.dump(
        {
            **shared,
            "model": full_model,
            "feature_order": FEATURES,
            "model_kind": "full",
        },
        full_path,
    )
    joblib.dump(
        {
            **shared,
            "model": static_model,
            "feature_order": STATIC_FEATURES,
            "model_kind": "static",
        },
        static_path,
    )
    return full_path, static_path


def test_score_repository_uses_full_and_static_models(tmp_path: Path) -> None:
    repository = make_repository(tmp_path / "team-repo")
    full_path, static_path = write_model_artifacts(tmp_path)

    report = score_repository(
        repository,
        model_path=full_path,
        static_model_path=static_path,
    )

    app = report.loc[report["File"] == "app.py"].iloc[0]
    quiet = report.loc[report["File"] == "quiet.py"].iloc[0]
    assert app["revisions_before"] == 6
    assert app["model_used"] == "Full nine-feature model"
    assert app["coupling_partner_count"] == 1
    assert app["avg_coupling_degree"] == 1
    assert quiet["revisions_before"] == 1
    assert quiet["used_static_model"]
    assert quiet["model_used"] == "Static-only model (fewer than 5 commits)"
    assert 0.0 <= app["probability_high"] <= 1.0
    assert 0.0 <= quiet["probability_high"] <= 1.0
    assert app["top_reason_1"]
    assert quiet["top_reason_1"]
    assert pd.notna(app["shap_num_authors"])
    assert pd.isna(quiet["shap_num_authors"])

    report_path = tmp_path / "risk_report.csv"
    report.to_csv(report_path, index=False)
    saved_report = pd.read_csv(report_path)
    assert set(saved_report["File"]) == {"app.py", "quiet.py", "tests/test_app.py"}


def test_exclude_tests_option_omits_test_paths(tmp_path: Path) -> None:
    repository = make_repository(tmp_path / "team-repo")
    full_path, static_path = write_model_artifacts(tmp_path)

    report = score_repository(
        repository,
        model_path=full_path,
        static_model_path=static_path,
        exclude_tests=True,
    )

    assert "tests/test_app.py" not in set(report["File"])
    assert set(report["File"]) == {"app.py", "quiet.py"}


def test_command_writes_a_local_csv_report(tmp_path: Path) -> None:
    repository = make_repository(tmp_path / "team-repo")
    full_path, static_path = write_model_artifacts(tmp_path)
    output_path = tmp_path / "risk_report.csv"
    script = Path(__file__).resolve().parents[1] / "score_repo.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo",
            str(repository),
            "--out",
            str(output_path),
            "--model",
            str(full_path),
            "--static-model",
            str(static_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Saved report:" in result.stdout
    assert output_path.is_file()
    assert set(pd.read_csv(output_path)["File"]) == {
        "app.py",
        "quiet.py",
        "tests/test_app.py",
    }
