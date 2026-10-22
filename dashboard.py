"""
RiskLens Dashboard - Historical Change-Proneness Explorer

An interactive dashboard that:
1. Loads your merged multi-repo dataset (full_merged_dataset.csv)
2. Trains a Random Forest to classify historical revision activity
3. Lets you pick any file from the dataset (or enter custom metrics) and see:
   - Its predicted historical change-proneness (Low / Medium / High)
   - A SHAP chart showing feature contributions to the model's prediction
4. Shows overall model performance and feature importance for context
5. Scores local repositories through the saved full and static-only models

HOW TO USE:
1. Place this script in the SAME folder as full_merged_dataset.csv
   (the main folder: C:\\Users\\idash\\SoftwareMetricsProject\\requests\\)
2. Install Streamlit once:
   pip install streamlit
3. Run it (note: "streamlit run", NOT "python"):
   streamlit run dashboard.py
4. It will open automatically in your browser at http://localhost:8501
5. For the Robustness Checks tab, first run:
   python feature_robustness_check.py
6. For the Cross-Repository Evaluation tab, run:
   python master_analysis.py
7. For the SonarQube Comparison tab, start SonarQube and scan the projects, then run:
   python sonarqube_file_comparison.py
   Or reuse downloaded SonarQube measures with:
   python sonarqube_file_comparison.py --use-cached
8. For the model-comparison section in the SonarQube tab, run:
   python sonarqube_vs_model.py
9. To score a team's own local Git repository, first run `python master_analysis.py`
   to save the trained model artifacts, then run:
   python score_repo.py --repo "C:\\path\\to\\repo" --out risk_report.csv
   Open the Score a New Repo tab to view it.
"""

import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR / "full_merged_dataset.csv"
MASTER_RESULTS_PATH = SCRIPT_DIR / "master_results.json"
BAYESIAN_RESULTS_PATH = SCRIPT_DIR / "bayesian_results.json"
BAYESIAN_GRAPH_PATH = SCRIPT_DIR / "bayesian_network_graph.png"
ROBUSTNESS_RESULTS_PATH = SCRIPT_DIR / "feature_robustness_results.json"
SONAR_COMPARISON_PATH = SCRIPT_DIR / "file_level_comparison.csv"
GENERALIZATION_RESULTS_PATH = SCRIPT_DIR / "generalization_results.json"
SONAR_MODEL_RESULTS_PATH = SCRIPT_DIR / "sonarqube_vs_model_results.csv"
RISK_REPORT_PATH = SCRIPT_DIR / "risk_report.csv"

st.set_page_config(page_title="RiskLens - Change-Proneness Analysis", layout="wide")

FEATURES = ["avg_complexity", "max_complexity", "total_nloc", "function_count",
            "maintainability_index", "num_authors", "code_age_days",
            "coupling_partner_count", "avg_coupling_degree"]

FEATURE_LABELS = {
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


@st.cache_data
def load_data():
    df = pd.read_csv(DATASET_PATH)

    # Safety: if this CSV was generated before coupling features were added,
    # create them as zero-filled columns so the dashboard doesn't crash.
    for col in ["coupling_partner_count", "avg_coupling_degree"]:
        if col not in df.columns:
            df[col] = 0.0

    df = df.dropna(subset=FEATURES + ["risk_label"]) if "risk_label" in df.columns else df.dropna(subset=FEATURES)

    if "risk_label" not in df.columns:
        q1 = df["n-revs"].quantile(0.33)
        q2 = df["n-revs"].quantile(0.66)

        def label_risk(revs):
            if revs <= q1:
                return "Low"
            elif revs <= q2:
                return "Medium"
            return "High"

        df["risk_label"] = df["n-revs"].apply(label_risk)

    return df


@st.cache_resource
def train_model(df):
    X = df[FEATURES]
    y = df["risk_label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=200, max_depth=6, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, output_dict=False)
    explainer = shap.TreeExplainer(model)
    return model, explainer, acc, f1, report, X_test


def render_sonarqube_model_comparison():
    st.subheader("Does adding SonarQube measurements improve classification?")
    st.write(
        "This compares five sets of measurements using the same files and "
        "five data splits. The model is classifying historical revision "
        "activity, not bugs."
    )
    if not SONAR_MODEL_RESULTS_PATH.exists():
        st.warning(
            "Model comparison results are missing. Run "
            "`python sonarqube_vs_model.py` from the project folder, then "
            "refresh the dashboard."
        )
        return

    comparison_results = pd.read_csv(SONAR_MODEL_RESULTS_PATH)
    required_columns = {
        "record_type",
        "feature_set",
        "feature_set_label",
        "f1_macro",
        "f1_macro_mean",
        "f1_macro_std",
        "f1_low_mean",
        "f1_low_std",
        "f1_medium_mean",
        "f1_medium_std",
        "f1_high_mean",
        "f1_high_std",
        "paired_e_minus_d_f1_macro",
        "mean_paired_e_minus_d_f1_macro",
        "std_paired_e_minus_d_f1_macro",
        "improved_d_by_more_than_one_fold_std",
    }
    missing_columns = sorted(required_columns - set(comparison_results.columns))
    if missing_columns:
        st.error(
            "The SonarQube model-comparison CSV is missing required columns: "
            + ", ".join(missing_columns)
            + ". Re-run `python sonarqube_vs_model.py`."
        )
        return

    summary_results = comparison_results[
        comparison_results["record_type"] == "summary"
    ].copy()
    fold_results = comparison_results[
        (comparison_results["record_type"] == "fold")
        & comparison_results["feature_set"].isin([
            "d_existing_nine_features",
            "e_existing_plus_sonarqube",
        ])
    ].copy()
    if summary_results.empty:
        st.warning(
            "The comparison CSV has no summary rows. Run "
            "`python sonarqube_vs_model.py` to regenerate it."
        )
        return

    summary_results = summary_results.sort_values(
        "f1_macro_mean", ascending=False
    )
    best_set = summary_results.iloc[0]
    sonar_added_set = summary_results[
        summary_results["feature_set"] == "e_existing_plus_sonarqube"
    ]
    if sonar_added_set.empty:
        st.warning(
            "The comparison CSV does not contain the combined "
            "nine-features-plus-SonarQube result."
        )
        return

    sonar_added_set = sonar_added_set.iloc[0]
    improved_by_more_than_std = bool(
        sonar_added_set["improved_d_by_more_than_one_fold_std"]
    )
    summary_columns = st.columns(3)
    summary_columns[0].metric(
        "Best feature set",
        str(best_set["feature_set_label"]),
        help="Selected from the CSV by highest mean five-fold macro F1.",
    )
    summary_columns[1].metric(
        "Best mean macro F1",
        f"{best_set['f1_macro_mean']:.3f}",
        delta=f"± {best_set['f1_macro_std']:.3f} fold SD",
    )
    summary_columns[2].metric(
        "SonarQube improves existing features by > 1 paired-fold SD",
        "Yes" if improved_by_more_than_std else "No",
        delta=(
            f"{sonar_added_set['mean_paired_e_minus_d_f1_macro']:+.3f} "
            "mean paired F1 change"
        ),
        delta_color="normal" if improved_by_more_than_std else "inverse",
    )

    st.markdown("#### Macro and per-class F1 across folds")
    display_summary = summary_results[
        [
            "feature_set_label",
            "f1_macro_mean",
            "f1_macro_std",
            "f1_low_mean",
            "f1_low_std",
            "f1_medium_mean",
            "f1_medium_std",
            "f1_high_mean",
            "f1_high_std",
        ]
    ].rename(
        columns={
            "feature_set_label": "Feature set",
            "f1_macro_mean": "Macro F1 mean",
            "f1_macro_std": "Macro F1 fold SD",
            "f1_low_mean": "Low F1 mean",
            "f1_low_std": "Low F1 fold SD",
            "f1_medium_mean": "Medium F1 mean",
            "f1_medium_std": "Medium F1 fold SD",
            "f1_high_mean": "High F1 mean",
            "f1_high_std": "High F1 fold SD",
        }
    )
    f1_columns = [
        column for column in display_summary.columns
        if column != "Feature set"
    ]
    st.dataframe(
        display_summary.style.format(
            {column: "{:.3f}" for column in f1_columns}
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Mean macro F1 by feature set")
    fig_sonar_models, ax_sonar_models = plt.subplots(figsize=(9, 5))
    ax_sonar_models.bar(
        summary_results["feature_set_label"],
        summary_results["f1_macro_mean"],
        yerr=summary_results["f1_macro_std"],
        capsize=4,
        color="#2E74B5",
    )
    ax_sonar_models.set_ylabel("Mean five-fold macro F1")
    ax_sonar_models.set_ylim(0, 1)
    ax_sonar_models.tick_params(axis="x", rotation=20)
    ax_sonar_models.grid(axis="y", alpha=0.25)
    fig_sonar_models.tight_layout()
    st.pyplot(fig_sonar_models)

    st.markdown("#### Paired fold-by-fold comparison: existing features vs. plus SonarQube")
    d_folds = fold_results[
        fold_results["feature_set"] == "d_existing_nine_features"
    ][["fold", "f1_macro"]].rename(
        columns={"f1_macro": "Existing nine features"}
    )
    e_folds = fold_results[
        fold_results["feature_set"] == "e_existing_plus_sonarqube"
    ][["fold", "f1_macro", "paired_e_minus_d_f1_macro"]].rename(
        columns={
            "f1_macro": "Nine features + SonarQube",
            "paired_e_minus_d_f1_macro": "Change (plus SonarQube)",
        }
    )
    paired_folds = d_folds.merge(
        e_folds, on="fold", validate="one_to_one"
    )
    st.dataframe(
        paired_folds.style.format(
            {
                "Existing nine features": "{:.3f}",
                "Nine features + SonarQube": "{:.3f}",
                "Change (plus SonarQube)": "{:+.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "A positive change means adding SonarQube raised macro F1 in that "
        "split; a negative change means it lowered F1. The summary check "
        "compares the mean paired change with its fold-to-fold variation. "
        "This is descriptive, not a formal statistical test."
    )
    st.info(
        "The target is historical revision activity. This comparison shows "
        "whether SonarQube measurements help classify those categories in "
        "this dataset; it does not test whether either system detects bugs."
    )


# ---------- Load data and train model ----------

st.title("RiskLens")

try:
    data = load_data()
except FileNotFoundError:
    st.error(
        "Could not find full_merged_dataset.csv. "
        "Make sure it is in the same folder as dashboard.py, "
        "and that you've already run master_analysis.py at least once."
    )
    st.stop()

master_results = (
    json.loads(MASTER_RESULTS_PATH.read_text(encoding="utf-8"))
    if MASTER_RESULTS_PATH.exists()
    else None
)

model, explainer, acc, f1, report, X_test = train_model(data)

# ---------- Sidebar: model performance summary ----------

with st.sidebar:
    st.header("Model Overview")
    st.metric("Files in dataset", len(data))
    st.metric("Random-split test accuracy", f"{acc:.1%}")
    st.metric("Random-split test F1-macro", f"{f1:.3f}")

# ---------- Main layout: two tabs ----------

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "🔍 Predict a File's Change Activity", "⚖️ Compare Two Files",
    "🏆 High Change-Proneness Leaderboard",
    "📊 Model Insights", "📁 Explore Dataset", "🕸️ Bayesian Network",
    "🧪 Robustness Checks", "🔎 SonarQube Comparison",
    "🌐 Cross-Repository Evaluation", "🗂️ Score a New Repo"
])

# ===================== TAB 1: Predict =====================
with tab1:
    st.caption(
        "Interactive predictions use the random-file-split model for exploration. "
        "For out-of-repository predictions, see the Cross-Repository Evaluation tab."
    )
    st.subheader("Choose how to test the model")
    mode = st.radio(
        "Mode",
        ["Pick an existing file from the dataset", "Enter custom metric values"],
        horizontal=True,
    )

    if mode == "Pick an existing file from the dataset":
        repo_choice = st.selectbox("Repository", sorted(data["repo"].unique()))
        repo_files = data[data["repo"] == repo_choice].sort_values("File")
        file_choice = st.selectbox("File", repo_files["File"].tolist())
        row = repo_files[repo_files["File"] == file_choice].iloc[0]
        input_values = {f: row[f] for f in FEATURES}
        actual_label = row["risk_label"]
        actual_revs = row["n-revs"]
    else:
        st.write("Drag the sliders to see how each metric affects the predicted risk in real time:")
        cols = st.columns(2)
        input_values = {}
        slider_ranges = {
            "avg_complexity": (1.0, 30.0),
            "max_complexity": (1.0, 60.0),
            "total_nloc": (10.0, 3000.0),
            "function_count": (1.0, 150.0),
            "maintainability_index": (0.0, 100.0),
            "num_authors": (1.0, 60.0),
            "code_age_days": (0.0, 6000.0),
            "coupling_partner_count": (0.0, 30.0),
            "avg_coupling_degree": (0.0, 100.0),
        }
        input_values = {}
        for i, feat in enumerate(FEATURES):
            col = cols[i % 2]
            lo, hi = slider_ranges[feat]
            default_val = float(np.clip(data[feat].median(), lo, hi))
            step = 1.0 if hi - lo > 20 else 0.5
            input_values[feat] = col.slider(FEATURE_LABELS[feat], min_value=lo, max_value=hi,
                                             value=default_val, step=step, key=f"slider_{feat}")
        actual_label = None
        actual_revs = None
        if master_results and master_results["feature_importance"]:
            strongest_feature = master_results["feature_importance"][0]["feature"]
            st.caption(
                f"Tip: try adjusting {FEATURE_LABELS[strongest_feature]} while keeping "
                "everything else fixed - it currently ranks highest in the master analysis."
            )
        else:
            st.caption("Adjust one metric at a time to see how it affects the prediction.")

    predict_clicked = st.button("Predict Risk", type="primary") if mode != "Enter custom metric values" else True

    if predict_clicked:
        input_df = pd.DataFrame([input_values])[FEATURES]
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        class_order = model.classes_

        color_map = {"Low": "green", "Medium": "orange", "High": "red"}
        st.markdown(f"### Predicted Risk: :{color_map.get(prediction,'blue')}[{prediction}]")

        if actual_label is not None:
            match = "✅ matches" if actual_label == prediction else "⚠️ differs from"
            st.write(f"Actual historical label for this file: **{actual_label}** "
                     f"({int(actual_revs)} real revisions) - model prediction {match} the real label.")

        prob_df = pd.DataFrame({"Risk Level": class_order, "Probability": probabilities})
        probability_fig, probability_ax = plt.subplots(figsize=(6, 3))
        probability_ax.bar(
            prob_df["Risk Level"],
            prob_df["Probability"],
            color=[color_map.get(label, "steelblue") for label in prob_df["Risk Level"]],
        )
        probability_ax.set_ylim(0, 1)
        probability_ax.set_ylabel("Predicted probability")
        probability_ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda value, _: f"{value:.0%}")
        )
        probability_ax.grid(axis="y", alpha=0.25)
        probability_fig.tight_layout()
        st.pyplot(probability_fig)

        st.markdown("#### Feature contributions to the predicted class")
        shap_values = explainer.shap_values(input_df)
        class_index = list(model.classes_).index(prediction)

        fig, ax = plt.subplots(figsize=(8, 4))
        shap_row = shap_values[0, :, class_index]
        feature_names = [FEATURE_LABELS[f] for f in FEATURES]
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in shap_row]
        order = np.argsort(np.abs(shap_row))
        ax.barh(
            [feature_names[i] for i in order],
            [shap_row[i] for i in order],
            color=[colors[i] for i in order],
        )
        ax.set_xlabel(f"Impact on prediction of '{prediction}' risk")
        ax.axvline(0, color="black", linewidth=0.8)
        st.pyplot(fig)
        st.caption(
            "Red bars increase the model's score for this class; blue bars "
            "decrease it. These are model contributions, not causes."
        )

# ===================== TAB 2: Compare Two Files =====================
with tab2:
    st.subheader("Compare Two Files Side by Side")
    st.write("Pick any two files - ideally one you expect to be low risk and one high risk - "
             "and see exactly why the model treats them differently.")

    col_a, col_b = st.columns(2)
    with col_a:
        repo_a = st.selectbox("Repository (File A)", sorted(data["repo"].unique()), key="repo_a")
        files_a = data[data["repo"] == repo_a].sort_values("File")
        file_a = st.selectbox("File A", files_a["File"].tolist(), key="file_a")
    with col_b:
        repo_b = st.selectbox("Repository (File B)", sorted(data["repo"].unique()), key="repo_b")
        files_b = data[data["repo"] == repo_b].sort_values("File")
        default_index_b = min(1, len(files_b) - 1)
        file_b = st.selectbox("File B", files_b["File"].tolist(), index=default_index_b, key="file_b")

    if st.button("Compare", type="primary"):
        row_a = files_a[files_a["File"] == file_a].iloc[0]
        row_b = files_b[files_b["File"] == file_b].iloc[0]

        input_a = pd.DataFrame([{f: row_a[f] for f in FEATURES}])[FEATURES]
        input_b = pd.DataFrame([{f: row_b[f] for f in FEATURES}])[FEATURES]

        pred_a = model.predict(input_a)[0]
        pred_b = model.predict(input_b)[0]

        color_map = {"Low": "green", "Medium": "orange", "High": "red"}

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.markdown(f"**{file_a}**")
            st.markdown(f"### :{color_map.get(pred_a,'blue')}[{pred_a}]")
            st.caption(f"Actual: {row_a['risk_label']} ({int(row_a['n-revs'])} revisions)")
        with res_col2:
            st.markdown(f"**{file_b}**")
            st.markdown(f"### :{color_map.get(pred_b,'blue')}[{pred_b}]")
            st.caption(f"Actual: {row_b['risk_label']} ({int(row_b['n-revs'])} revisions)")

        st.markdown("#### Metric-by-Metric Comparison")
        compare_df = pd.DataFrame({
            "Metric": [FEATURE_LABELS[f] for f in FEATURES],
            file_a: [row_a[f] for f in FEATURES],
            file_b: [row_b[f] for f in FEATURES],
        })
        st.dataframe(compare_df.set_index("Metric"), use_container_width=True)

        st.markdown("#### Feature contributions to each predicted class")
        shap_a = explainer.shap_values(input_a)
        shap_b = explainer.shap_values(input_b)
        idx_a = list(model.classes_).index(pred_a)
        idx_b = list(model.classes_).index(pred_b)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, shap_vals, pred, name in [
            (axes[0], shap_a[0, :, idx_a], pred_a, file_a),
            (axes[1], shap_b[0, :, idx_b], pred_b, file_b),
        ]:
            feature_names = [FEATURE_LABELS[f] for f in FEATURES]
            order = np.argsort(np.abs(shap_vals))
            colors = ["#d62728" if v > 0 else "#1f77b4" for v in shap_vals]
            ax.barh([feature_names[i] for i in order], [shap_vals[i] for i in order],
                    color=[colors[i] for i in order])
            ax.set_title(f"{name}\n→ {pred}", fontsize=10)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("SHAP contribution to predicted-class score")
        plt.tight_layout()
        st.pyplot(fig)
        st.caption(
            "Positive values increase the model's score for the predicted class; "
            "negative values decrease it. These are model contributions, not causes."
        )

# ===================== TAB 3: Leaderboard =====================
with tab3:
    st.subheader("Files most likely to have high historical change-proneness")
    holdout_columns = {
        "repo_holdout_prediction",
        "repo_holdout_high_probability",
    }
    if not holdout_columns.issubset(data.columns):
        st.warning("Run `python master_analysis.py` to generate held-out predictions.")
    else:
        st.caption(
            "Every score below comes from a model trained without that file's "
            "repository. It describes historical revision activity, not future bugs."
        )
        leaderboard = data.sort_values(
            "repo_holdout_high_probability", ascending=False
        ).head(15)
        st.dataframe(
            leaderboard[
                [
                    "repo", "File", "repo_holdout_prediction",
                    "repo_holdout_high_probability", "risk_label", "n-revs",
                    "num_authors", "max_complexity",
                ]
            ].rename(
                columns={
                    "repo_holdout_prediction": "Held-out prediction",
                    "repo_holdout_high_probability": "High-change probability",
                    "risk_label": "Historical category",
                    "n-revs": "Historical revisions",
                    "num_authors": "Distinct authors",
                }
            ).style.format({"High-change probability": "{:.1%}"}),
            use_container_width=True,
        )

        st.markdown("#### Average held-out high-change probability by repository")
        repo_avg = data.groupby("repo")[
            "repo_holdout_high_probability"
        ].mean().sort_values(ascending=False)
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ax3.bar(repo_avg.index, repo_avg.values, color="#C00000")
        ax3.set_ylabel("Mean held-out high-change probability")
        ax3.set_ylim(0, 1)
        plt.tight_layout()
        st.pyplot(fig3)

# ===================== TAB 4: Model Insights =====================
with tab4:
    st.subheader("Overall Model Performance")
    st.text(report)

    st.subheader("Which Metrics Matter Most Overall?")
    if master_results:
        importance_rows = master_results["feature_importance"]
        importance_by_feature = {
            row["feature"]: row["importance"] for row in importance_rows
        }
        importances = pd.Series(importance_by_feature).sort_values()
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.barh(
            [FEATURE_LABELS[feature] for feature in importances.index],
            importances.values,
            color="#2E74B5",
        )
        ax2.set_xlabel("Importance")
        st.pyplot(fig2)
        strongest_features = [
            FEATURE_LABELS[row["feature"]]
            for row in master_results["feature_importance"][:2]
        ]
        st.info(
            f"Master analysis feature-importance ranking: "
            f"{', '.join(strongest_features)} currently rank highest."
        )
    else:
        st.warning("master_results.json not found. Run master_analysis.py to generate it.")

# ===================== TAB 5: Explore Dataset =====================
with tab5:
    st.subheader("Full Merged Dataset")
    repo_filter = st.multiselect("Filter by repo", sorted(data["repo"].unique()), default=list(data["repo"].unique()))
    risk_filter = st.multiselect("Filter by risk label", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])
    filtered = data[data["repo"].isin(repo_filter) & data["risk_label"].isin(risk_filter)]
    st.dataframe(filtered[["repo", "File", "risk_label", "n-revs"] + FEATURES], use_container_width=True)
    st.caption(f"Showing {len(filtered)} of {len(data)} files")

# ===================== TAB 6: Bayesian Network =====================
with tab6:
    st.subheader("Learned Bayesian Network ( Bayesian Networks for Risk Assessment)")
    st.write(
        "Unlike the Random Forest, this network's structure was **learned directly from the data** "
        "using a structure-learning algorithm (Hill-Climb search) - nothing here was hand-picked. "
        "Each arrow shows a probabilistic dependency the algorithm discovered between two metrics."
    )

    if BAYESIAN_RESULTS_PATH.exists() and BAYESIAN_GRAPH_PATH.exists():
        bayesian_results = json.loads(
            BAYESIAN_RESULTS_PATH.read_text(encoding="utf-8")
        )
        st.image(BAYESIAN_GRAPH_PATH, use_container_width=True)
        st.markdown("#### Learned edges")
        edges = pd.DataFrame(bayesian_results["edges"])
        if edges.empty:
            st.write("No edges were learned.")
        else:
            st.dataframe(edges, use_container_width=True, hide_index=True)

        st.markdown("#### Model performance")
        col_a, col_b = st.columns(2)
        with col_a:
            if master_results:
                st.metric(
                    f"Random Forest F1-macro ({master_results['evaluation_method']})",
                    f"{master_results['f1']:.3f}",
                )
            else:
                st.warning(
                    "master_results.json not found. Run master_analysis.py "
                    "to generate the Random Forest result."
                )
        with col_b:
            st.metric(
                f"Bayesian Network F1-macro "
                f"({bayesian_results['evaluation_method']})",
                f"{bayesian_results['f1']:.3f}",
            )
        st.caption(
            f"Bayesian Network accuracy: {bayesian_results['accuracy']:.3f}; "
            f"rows: {bayesian_results['rows']}."
        )
        st.markdown("#### Bayesian Network classification report")
        classification_metrics = bayesian_results.get("classification_report")
        if classification_metrics:
            report_rows = []
            for label in ["Low", "Medium", "High"]:
                if label in classification_metrics:
                    report_rows.append({
                        "Class / average": label,
                        **classification_metrics[label],
                    })
            report_rows.append({
                "Class / average": "Accuracy",
                "precision": None,
                "recall": None,
                "f1-score": classification_metrics["accuracy"],
                "support": sum(
                    classification_metrics[label]["support"]
                    for label in ["Low", "Medium", "High"]
                    if label in classification_metrics
                ),
            })
            for label in ["macro avg", "weighted avg"]:
                if label in classification_metrics:
                    report_rows.append({
                        "Class / average": label,
                        **classification_metrics[label],
                    })
            report_table = pd.DataFrame(report_rows).rename(
                columns={
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1-score": "F1-score",
                    "support": "Support",
                }
            )
            st.dataframe(
                report_table.style.format({
                    "Precision": "{:.2f}",
                    "Recall": "{:.2f}",
                    "F1-score": "{:.2f}",
                    "Support": "{:.0f}",
                }, na_rep="—"),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Class-level metrics are calculated on the stratified held-out "
                "test set. Support is the number of test files in each row."
            )
        else:
            st.info(
                "Re-run `python bayesian_network_analysis.py` to save the "
                "class-by-class metrics for this table."
            )
    else:
        st.warning(
            "Bayesian Network results or graph are missing. "
            "Run bayesian_network_analysis.py first to generate them."
        )

    st.markdown("#### Why Two Different Models?")
    st.write(
        "The Random Forest is optimized for prediction accuracy. The Bayesian Network "
        "trades some accuracy for **interpretability** - "
        "it explicitly maps out which metrics probabilistically influence which, as a visual graph, "
        "rather than acting as a black-box vote."
    )

# ===================== TAB 7: Robustness Checks =====================
with tab7:
    st.subheader("Do the feature findings hold up?")
    if not ROBUSTNESS_RESULTS_PATH.exists():
        st.warning(
            "Robustness results are missing. Run "
            "`python feature_robustness_check.py` from the project folder, "
            "then refresh this dashboard."
        )
    else:
        robustness = json.loads(
            ROBUSTNESS_RESULTS_PATH.read_text(encoding="utf-8")
        )
        correlations = pd.DataFrame(robustness["correlations"])
        permutation = pd.DataFrame(robustness["permutation_importance"])
        drop_one = pd.DataFrame(robustness["drop_one_feature"])
        groups = pd.DataFrame(robustness["feature_groups"])
        author_context = pd.DataFrame(robustness["author_context"])

        author_revision_r = float(
            author_context.loc[
                author_context["feature"] == "n-revs", "spearman_r"
            ].iloc[0]
        )
        without_authors = drop_one.loc[
            drop_one["feature"] == "num_authors"
        ].iloc[0]
        strongest_correlation = correlations.loc[
            correlations["spearman_r"].abs().idxmax()
        ]
        strongest_permutation = permutation.loc[
            permutation["f1_drop_mean"].idxmax()
        ]
        largest_drop_without_feature = drop_one.loc[
            drop_one["change_vs_all"].idxmin()
        ]

        st.caption(
            f"Calculated from {robustness['rows']} files in "
            f"{robustness['dataset']} · generated "
            f"{robustness['generated_at_utc']}"
        )
        metric_cols = st.columns(3)
        metric_cols[0].metric(
            "Strongest feature/revision association",
            f"{strongest_correlation['label']} "
            f"({strongest_correlation['spearman_r']:+.3f})",
            help=(
                "Largest absolute Spearman correlation in this dataset. "
                "Association is not causation."
            ),
        )
        metric_cols[1].metric(
            "Largest held-out F1 drop when shuffled",
            f"{strongest_permutation['label']}: "
            f"{strongest_permutation['f1_drop_mean']:.3f} "
            f"± {strongest_permutation['f1_drop_std']:.3f}",
            help=(
                "Largest mean macro-F1 drop among the features, averaged over "
                "30 shuffles on the held-out file split."
            ),
        )
        metric_cols[2].metric(
            "Largest F1 drop when a feature is removed",
            f"{largest_drop_without_feature['label']}: "
            f"{largest_drop_without_feature['change_vs_all']:+.3f}",
            help=(
                "Most negative change in five-fold macro F1 when one feature "
                "is removed; a negative change means performance decreased."
            ),
            delta_color="inverse",
        )

        st.markdown("#### 1. How is each feature associated with historical revisions?")
        st.write(
            "This section compares all nine input features against revision "
            "count without training a prediction model. Positive values mean "
            "higher feature values tend to go with higher revision counts; "
            "negative values mean they tend to move in opposite directions. "
            "The strength and direction can differ between features."
        )
        correlation_plot = correlations.sort_values("spearman_r")
        fig_corr, ax_corr = plt.subplots(figsize=(9, 5))
        ax_corr.barh(
            correlation_plot["label"],
            correlation_plot["spearman_r"],
            color=[
                "#C62828" if value > 0 else "#2E74B5"
                for value in correlation_plot["spearman_r"]
            ],
        )
        ax_corr.axvline(0, color="black", linewidth=0.8)
        ax_corr.set_xlabel("Spearman correlation with revision count")
        ax_corr.set_xlim(-1, 1)
        st.pyplot(fig_corr)
        st.dataframe(
            correlations[["label", "spearman_r", "p_value"]].rename(
                columns={
                    "label": "Feature",
                    "spearman_r": "Spearman correlation",
                    "p_value": "p-value",
                }
            ).style.format(
                {"Spearman correlation": "{:.3f}", "p-value": "{:.3g}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### 2. Permutation importance on held-out files")
        st.write(
            "Bars show how much macro F1 changed when one feature was shuffled. "
            "A mean drop larger than its shuffle-to-shuffle standard deviation is "
            "a useful signal here, not a formal significance test."
        )
        permutation_plot = permutation.sort_values("f1_drop_mean")
        fig_perm, ax_perm = plt.subplots(figsize=(9, 5))
        ax_perm.barh(
            permutation_plot["label"],
            permutation_plot["f1_drop_mean"],
            xerr=permutation_plot["f1_drop_std"],
            color=[
                "#2E74B5" if reliable else "#AAB7C4"
                for reliable in permutation_plot["reliably_above_noise"]
            ],
            capsize=3,
        )
        ax_perm.axvline(0, color="black", linewidth=0.8)
        ax_perm.set_xlabel("Macro F1 drop after shuffling a feature")
        st.pyplot(fig_perm)
        st.caption(
            "Blue: mean drop exceeds its standard deviation. Gray: the observed "
            "drop is not larger than that variability."
        )

        st.markdown("#### 3. Does each feature add information beyond the others?")
        st.write(
            "This removes one feature at a time and compares five-fold macro F1 "
            "with the all-feature baseline. Correlated features can substitute "
            "for one another, so a small drop does not mean a feature is useless."
        )
        st.dataframe(
            drop_one.sort_values("change_vs_all")[
                ["label", "f1_without_feature", "change_vs_all"]
            ].rename(
                columns={
                    "label": "Removed feature",
                    "f1_without_feature": "F1 without feature",
                    "change_vs_all": "Change vs all features",
                }
            ).style.format(
                {"F1 without feature": "{:.3f}", "Change vs all features": "{:+.3f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### 4. What do groups of metrics contribute?")
        groups = groups.copy()
        groups["feature_count"] = groups["features"].apply(len)
        fig_groups, ax_groups = plt.subplots(figsize=(9, 4))
        ax_groups.barh(groups["label"], groups["f1_macro"], color="#2E74B5")
        ax_groups.set_xlabel("Five-fold macro F1")
        ax_groups.set_xlim(0, 1)
        st.pyplot(fig_groups)
        st.dataframe(
            groups[["label", "feature_count", "f1_macro"]].rename(
                columns={
                    "label": "Feature group",
                    "feature_count": "Features",
                    "f1_macro": "Five-fold macro F1",
                }
            ).style.format({"Five-fold macro F1": "{:.3f}"}),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Example finding: why does Number of Authors stand out?")
        st.info(
            f"In this dataset, author count has a Spearman correlation of "
            f"{author_revision_r:.3f} with revision count, and removing it lowers "
            f"cross-validated macro F1 from "
            f"{robustness['cross_validated_f1_all_features']:.3f} to "
            f"{without_authors['f1_without_feature']:.3f}. That makes it a strong "
            "predictive signal in these historical data."
        )
        st.write(
            "It does **not** show that adding authors causes revisions. Both "
            "metrics are derived from the same repository history: files with more "
            "commits have had more opportunities to accumulate distinct authors. "
            "Code age, size, and function count also correlate with author count, "
            "so the features overlap rather than acting as independent causes."
        )
        st.warning(
            "**Interpretation limit:** the target is historical change-proneness "
            "(revision counts), not confirmed bugs or future defect risk. The "
            "random file split tests unseen files from the same repositories; see "
            "the Cross-Repository Evaluation tab for the stricter repository-held-out "
            "test. Neither evaluation predicts future commits because features are "
            "whole-history aggregates. Treat these as associations, not causal claims."
        )

# ===================== TAB 8: SonarQube Comparison =====================
with tab8:
    st.subheader("Compare SonarQube findings with repository risk metrics")
    st.write(
        "This view joins SonarQube's per-file measurements with your merged "
        "dataset using repository and file path. SonarQube measures code quality "
        "signals such as complexity, bugs, and code smells; your risk label "
        "represents historical revision activity, so the two are related but "
        "not interchangeable."
    )
    if not SONAR_COMPARISON_PATH.exists():
        st.warning(
            "The comparison data is not available yet. Start SonarQube, scan the "
            "configured projects, and run `python sonarqube_file_comparison.py` "
            "from the project folder. Then refresh this dashboard."
        )
    else:
        sonar_data = pd.read_csv(SONAR_COMPARISON_PATH)
        prediction_column = (
            "repo_holdout_prediction"
            if "repo_holdout_prediction" in sonar_data.columns else "risk_label"
        )
        has_repo_holdout_predictions = {
            "repo_holdout_prediction",
            "repo_holdout_high_probability",
            "repo_holdout_actual_label",
        }.issubset(sonar_data.columns)
        sonar_metric_columns = [
            column for column in [
                "sonar_complexity",
                "sonar_cognitive_complexity",
                "sonar_code_smells",
                "sonar_bugs",
                "sonar_duplicated_lines_density",
                "sonar_ncloc",
            ]
            if column in sonar_data.columns
        ]
        for column in sonar_metric_columns:
            sonar_data[column] = pd.to_numeric(sonar_data[column], errors="coerce")

        if sonar_data.empty:
            st.warning(
                "The comparison CSV exists but has no matched files. Check that "
                "SonarQube project paths match the repository/file paths in your dataset."
            )
        else:
            if not has_repo_holdout_predictions:
                st.warning(
                    "This comparison file predates the repository-held-out "
                    "predictions. Run `python master_analysis.py`, then "
                    "`python sonarqube_file_comparison.py --use-cached`, and "
                    "refresh the dashboard to compare SonarQube with predictions "
                    "from a model that did not train on each file's repository."
                )
            metric_columns = st.columns(3)
            metric_columns[0].metric("Matched files", len(sonar_data))
            metric_columns[1].metric(
                "Repositories represented",
                sonar_data["repo"].nunique() if "repo" in sonar_data else 0,
            )
            metric_columns[2].metric(
                "Files predicted High change-proneness",
                int((sonar_data[prediction_column] == "High").sum())
                if prediction_column in sonar_data else 0,
            )

            st.caption(
                f"Showing the saved comparison from {SONAR_COMPARISON_PATH.name}. "
                "Re-run sonarqube_file_comparison.py after rescanning SonarQube."
            )
            st.caption(
                "Missing SonarQube code-smell, bug, and duplication measures are "
                "treated as zero reported findings; other missing measures remain N/A."
            )

            if {"sonar_code_smells", prediction_column}.issubset(sonar_data.columns):
                st.markdown("#### SonarQube findings by model-predicted category")
                summary_metrics = [
                    column for column in [
                        "sonar_code_smells",
                        "sonar_bugs",
                        "sonar_cognitive_complexity",
                        "sonar_duplicated_lines_density",
                    ]
                    if column in sonar_data.columns
                ]
                summary = (
                    sonar_data.groupby(prediction_column)[summary_metrics]
                    .agg(["count", "median", "mean"])
                    .reindex(["Low", "Medium", "High"])
                )
                summary.columns = [
                    f"{metric} — {statistic}"
                    for metric, statistic in summary.columns
                ]
                st.dataframe(
                    summary.style.format("{:.2f}", na_rep="—"),
                    use_container_width=True,
                )

            st.markdown("#### Matched file details")
            available_repos = (
                sorted(sonar_data["repo"].dropna().unique())
                if "repo" in sonar_data else []
            )
            filter_cols = st.columns(2)
            selected_repos = filter_cols[0].multiselect(
                "Repository",
                available_repos,
                default=available_repos,
                key="sonar_repo_filter",
            )
            available_risks = (
                [risk for risk in ["Low", "Medium", "High"]
                 if risk in sonar_data[prediction_column].dropna().unique()]
                if prediction_column in sonar_data else []
            )
            selected_risks = filter_cols[1].multiselect(
                "Model-predicted category",
                available_risks,
                default=available_risks,
                key="sonar_risk_filter",
            )
            filtered_sonar_data = sonar_data
            if "repo" in filtered_sonar_data:
                filtered_sonar_data = filtered_sonar_data[
                    filtered_sonar_data["repo"].isin(selected_repos)
                ]
            if prediction_column in filtered_sonar_data:
                filtered_sonar_data = filtered_sonar_data[
                    filtered_sonar_data[prediction_column].isin(selected_risks)
                ]
            display_columns = [
                column for column in [
                    "repo", "File", "risk_label",
                    "repo_holdout_actual_label", "repo_holdout_prediction",
                    "repo_holdout_high_probability", "n-revs",
                    "sonar_complexity", "sonar_cognitive_complexity",
                    "sonar_code_smells", "sonar_bugs",
                    "sonar_duplicated_lines_density", "sonar_ncloc",
                    "max_complexity",
                ]
                if column in filtered_sonar_data.columns
            ]
            st.dataframe(
                filtered_sonar_data[display_columns],
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                f"Showing {len(filtered_sonar_data)} of {len(sonar_data)} matched files."
            )

    st.divider()
    render_sonarqube_model_comparison()

# ===================== TAB 9: Cross-Repository Evaluation =====================
with tab9:
    st.subheader("Does the model generalize to a repository it has not seen?")
    st.write(
        "Leave-one-repository-out evaluation trains on four repositories and "
        "tests on the fifth, repeating this for each repository. Every file's "
        "displayed held-out prediction comes from a model that did not train on "
        "that file's repository."
    )
    if not GENERALIZATION_RESULTS_PATH.exists():
        st.warning(
            "Cross-repository results are missing. Run `python master_analysis.py` "
            "to regenerate the dataset with held-out predictions and this report."
        )
    else:
        generalization = json.loads(
            GENERALIZATION_RESULTS_PATH.read_text(encoding="utf-8")
        )
        summary_columns = st.columns(3)
        summary_columns[0].metric(
            "All held-out files: macro F1",
            f"{generalization['overall_oof_f1_macro']:.3f}",
        )
        summary_columns[1].metric(
            "All held-out files: accuracy",
            f"{generalization['overall_oof_accuracy']:.3f}",
        )
        summary_columns[2].metric(
            "Mean F1 across repositories",
            f"{generalization['mean_repository_f1_macro']:.3f}",
        )

        st.markdown("#### Results for each held-out repository")
        repository_results = pd.DataFrame(generalization["repositories"])
        st.dataframe(
            repository_results.rename(
                columns={
                    "repository": "Held-out repository",
                    "rows": "Test files",
                    "f1_macro": "Macro F1",
                    "accuracy": "Accuracy",
                    "train_low_max_revisions": "Training Low cutoff",
                    "train_medium_max_revisions": "Training Medium cutoff",
                }
            ).style.format(
                {
                    "Macro F1": "{:.3f}",
                    "Accuracy": "{:.3f}",
                    "Training Low cutoff": "{:.1f}",
                    "Training Medium cutoff": "{:.1f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"Evaluated {generalization['rows']} files using "
            f"{generalization['evaluation_method']}."
        )
        st.info(
            "For each held-out repository, Low/Medium/High cutoffs are calculated "
            "from revision counts in the four training repositories, then applied "
            "to the held-out repository. Therefore class proportions can differ "
            "across the held-out repositories."
        )
        st.warning(
            "**Scope limitation:** this is a test of cross-repository "
            "generalization, not a prediction of future changes. The dataset uses "
            "whole-history aggregates (including author counts and revision counts), "
            "so a valid time-based evaluation would require rebuilding metrics from "
            "a historical snapshot and predicting changes after that date. The "
            "target remains historical change-proneness, not confirmed defects."
        )

# ===================== TAB 10: Score a new repository =====================
with tab10:
    st.subheader("Score Python files from a team's own repository")
    st.write(
        "Run `score_repo.py` locally to extract file metrics and generate "
        "`risk_report.csv`, then load the report here. The report uses the model "
        "and class cutoffs saved from the training repositories."
    )
    uploaded_report = st.file_uploader(
        "Upload risk_report.csv (or use the default report beside the dashboard)",
        type=["csv"],
        key="risk_report_upload",
    )
    report_data = None
    report_source = uploaded_report or (
        RISK_REPORT_PATH if RISK_REPORT_PATH.is_file() else None
    )
    if report_source is not None:
        try:
            report_data = pd.read_csv(report_source)
        except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
            st.error(f"Could not read the selected risk report: {error}")
        if uploaded_report is None and report_data is not None:
            st.caption(f"Loaded `{RISK_REPORT_PATH.name}` from the dashboard folder.")
    else:
        st.info(
            "No report is loaded yet. Generate one with "
            "`python score_repo.py --repo C:\\path\\to\\repo --out risk_report.csv` "
            "after running `python master_analysis.py` to create the model files."
        )

    if report_data is not None:
        required_report_columns = {
            "File",
            "revisions_before",
            "predicted_class",
            "probability_high",
            "model_used",
            "used_static_model",
            "top_reason_1",
            "top_reason_2",
            "top_reason_3",
        }
        missing_report_columns = sorted(
            required_report_columns - set(report_data.columns)
        )
        if missing_report_columns:
            st.error(
                "This CSV does not look like a RiskLens report. Missing columns: "
                + ", ".join(missing_report_columns)
            )
        else:
            ranked_columns = [
                "File",
                "predicted_class",
                "probability_high",
                "revisions_before",
                "model_used",
                "top_reason_1",
                "top_reason_2",
                "top_reason_3",
            ]
            ranked_report = report_data.sort_values(
                ["probability_high", "File"], ascending=[False, True]
            )
            st.markdown("#### Ranked files")
            st.dataframe(
                ranked_report[ranked_columns].rename(
                    columns={
                        "File": "File",
                        "predicted_class": "Predicted historical activity",
                        "probability_high": "Relative High score",
                        "revisions_before": "Commits touching file",
                        "model_used": "Model used",
                        "top_reason_1": "SHAP reason 1",
                        "top_reason_2": "SHAP reason 2",
                        "top_reason_3": "SHAP reason 3",
                    }
                ).style.format({"Relative High score": "{:.1%}"}),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Files with fewer than five history commits use the static-only "
                "model. “Relative High score” is the classifier's probability for "
                "the High historical-activity class, not a probability of bugs."
            )
            if {
                "training_low_cutoff_revisions",
                "training_medium_cutoff_revisions",
            }.issubset(report_data.columns):
                first_report_row = report_data.iloc[0]
                st.caption(
                    "Training revision-count cutoffs used for the classes: "
                    f"Low ≤ {first_report_row['training_low_cutoff_revisions']:.1f}; "
                    "Medium ≤ "
                    f"{first_report_row['training_medium_cutoff_revisions']:.1f}; "
                    "High above the Medium cutoff. They are not recalculated from "
                    "this repository."
                )

            selected_file = st.selectbox(
                "Select a file to inspect its High-class feature contributions",
                ranked_report["File"].tolist(),
                key="risk_report_selected_file",
            )
            selected_row = report_data.loc[
                report_data["File"] == selected_file
            ].iloc[0]
            st.write(
                f"**Prediction:** {selected_row['predicted_class']} historical "
                f"change-proneness · **High score:** "
                f"{selected_row['probability_high']:.1%} · "
                f"**Model:** {selected_row['model_used']}"
            )
            for reason_column in [
                "top_reason_1",
                "top_reason_2",
                "top_reason_3",
            ]:
                if pd.notna(selected_row[reason_column]) and selected_row[reason_column]:
                    st.write(f"- {selected_row[reason_column]}")

            shap_features = [
                feature for feature in FEATURES
                if f"shap_{feature}" in report_data.columns
                and pd.notna(selected_row[f"shap_{feature}"])
            ]
            if shap_features:
                shap_scores = pd.Series(
                    {
                        feature: selected_row[f"shap_{feature}"]
                        for feature in shap_features
                    }
                ).sort_values()
                shap_labels = {
                    feature: FEATURE_LABELS.get(feature, feature)
                    for feature in shap_features
                }
                fig_repo_shap, ax_repo_shap = plt.subplots(figsize=(9, 5))
                colors = [
                    "#C62828" if value > 0 else "#2E74B5"
                    for value in shap_scores.values
                ]
                ax_repo_shap.barh(
                    [shap_labels[feature] for feature in shap_scores.index],
                    shap_scores.values,
                    color=colors,
                )
                ax_repo_shap.axvline(0, color="black", linewidth=0.8)
                ax_repo_shap.set_xlabel(
                    "SHAP contribution to the model's High-class score"
                )
                ax_repo_shap.set_title(
                    f"Feature contributions to the High-class score: {selected_file}"
                )
                ax_repo_shap.grid(axis="x", alpha=0.25)
                fig_repo_shap.tight_layout()
                st.pyplot(fig_repo_shap)
                st.caption(
                    "Red contributions move this model's High-class score up; "
                    "blue contributions move it down. SHAP explains the model, "
                    "not a causal effect."
                )
            else:
                st.warning(
                    "This report has no SHAP contribution columns for the selected "
                    "file. Generate it with the current `score_repo.py`."
                )
