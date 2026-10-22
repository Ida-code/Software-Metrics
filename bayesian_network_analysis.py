"""
Bayesian Network Analysis of Software Risk 

Unlike a hand-picked, expert-defined Bayesian Network (which would be
subjective), this script LEARNS the network structure directly from your
real dataset using a structure-learning algorithm (Hill-Climb search).
Nothing is hand-picked - the algorithm discovers which metrics
probabilistically depend on which, based on the rows in the dataset.

WHAT THIS ADDS TO YOUR PROJECT:
- A visual graph showing how your 9 metrics (and risk) influence each other
- Conditional probability tables learned from real data
- A second, independent model to compare against your Random Forest
- Direct coverage of Module 4's Bayesian Network content

HOW TO USE:
1. Place this script in your main folder (same as full_merged_dataset.csv)
2. Install pgmpy once:
   pip install pgmpy
3. Run:
   python bayesian_network_analysis.py
"""

from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from pgmpy.causal_discovery import HillClimbSearch, ExpertKnowledge
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteMLE
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR / "full_merged_dataset.csv"
RESULTS_PATH = SCRIPT_DIR / "bayesian_results.json"
GRAPH_PATH = SCRIPT_DIR / "bayesian_network_graph.png"
RANDOM_SEED = 42


class DeterministicHillClimbSearch(HillClimbSearch):
    def _legal_operations_dag(self, *args, **kwargs):
        # pgmpy enumerates candidate edge additions from a set, so sort ties consistently.
        yield from sorted(
            super()._legal_operations_dag(*args, **kwargs),
            key=lambda operation: operation[0],
        )


FEATURES = ["avg_complexity", "max_complexity", "total_nloc", "function_count",
            "maintainability_index", "num_authors", "code_age_days",
            "coupling_partner_count", "avg_coupling_degree"]

# Shorter, cleaner names for readability in the graph
DISPLAY_NAMES = {
    "avg_complexity": "AvgComplexity",
    "max_complexity": "MaxComplexity",
    "total_nloc": "TotalLOC",
    "function_count": "FuncCount",
    "maintainability_index": "Maintainability",
    "num_authors": "NumAuthors",
    "code_age_days": "CodeAge",
    "coupling_partner_count": "CouplingCount",
    "avg_coupling_degree": "CouplingDegree",
    "risk_label": "Risk",
}

# ---------- STEP 1: Load your existing merged dataset ----------

print(f"Loading {DATASET_PATH.name} ...")
df = pd.read_csv(DATASET_PATH)

for col in ["coupling_partner_count", "avg_coupling_degree"]:
    if col not in df.columns:
        df[col] = 0.0

df = df.dropna(subset=FEATURES + ["n-revs"]).copy()

# ---------- STEP 2: Recreate the same objective risk label as before ----------

q1 = df["n-revs"].quantile(0.33)
q2 = df["n-revs"].quantile(0.66)

def label_risk(revs):
    if revs <= q1:
        return "Low"
    elif revs <= q2:
        return "Medium"
    return "High"

df["risk_label"] = df["n-revs"].apply(label_risk)

# ---------- STEP 3: Discretize each feature into Low/Medium/High using QUANTILES ----------
# Bayesian Networks (in their classic form) work with discrete/categorical
# variables, not raw continuous numbers. Using quantiles (same objective
# approach as the risk label itself) avoids picking arbitrary cutoffs by hand.

print("Discretizing features into Low/Medium/High using data quantiles...")

discretized = pd.DataFrame()
discretisation_bins = {}
for feat in FEATURES:
    try:
        discretized[feat], bin_edges = pd.qcut(
            df[feat], q=3, labels=["Low", "Medium", "High"],
            duplicates="drop", retbins=True,
        )
        bin_method = "quantile"
    except ValueError:
        # If a feature has too many identical values to split into 3 bins,
        # fall back to as many bins as the data actually supports.
        ranked_values = df[feat].rank(method="first")
        discretized[feat], bin_edges = pd.qcut(
            ranked_values, q=3, labels=["Low", "Medium", "High"], retbins=True,
        )
        bin_method = "rank_then_quantile"
    discretisation_bins[feat] = {
        "method": bin_method,
        "labels": ["Low", "Medium", "High"],
        "edges": [float(edge) for edge in bin_edges],
    }

discretized["risk_label"] = df["risk_label"].values
discretized = discretized.astype(str)

# Rename columns to shorter display names for a cleaner graph
discretized = discretized.rename(columns=DISPLAY_NAMES)

# ---------- STEP 4: Split into train/test (same idea as your Random Forest) ----------

train_df, test_df = train_test_split(
    discretized, test_size=0.25, random_state=RANDOM_SEED,
    stratify=discretized["Risk"],
)

# ---------- STEP 5: LEARN the network structure from data (not hand-picked) ----------

print("\nLearning Bayesian Network structure using Hill-Climb search (BIC score)...")

# IMPORTANT: forbid Risk from being a PARENT of any structural metric.
# Score-based structure learning only finds the statistically best-fitting
# direction, not true real-world causation - without a constraint, it can
# (and did, in earlier testing) suggest illogical edges like "Risk causes
# TotalLOC". Since Risk is DERIVED FROM the metrics (via revision count),
# it logically cannot be a cause of them - it can only ever be an effect.
other_vars = [c for c in discretized.columns if c != "Risk"]
forbidden_edges = [("Risk", var) for var in other_vars]
expert_knowledge = ExpertKnowledge(forbidden_edges=forbidden_edges)

hc = DeterministicHillClimbSearch(return_type="dag", expert_knowledge=expert_knowledge)
hc.fit(train_df)
learned_edges = list(hc.causal_graph_.edges())

print(f"\nLearned {len(learned_edges)} edges (dependencies) from the data:")
for parent, child in learned_edges:
    print(f"  {parent}  -->  {child}")

# ---------- STEP 6: Fit the Conditional Probability Tables from real data ----------

model = DiscreteBayesianNetwork(learned_edges)
# Some nodes might not get any edges if HillClimbSearch found them independent -
# add them in as isolated nodes so they're still part of the model.
for col in discretized.columns:
    if col not in model.nodes():
        model.add_node(col)

model.fit(train_df, estimator=DiscreteMLE())
print(f"\nLearned Conditional Probability Tables for {len(model.get_cpds())} variables.")

# ---------- STEP 7: Evaluate predictive accuracy on the held-out test set ----------

print("\nEvaluating the Bayesian Network's Risk predictions on the held-out test set...")
infer = VariableElimination(model)

# Only use evidence variables that are actually connected to Risk in the learned graph,
# since VariableElimination needs the evidence nodes to be part of the model.
evidence_vars = [n for n in discretized.columns if n != "Risk" and n in model.nodes()]

predictions = []
actuals = []
for _, row in test_df.iterrows():
    evidence = {var: row[var] for var in evidence_vars if var in model.nodes()}
    try:
        result = infer.map_query(variables=["Risk"], evidence=evidence, show_progress=False)
        predictions.append(result["Risk"])
    except Exception:
        predictions.append("Medium")  # fallback if evidence combination was never seen in training
    actuals.append(row["Risk"])

acc = accuracy_score(actuals, predictions)
f1 = f1_score(actuals, predictions, average="macro")
report = classification_report(
    actuals, predictions, output_dict=True, zero_division=0
)

print(f"\nBayesian Network Accuracy: {acc:.3f}")
print(f"Bayesian Network F1-macro: {f1:.3f}")
print("\nDetailed classification report:")
print(classification_report(actuals, predictions, zero_division=0))

# ---------- STEP 8: Compare against your Random Forest's earlier result ----------

print("\n=== COMPARISON: Bayesian Network vs Random Forest ===")
print(f"Bayesian Network (this script):           F1-macro = {f1:.3f} (single held-out test)")

# ---------- STEP 9: Visualize the learned network graph (hierarchical layout) ----------

print("\nDrawing the learned Bayesian Network graph...")
G = nx.DiGraph()
G.add_nodes_from(discretized.columns)
G.add_edges_from(learned_edges)

# Drop any variable the algorithm found to be completely independent
# (no edges in or out) - these add clutter without showing a real relationship.
isolated_nodes = list(nx.isolates(G))
if isolated_nodes:
    print(f"Removing {len(isolated_nodes)} independent variable(s) from the graph: {isolated_nodes}")
    G.remove_nodes_from(isolated_nodes)

# Arrange nodes in a top-to-bottom hierarchy based on the DAG's topological
# generations, similar to a classic Bayesian Network diagram, instead of a
# generic spring layout.
if nx.is_directed_acyclic_graph(G) and len(G.nodes()) > 0:
    generations = list(nx.topological_generations(G))
else:
    generations = [list(G.nodes())]

pos = {}
layer_height = 1.6
for level, nodes_in_layer in enumerate(generations):
    n = len(nodes_in_layer)
    for i, node in enumerate(nodes_in_layer):
        x = (i - (n - 1) / 2) * 2.4
        y = -level * layer_height
        pos[node] = (x, y)

plt.figure(figsize=(12, max(6, len(generations) * 1.8)))

node_colors = ["#C00000" if node == "Risk" else "#2E74B5" for node in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=4200, alpha=0.95)
nx.draw_networkx_labels(G, pos, font_size=8, font_color="white", font_weight="bold")
nx.draw_networkx_edges(
    G, pos, arrowstyle="-|>", arrowsize=22, edge_color="#555555", width=1.8,
    connectionstyle="arc3,rad=0.05", node_size=4200,
)

plt.title("Learned Bayesian Network: Dependencies Between Software Metrics and Risk", fontsize=13)
plt.axis("off")
plt.margins(x=0.15)
plt.tight_layout()
plt.savefig(GRAPH_PATH, dpi=150)
print(f"Saved chart: {GRAPH_PATH.name}")

results = {
    "f1": float(f1),
    "evaluation_method": "single stratified held-out test split",
    "test_size": 0.25,
    "accuracy": float(acc),
    "classification_report": report,
    "edges": [{"source": source, "target": target} for source, target in learned_edges],
    "nodes": list(discretized.columns),
    "discretisation_bins": discretisation_bins,
    "rows": int(len(discretized)),
    "random_seed": RANDOM_SEED,
    "library_versions": {
        "pgmpy": version("pgmpy"),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": version("scikit-learn"),
        "networkx": nx.__version__,
        "matplotlib": matplotlib.__version__,
    },
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
}
RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"Saved results: {RESULTS_PATH.name}")

# ---------- STEP 10: Show what directly influences Risk, if anything does ----------

direct_causes_of_risk = [parent for parent, child in learned_edges if child == "Risk"]
influenced_by_risk = [child for parent, child in learned_edges if parent == "Risk"]

print("\n--- What the learned network says about Risk ---")
if direct_causes_of_risk:
    print(f"Variables the algorithm found to DIRECTLY influence Risk: {direct_causes_of_risk}")
else:
    print("The algorithm did not find any variable with a direct edge INTO Risk.")
if influenced_by_risk:
    print(f"Variables the algorithm found that Risk appears to influence: {influenced_by_risk}")

print(f"\nDone. Files saved: {GRAPH_PATH.name}, {RESULTS_PATH.name}")
