# RiskLens Project Report

**Project:** Software metrics and repository-history analysis  
**Report date:** 20 October 2026  
**Purpose of this report:** Explain what the project does, what has been built, what the results mean, and what they do not mean, in plain language.

## 1. Executive summary

RiskLens is a Python project that studies source files in five software
repositories: Requests, Flask, Click, HTTPX, and Rich. It combines measurements of the code itself with information from each repository's Git history, uses machine-learning models to group files by how often they have changed, and shows findings in a Streamlit dashboard.

The most important qualification is the meaning of the categories. The current
Low, Medium, and High labels are made from the number of past commits touching
each file. They describe **historical change activity**. They do not say that a
file contains a bug, will have a bug, or will necessarily change again.

The project has several related parts:

1. **Code and history measurements:** Lizard, Radon, CodeMaat, PyDriller, and
   SonarQube provide different measurements of source files and their history.
2. **Historical classification:** Random Forest models learn to classify the
   dataset's Low/Medium/High historical revision groups.
3. **Evaluation:** Cross-validation, held-out-repository evaluation, feature
   checks, a time-based experiment, and a SonarQube comparison help show what
   the models learn and where their evidence is limited.
4. **Explanations and dashboard:** Streamlit displays results; SHAP explains
   which inputs pushed an individual model score up or down.
5. **Local-repository scoring:** A separate command can measure files in a
   team's local Git repository and produce a CSV report.

This makes RiskLens a **software measurement and empirical analysis project
with machine-learning classification**. Static analysis is one ingredient,
but it is not the whole project: several inputs come from repository history,
and the model and dashboard add analysis on top.

## 2. What question the project asks

The central question is:

> In this collection of repositories, what code and history measurements are
> associated with files that have had more revision activity?

The model learns patterns from known files and assigns each file a Low,
Medium, or High class. For a new repository, RiskLens applies the model trained
on the original five repositories and reports a relative score.

The project does **not** currently answer:

- “Does this file have a bug?”
- “Will this file have a bug next month?”
- “Will this file definitely change again?”
- “Is this file objectively dangerous?”

Those questions would require different targets and, for future claims,
carefully measured future outcomes. A high score here means the model finds
the file more like examples in its **High historical-change class**.

## 3. Data used so far

The merged training dataset contains **381 Python-file rows** from five
repositories:

| Repository | Files in merged dataset |
| ---------- | ----------------------: |
| Rich       |                     154 |
| Click      |                      81 |
| Flask      |                      63 |
| HTTPX      |                      54 |
| Requests   |                      29 |
| **Total**  |                 **381** |

The historical revision-count thresholds in this dataset are **7** and **27**:

- **Low:** at most 7 revisions
- **Medium:** more than 7 and at most 27 revisions
- **High:** more than 27 revisions

The current dataset has 131 Low, 123 Medium, and 127 High labels. These
thresholds are calculated from the training data. The local-repository scorer
does not calculate new percentile thresholds from the team's files.

The principal nine model inputs are:

| Input                   | Plain-language meaning                                                                                                                        |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Average Complexity      | Average Lizard cyclomatic-complexity score across functions in the file. It is a measure of branching paths, not a direct measure of quality. |
| Maximum Complexity      | Highest Lizard cyclomatic-complexity score among the file's functions.                                                                        |
| Total Lines of Code     | Sum of Lizard's non-comment source lines across the file's functions.                                                                         |
| Function Count          | Number of functions Lizard identifies in the file.                                                                                            |
| Maintainability Index   | Radon's formula-based maintainability score. It is a tool-derived indicator, not a guarantee that code is easy or hard to maintain.           |
| Number of Authors       | Number of distinct commit author names recorded for the file.                                                                                 |
| Code Age (days)         | Time from the file's first recorded commit to the measurement date.                                                                           |
| Coupling Partner Count  | Number of other files that changed in the same commits as this file, under the project's CodeMaat aggregation.                                |
| Average Coupling Degree | Average shared-commit degree of those coupled file pairs in the CodeMaat data.                                                                |

The project also has revision counts and churn (added and deleted lines) in
its wider dataset. The nine-feature classifier does not use churn or the
bug-fix-commit columns as inputs. Bug-fix-related commit counts are kept
separate because they are also history-derived and are not independently
verified defect labels.

## 4. How the data pipeline works

The repository-level analysis brings together outputs from several tools:

- **Lizard** reads Python source and reports function-level complexity,
  non-comment lines, and function counts. The project aggregates those
  function measurements to file-level values.
- **Radon** reads Python source and computes a maintainability index.
- **CodeMaat** processes change-history data to provide file revision, churn,
  and coupling measurements.
- **PyDriller** reads local Git history to count distinct authors, measure
  file age, and cross-check commit activity.
- **SonarQube** provides a separate set of static-analysis measures when
  scanned results are available. These include complexity, cognitive
  complexity, code smells, bugs, duplication density, and lines of code.

The original repository data is assembled in the existing analysis workflow
and exported as `full_merged_dataset.csv`. The model turns the numeric revision
counts into the three historical-activity categories. SonarQube's file-level
data is matched to the model dataset using repository name and normalized
file path; **381 of 381** model files matched in the current comparison.

The tools are measuring different things. For example, SonarQube's reported
“bugs” are findings from its rules and configuration. They are not the same as
confirmed bug-fix commits, and neither is the target used for the main model.

## 5. Main machine-learning model

The main model is a **Random Forest** classifier. In simple terms, it combines
many small decision trees. Each tree looks at combinations of measurements,
and the forest combines their votes.

The current training setup uses:

- 200 trees
- Maximum tree depth of 6
- Balanced class weighting, so the model does not simply favor the most common
  class
- A fixed random seed for reproducible results
- The nine inputs listed above

The model is trained to reproduce the Low/Medium/High labels derived from the
revision counts. It is therefore learning relationships in this dataset; its
score is not a calibrated chance that a defect exists.

### Why author count is so influential

In the current dataset, the number of authors has a Spearman correlation of
about **0.84** with revision count. Spearman correlation measures whether two
values tend to rise together in rank; it does not establish that one causes
the other.

When author count was shuffled for held-out examples, macro F1 fell by about
**0.254 ± 0.036**. In a separate check, removing author count reduced
five-fold macro F1 from about **0.784 to 0.711**. Adding authors to static code
metrics raised macro F1 from about **0.654 to 0.760**.

That strong result is understandable: files touched repeatedly have more
opportunities to be touched by multiple authors. Author count and revision
count are two related views of repository history. This makes author count
useful for recognizing historical activity, but it also means the model's
performance should not be presented as evidence that code authorship predicts
bugs. It is one of the reasons time-based evaluation is valuable.

Static code measures still carry information, but in this particular dataset
they do less to distinguish the historical revision groups on their own.
Measurements can also overlap: a large file may have more functions, and those
functions may create more opportunities for complexity. A feature's small
individual effect does not automatically mean it has no value in combination.

## 6. What the evaluation results say

### 6.1 Random stratified five-fold evaluation

The ordinary five-fold evaluation splits files into five parts, trains on four
parts, and evaluates on the remaining part, repeating the process. It helps
measure performance on other files drawn from a similar collection.

The nine-feature model achieved a mean macro F1 of about **0.784**. Macro F1
averages performance across the three classes so that each class contributes
equally to the overall score. It is not the same as “78.4% of future bugs
predicted correctly.”

This evaluation is useful, but it is not the strongest test of transfer to a
new project: files from one repository can appear in both training and
evaluation folds.

### 6.2 Holding out an entire repository

The leave-one-repository-out test trains on four repositories and evaluates
on the fifth, rotating which repository is held out. In this test, the class
thresholds are calculated from the four training repositories and then applied
to the held-out one.

Across all 381 held-out predictions, the model achieved:

- Overall macro F1: **0.735**
- Accuracy: **0.735**
- Mean macro F1 across the five repositories: **0.706**

Results differed between repositories. Macro F1 ranged from about **0.546**
for Requests to **0.779** for Click. This variation matters: a single average
does not mean the model performs equally well on every project.

This test is stronger evidence about transfer to another repository than a
random file split. However, its inputs still use whole-history aggregates.
It tests whether historical patterns transfer across these repositories; it
does **not** test whether information available today predicts changes that
happen later.

### 6.3 Time-based validation

The temporal experiment uses each repository's commit history to define a
cutoff at about 70% of its first-parent history. It measures the code and
history available at that cutoff, then counts the file's revisions after the
cutoff. Files that did not exist at the cutoff are not included. This produced
**405 files** across the five cutoff snapshots.

The experiment compares a simple baseline—using only revisions before the
cutoff—with models using static metrics, authors, and combinations of
pre-cutoff information. In the pooled repository-held-out predictions:

| Model inputs                                 | Macro F1 | High-score/revision Spearman |
| -------------------------------------------- | -------: | ---------------------------: |
| Revisions before cutoff only                 |    0.545 |                        0.672 |
| Static code metrics at cutoff only           |    0.583 |                        0.615 |
| All pre-cutoff inputs                        |    0.635 |                        0.759 |
| All pre-cutoff inputs except prior revisions |    0.622 |                        0.743 |

The temporal report also measures how many of the files in the top 10% of the
model's High-score ranking actually fell into the future High-revision group.
On the pooled repository-held-out predictions this was **87.8%** for all
pre-cutoff inputs and **90.2%** for the revisions-only baseline. In this
particular top-10% measure, adding all inputs did not beat the baseline.

The pooled macro F1 for the full pre-cutoff model is higher than for the
revisions-only baseline, and the full model beat the baseline in **four of the
five** individual repository-held-out folds. The baseline did better on the
Click fold. These results are encouraging in some respects, but they also show
that the answer depends on which performance measure and repository are
considered.

The within-repository file-split result was more mixed: all pre-cutoff inputs
scored **0.620**, while revisions-only scored **0.603**; the model without
prior revisions scored **0.654**. This cautions against treating one model
configuration as universally best. These are still experiments on five
specific repositories and their available histories—not proof of performance
on arbitrary teams or future projects.

The temporal experiment is the project's closest test of future change
activity. Its target is still **future revisions**, not future bugs.

### 6.4 SonarQube comparison

SonarQube was compared on the same matched files and same five stratified
folds as the model feature groups:

| Inputs given to classifier       | Mean macro F1 |
| -------------------------------- | ------------: |
| SonarQube measures only          |         0.640 |
| Five static code measures        |         0.654 |
| Four repository-history measures |         0.762 |
| Existing nine measures           |         0.784 |
| Existing nine plus SonarQube     |         0.783 |

Adding SonarQube to the existing nine features changed the mean paired F1 by
about **−0.001** (paired-fold standard deviation **0.025**). Under the
project's stated descriptive rule, this is **not an improvement greater than
one paired-fold standard deviation**.

This does not mean SonarQube is useless. It answers a different question:
SonarQube identifies code-rule findings and quality indicators, while this
classifier groups files by historical or future revision activity depending
on the evaluation. The comparison only says that SonarQube measures did not
improve this particular historical-label classification score in the current
matched dataset and test setup.

For this comparison, missing SonarQube code-smell and bug counts are treated
as zero reported findings; other missing SonarQube numeric values are imputed
using medians calculated within the training fold. This is a data-handling
choice, not evidence that a missing measurement always means “no issue.”

## 7. Scoring another team's local repository

The project includes a separate scoring path for a team's local repository:

1. `master_analysis.py` trains and saves the **full nine-feature model** and a
   **static-only model**.
2. `score_repo.py` reads Python files from a local Git working tree, measures
   code and history locally, scores each file, and writes a CSV.
3. The **Score a New Repo** dashboard tab loads `risk_report.csv` or accepts an
   uploaded report. It shows files ranked by the model's High-class score and
   displays the selected file's SHAP contributions.

A file with **fewer than five commits touching it** uses the static-only
model, because its history measurements are limited. The report flags which
model was used. The static model uses complexity, lines of code, function
count, and maintainability index; it does not use authors, age, or coupling.

The scoring artifacts store the expected feature order and training cutoffs.
The cutoffs remain the training dataset's values; they are **not recalculated
from the new repository's percentiles**. Consequently, the High-class score
is relative to what the model learned from Requests, Flask, Click, HTTPX, and
Rich. It is not a measured probability that the team's file will change or
contain a bug.

The scorer processes local files and Git history and does not clone a
repository or call a network service. It includes tracked files and
non-ignored untracked Python files. The `--exclude-tests` option omits common
test paths and test-style filenames. Its history measurements use the local
repository's available commit history, so a shallow or incomplete clone can
produce incomplete history features.

### Example commands

Run the training workflow from the project folder:

```powershell
python master_analysis.py
```

Score a local repository:

```powershell
python score_repo.py --repo "C:\path\to\team-repo" --out risk_report.csv
```

Optionally exclude test files:

```powershell
python score_repo.py --repo "C:\path\to\team-repo" --out risk_report.csv --exclude-tests
```

Run the dashboard:

```powershell
streamlit run dashboard.py
```

For the dashboard to find a report automatically, save it as `risk_report.csv`
in the same folder as `dashboard.py`. Otherwise, upload the CSV in the
Score a New Repo tab.

## 8. Dashboard contents

The Streamlit dashboard brings together the current analysis and scoring
features. Its tabs cover:

- Exploring a file's historical-activity prediction and its SHAP explanation
- Comparing two files
- A high-change-proneness leaderboard
- Model and feature insights
- Browsing the dataset
- A Bayesian-network view
- Feature robustness checks
- SonarQube file comparisons
- Leave-one-repository-out results
- SonarQube versus model feature-set scores
- Scoring a new repository from a generated or uploaded CSV

The dashboard is a presentation and exploration interface. It does not change
what the labels mean: the classification label is based on historical
revision activity, while the temporal experiment separately evaluates later
revision activity.

## 9. What the explanations mean

SHAP is used to explain a model decision. In plain language, it assigns each
input a contribution showing whether that input pushed the model's High-class
score up or down for a particular file, and roughly by how much on the model
scale.

For the local scoring report, the top three reasons are the inputs with the
largest absolute SHAP contributions to the High class. These are explanations
of the model's calculation. They do not prove that changing that input would
cause the file to become safer, more maintainable, or less likely to change.

Similarly, a correlation or feature-importance result tells us that a
measurement and the model's label are related in this dataset. It does not
show that the measurement causes file changes.

## 10. Main limitations and responsible interpretation

1. **The original target is not a defect label.** Revision-count terciles
   measure how often files were changed in the chosen history. They do not
   verify defects.
2. **History inputs and history labels are related.** Author count, age, and
   coupling come from Git history, while the label also comes from revisions.
   This can make historical classification easier without proving future
   predictive ability.
3. **Repositories are a small sample.** The five projects are established
   Python repositories. A team's codebase may use different practices,
   languages, structures, or commit habits.
   The scorer has only been exercised on generated tiny Git repositories as
   a functional test; no independent real team repository has yet been used
   to establish scoring accuracy.
4. **Results vary by repository.** The held-out Requests score was lower than
   the held-out Click score. Average performance hides such differences.
5. **The model's probabilities are model scores.** They should not be
   interpreted as calibrated real-world chances of bugs or future changes.
6. **Measurements are imperfect.** Tools can differ in what they count; source
   parsing, path matching, incomplete Git history, missing SonarQube values,
   and repository conventions can affect results.
7. **SonarQube and RiskLens have different purposes.** SonarQube's static
   findings can be useful for code review even if they do not improve this
   classifier's F1 score.
8. **The temporal test is promising, not final certification.** It uses
   cutoffs within five existing histories and a particular Random Forest
   setup. More independent repositories and later observation periods would
   be needed to establish broad reliability.
9. **Training cutoffs and feature definitions are part of the model.** If the
   training dataset or extraction process changes, retrain and regenerate the
   artifacts before comparing scores.
10. **Joblib files are executable Python artifacts.** Only load model files
    created by a trusted run of this project.

## 11. What is already working and what remains

### Implemented and checked

- A merged dataset and historical revision-based Low/Medium/High labels for
  381 files across five repositories.
- A nine-feature Random Forest analysis and feature-robustness report.
- A leave-one-repository-out evaluation.
- A temporal validation experiment using historical cutoffs and later
  revisions.
- A matched SonarQube-versus-model feature-set comparison.
- Full and static-only joblib scoring artifacts.
- A local repository scoring command that writes predictions and SHAP reasons.
- A dashboard tab for a generated or uploaded repository report.
- Tests using small Git repositories generated in temporary directories.

The new repository-scoring tests passed **3 tests**. Python compilation
completed, and a Streamlit smoke test found **11 tabs with no application
exceptions**. These checks show the implemented paths run in this environment;
they do not establish accuracy on an independent team's repository.

### Sensible next improvements

- Evaluate the scorer on one or more repositories not used for training and
  report results separately by repository.
- Expand the temporal evaluation to additional projects and observation
  periods.
- Decide whether the primary deliverable should be historical file
  classification or future change prediction; keep those evaluations clearly
  separate.
- Add a real externally verified outcome if the goal ever changes to bug
  prediction, and avoid treating SonarQube findings or commit-message keywords
  as confirmed bugs without validation.
- Check probability calibration and compare against simple baselines before
  presenting scores as numeric likelihoods.
- Keep extraction definitions and dependency versions documented so another
  person can reproduce the measurements.

## 12. Final plain-language description

RiskLens measures Python files and their Git histories, looks for patterns
associated with files that have changed more often, and presents those
patterns through a dashboard and a local scoring tool. It also tests whether
the patterns survive when a whole repository is held out, when the analysis
uses an earlier point in time, and when SonarQube measures are added.

The most defensible current conclusion is:

> In the five repositories studied, code and repository-history measurements
> contain useful information about historical file changes. A time-based test
> suggests that combining measurements available before a cutoff can help
> classify later revision activity better than using prior revision count
> alone. The evidence does not show that RiskLens detects bugs, predicts
> future defects, or is guaranteed to perform similarly on every team's
> repository.

The separate file `vitmate_academic_report_part3.md` describes a different
project, VITMate, and was not used as a source for this RiskLens report.
