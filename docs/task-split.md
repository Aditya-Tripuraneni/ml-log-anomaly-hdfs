# Task Split

Each person owns their models end to end: script, training, evaluation, and presenting
results at team meetings. Everyone reviews someone else's work.

| Owner | Models / work | Status |
|---|---|---|
| Aditya | Logistic Regression (supervised) | Done: model, script, tests, metrics, confusion matrix |
| Aditya | XGBoost (supervised) | Not started (`models/xgboost_model.py` is a stub) |
| Aditya | Shared pipeline (`data.py`, `features.py`, `evaluation.py`, `plots.py`) | Done |
| Aditya | Label-fraction experiment (1% / 10% / 100%) | Not started |
| Vanessa | Isolation Forest (unsupervised) | Done: model, script, tests, metrics, confusion matrix |
| Harmanpreet | DeepLog-style LSTM (unsupervised) | Done: model, script, tests, metrics, confusion matrix (trained on a random 50,000-session subset; the PCA fallback was not needed) |

Teammates: Harmanpreet Singh Sagar, Vanessa Lai.

## Contributions so far

**Aditya Tripuraneni**
- Set up the repository: uv project, `src/` layout, docs, `.gitignore`, CPU-only torch and `xgboost-cpu`.
- Built the shared pipeline: data loading and the stratified 70/10/20 split (`data.py`), count-vector features (`features.py`), metrics and the F1 threshold helper (`evaluation.py`), and the confusion-matrix plot (`plots.py`), with tests.
- Wrote the dataset download script and the dataset primer.
- Implemented and evaluated the Logistic Regression baseline.
- Wrote the report's Dataset section (source, properties, preprocessing, annotation) and set up its IEEE citations.
- Wrote the report's Related Work, the logistic regression paragraph of Implementation and the issues-encountered paragraph.

**Vanessa Lai**
- Added the LaTeX report template.
- Implemented and evaluated Isolation Forest: model, run script, tests, metrics and confusion matrix.
- Wrote the report's Introduction, the Train/Validation/Test Splits subsection, the Isolation Forest paragraph of Implementation, the TA Feedback subsection and the future plans.
- Set up the report build: LaTeX build-file ignores in `.gitignore`, the compiled report PDF, and the Isolation Forest reference in the bibliography.

**Harmanpreet Singh Sagar**
- Implemented and evaluated the DeepLog-style LSTM: model, run script, tests, metrics and confusion matrix, plus `features.event_sequences`.
- Wrote the report's Features and Results and Evaluation sections.
- Added the top-k ablation of DeepLog's original rule next to the probability-based score, and the DeepLog and PR-AUC references in the bibliography.

**Everyone**
- Reviewed each other's work, cleaned the unused template sections out of the report, added the confusion-matrix figure and filled in Team Contributions.

## Shared work (everyone)

- [x] Download HDFS_v1 and confirm the 575,061 / 16,838 counts
- [x] Sessionization and label join, using Loghub's pre-parsed event traces (we did not run Drain ourselves)
- [x] Shared split and feature code merged to `main` before model work started
- [ ] Final comparison table and plots (results table and confusion matrices are in the report; XGBoost and the cross-model plots are pending)
- [ ] Project submissions and write-up (in progress: Related Work, Dataset, Features, Results, issues encountered and the logistic regression and Isolation Forest paragraphs are written; the LSTM and XGBoost Implementation paragraphs are still to do)
