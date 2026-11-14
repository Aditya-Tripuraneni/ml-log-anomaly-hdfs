# Progress Log

Newest first. One line per update: date, who, what.

- 2026-11-13, Aditya: label-fraction experiment (logistic regression and XGBoost on 1%, 10%, 100% of training labels, several draws), repetition analysis script that saves the dataset-duplication statistics cited in the report.
- 2026-11-12, Aditya: XGBoost on count vectors (class weighting via `scale_pos_weight`, tree depth and number of trees tuned on validation PR-AUC, F1 cutoff from validation), with run script, tests, metrics and confusion matrix.
- 2026-11-10, Vanessa: switched the report's citations to the template's ACL author-year style.
- 2026-11-09, Team: reviewed each other's work, cleaned the report of unused template sections (example figures, citation demos, unused bib entries), added the confusion-matrix figure, filled in Team Contributions and updated the task split with statuses and contributions.
- 2026-11-09, Vanessa: report Introduction, TA Feedback and Isolation Forest paragraph of Implementation.
- 2026-11-09, Harmanpreet: report Features and Results and Evaluation sections.
- 2026-11-09, Harmanpreet: DeepLog-style LSTM on ordered event sequences (end-of-session token, max-surprise score, top-k ablation); implemented `features.event_sequences`.
- 2026-11-06, Vanessa: Isolation Forest on count vectors (inverted inlier scores, shared metrics); report motivation, future plans and splits subsection.
- 2026-11-02, Aditya: report Dataset section and the verified bibliography.
- 2026-11-02, Vanessa: added the LaTeX report template.
- 2026-10-25, Aditya: Logistic Regression baseline with the shared pipeline (data split, count vectors, metrics, F1 threshold, confusion-matrix plot) and tests.
- 2026-10-23, Aditya: repo scaffolded (uv project, `src/` layout, docs, .gitignore); dataset primer and download script.
