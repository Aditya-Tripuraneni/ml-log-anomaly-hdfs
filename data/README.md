# Data

Data files are gitignored. Download HDFS_v1 into `data/raw/` with
`uv run python scripts/download_data.py` (it downloads and extracts the Loghub archive).

- Dataset: Loghub HDFS_v1, <https://github.com/logpai/loghub> (mirror with preprocessed
  files: <https://github.com/logpai/loglizer/tree/master/data/HDFS>)
- Preprocessed files (extracted to `data/raw/preprocessed/`): `HDFS.log_templates.csv`,
  `anomaly_label.csv`, `Event_traces.csv`, `Event_occurrence_matrix.csv`, `HDFS.npz`.
  The code reads `Event_traces.csv` and `anomaly_label.csv`.
- Expected size: 575,061 block sessions, 16,838 (2.93%) anomalous

`data/processed/` holds anything derived from raw data (parsed sequences, split indices).
Regenerate it from code rather than committing it.
