# Dataset Primer: What Is in the HDFS Logs

Read this first if you have never touched HDFS or log data.

## 0. The problem we are solving

Big systems write millions of log lines a day, far too many to read. When something breaks,
engineers dig through logs for hours. We automate the first step: for each unit of
activity, decide **did something go wrong, yes or no?**

| Piece | Role |
|---|---|
| **HDFS** (the monitored system) | Does the real work. Failures happen inside it. |
| **Its logs** | The evidence. This is our dataset. |
| **Our models** (the monitoring layer) | Read the logs and flag abnormal activity. This is what we build and evaluate. |

- This is **detection, not diagnosis**: we say *whether* a block went wrong, not *where* or
  *why*.
- We do not evaluate or fix HDFS. We evaluate how well each of our four models finds its
  failures, using the researchers' labels.
- HDFS is just our example system. It was chosen because it is the standard benchmark with
  trustworthy labels. Nothing we build depends on how replication works.
- The research question (supervised vs. unsupervised) is in [overview.md](overview.md).

## 1. What is HDFS?

**HDFS = Hadoop Distributed File System.** It stores very large files across many
machines, so no single machine has to hold a whole file.

- A file is cut into fixed-size chunks called **blocks** (64 MB in this dataset, see
  `size 67108864` in the logs).
- Each block is stored on several machines (normally **3 copies**, called replicas), so
  losing one machine loses no data.
- Why blocks? Files are too big for one disk, and big chunks keep bookkeeping small. The
  block is the unit that gets stored, copied and tracked.

Two kinds of machines:

| Role | What it does |
|---|---|
| **NameNode** | The coordinator. Keeps the map: file → blocks → which machines hold them. Stores no file data. |
| **DataNode** | A worker. Stores the actual block data. This dataset ran on 203 of them. |

## 2. Where do the logs come from?

Each HDFS machine writes a log of what it is doing. These are not application or business
logs. They are the storage system's own diary of its internal operations. The logs
record, line by line, things like "I started receiving block X" or "I deleted block X".

## 3. Anatomy of a raw log line

```
081109 204655 556 INFO dfs.DataNode$PacketResponder: Received block blk_3587508140051953248 of size 67108864 from /10.251.42.84
```

| Piece | Meaning |
|---|---|
| `081109` | date (9 Nov 2008) |
| `204655` | time (20:46:55) |
| `556` | process/thread ID |
| `INFO` | log level |
| `dfs.DataNode$PacketResponder` | which part of HDFS wrote it |
| rest | the message |

Inside the message:

- **`blk_3587508140051953248`** is a **block ID**, the unique name of one block.
- **`/10.251.42.84`** is an **IP address**: which machine the data came from. `10.x.x.x` is a
  private network address. `:50010` after an IP is the port DataNodes listen on.
- **`src` / `dest`** (seen in E5) are the sender and receiver of one data transfer.

Block IDs and IPs change from line to line. They are identifiers, not behaviour.

Replication is a **chain**, not a fan-out: the writer sends the block once to DataNode 1,
which stores it and forwards it to DataNode 2, which forwards it to DataNode 3. This saves
the writer's network bandwidth, and data streams through in small packets, so the extra hops
add little delay. Each hop logs its own "Receiving block ... src ... dest ..." line.

## 4. Templates and event IDs (E1 ... E30)

The same *kind* of message appears millions of times with different IDs and IPs:

```
Receiving block blk_579...  src: /10.251.30.6:33145 dest: /10.251.30.6:50010
Receiving block blk_123...  src: /10.250.1.9:41000  dest: /10.250.4.2:50010
```

A **template** keeps the fixed words and replaces the changing parts with `<*>`:

```
Receiving block <*> src: <*> dest: <*>
```

Each template gets a short label, its **event ID**: this one is **E5**. So `E5` simply
means "a DataNode started receiving a block". Turning raw lines into templates is called log
parsing (tools such as Drain do this). We did not run a parser ourselves: Loghub already
provides the parsed templates and per-block event sequences. 30 templates are defined
(E1-E30), and **E30 never occurs**, so the data uses 29 event types.

An event ID is a **message category**, not a machine type: some come from DataNodes (E5), some
from the NameNode (E22, E26). The categories come from log parsing, done with fixed rules
before any model runs. The models do not learn the categories, they learn patterns over them.
This shrinks millions of different-looking lines into 29 types.

Events used in the example below:

| Event | Template | Plain meaning |
|---|---|---|
| E5 | `Receiving block <*> src: <*> dest: <*>` | a DataNode starts receiving a block |
| E22 | `BLOCK* NameSystem.allocateBlock: <*>` | the NameNode creates/allocates a new block |
| E11 | `PacketResponder <*> for block <*> terminating` | the DataNode's acknowledger thread for that block finishes |
| E9 | `Received block <*> of size <*> from <*>` | a DataNode has fully received the block |
| E26 | `BLOCK* NameSystem.addStoredBlock: blockMap updated: <*> is added to <*> size <*>` | the NameNode records that a DataNode now holds the block |
| E3 | `Served block <*> to <*>` | the block was read by someone |
| E2 | `Verification succeeded for <*>` | the stored block passed an integrity check |
| E23 | `BLOCK* NameSystem.delete: <*> is added to invalidSet of <*>` | the NameNode marks the block for deletion |
| E21 | `Deleting block <*> file <*>` | a DataNode deletes the block |

(`PacketResponder` is the DataNode thread that acknowledges each chunk of data as it
arrives, so the writer knows it was stored.)

## 5. A session, step by step

A **session** is all log events for one block ID, in time order. It is one data point.
The proposal's first normal example, `blk_7503483334202473044`:

```
[E5, E5, E22, E5, E11, E9, E11, E9, E11, E9, E26, E26, E26, E3, E2, E2, E23, E23, E23, E21, E21, E21]
```

Read as the block's life story:

1. **E22** the NameNode allocates the block.
2. **E5 x3** three DataNodes begin receiving it (one per replica: the 3 copies).
3. **E11, E9 x3** each DataNode finishes receiving it and its acknowledger terminates.
4. **E26 x3** the NameNode records that all 3 DataNodes now hold the block.
5. **E3, E2 x2** the block is read once and verified.
6. **E23 x3, E21 x3** the block is marked for deletion, and all 3 copies are deleted.

Notice the "x3" pattern everywhere: three replicas. A **normal** session has this regular
shape. An **anomalous** session breaks it, for example the proposal's
`[E5, E5, E22, E7, E11, E10, E14, E7]` contains E7 (`writeBlock <*> received exception
<*>`) and E10/E14 (exceptions) and never reaches the "stored and recorded" steps. That is
the signal the models try to learn.

## 6. How the dataset is labelled

Sessions were labelled normal or anomalous by the original researchers using handcrafted
rules. We do no labelling ourselves. 575,061 sessions, 16,838 (2.93%) anomalous.

## 7. Two ways to feed a session to a model

**Event sequence:** keep the ordered list as is. Used by the LSTM, which needs order.

**Count vector:** count each event, ignore order. 29 event types, so each session
becomes a row of 29 numbers:

| | E2 | E3 | E5 | E9 | E11 | E21 | E22 | E23 | E26 | ... |
|---|---|---|---|---|---|---|---|---|---|---|
| example above | 2 | 1 | 3 | 3 | 3 | 3 | 1 | 3 | 3 | ... |

Used by Logistic Regression, XGBoost and Isolation Forest.

**TF-IDF** reweights counts: events found in almost every session (E11) count for less,
rare events (E7) count for more. Roughly `count x log(total sessions / sessions containing
the event)`.

**n-grams** (`ngram_range`) count runs of consecutive events as one feature. `(1,1)` counts
single events only. `(1,2)` also counts pairs, so `[E5, E22, E7]` adds `E5->E22` and
`E22->E7`. This gives count-based models a little order information at the cost of many
more columns.

## 8. Source files (in `data/raw/preprocessed/`)

| File | Contents |
|---|---|
| `anomaly_label.csv` | block ID and Normal/Anomaly label |
| `Event_traces.csv` | block ID, the ordered event sequence, plus extra columns (see below) |
| `Event_occurrence_matrix.csv` | the count vector per session |
| `HDFS.log_templates.csv` | the E1-E30 template list |
| `HDFS.npz` | the same sessions as NumPy arrays |

`HDFS.log` (1.5 GB) is the full raw log.

## 9. Data quality findings

- No missing labels or sequences, no duplicate block IDs.
- `Event_traces.csv` has a **`Type`** column that is empty for every normal session and
  filled only for anomalous ones. **This leaks the label. Never use it.**
- `Latency` and `TimeInterval` are not part of the agreed input (event IDs only), so we
  do not use them.
- E22 appears exactly once in every session, so it has no information as a count feature.
- Several events are highly correlated (E18/E25 = 1.0, E9/E11 = 0.996, E21/E23 = 0.994,
  E6/E16/E18/E25 about 0.98). This matters for Logistic Regression, not for tree models.
- Session length: median 19 events, maximum 298.
- These statistics were computed on the full dataset for understanding only. Any real
  feature selection must use the **training split only**.
