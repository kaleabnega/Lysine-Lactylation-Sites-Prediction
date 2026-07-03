# Clean PCBert-Kla Replication

This folder contains a cleaned replication scaffold for the upstream
`PCBert-Kla` notebook in `../../baselines/PCBert-Kla-original`.

The goal is to keep the original cloned baseline untouched while making a
version that can be run and audited on Google Colab or a local GPU machine.

## What This Fixes

- Parses the upstream `train.csv` and `test.csv` files as FASTA-like files.
- Uses the provided `feature_train.csv` and `feature_test.csv` tables.
- Removes hardcoded author paths such as `/home/hqzhang/...`.
- Removes hardcoded GPU selection such as `cuda:2`.
- Fixes the model `forward()` method so it uses its actual input argument.
- Fits the feature scaler inside each training fold to avoid validation leakage.
- Reports the metrics used in the paper: ACC, Rec, Pre, AUC, MCC, F1, SP, AUPRC.

## Local Data Check

From the repository root:

```bash
python3 replication/pcbert_kla_clean/scripts/run_replication.py --run data-check
```

This does not download ProtBert or train anything. It only validates the files
and prints dataset counts, duplicate sequence counts, and train/test overlap.

## Colab Quick Start

In a Colab notebook, after cloning your research repository:

```bash
%cd /content/Dr-Mobeen-Research/replication/pcbert_kla_clean
!pip install -r requirements.txt
!python3 scripts/run_replication.py --run data-check
```

For five-fold cross-validation:

```bash
!python3 scripts/run_replication.py \
  --run cv \
  --epochs 30 \
  --batch-size 4 \
  --device cuda \
  --save-models
```

For an independent test run:

```bash
!python3 scripts/run_replication.py \
  --run independent \
  --epochs 30 \
  --batch-size 4 \
  --device cuda \
  --save-models
```

For validation-calibrated thresholding, choose the threshold on the held-out
validation subset and then apply that fixed threshold to the independent test
set:

```bash
!python3 scripts/run_replication.py \
  --run independent \
  --epochs 30 \
  --batch-size 4 \
  --device cuda \
  --calibrate-threshold MCC \
  --save-models
```

Use `--calibrate-threshold F1` if the goal is to maximize validation F1
instead of MCC. The independent test set is not used to choose the threshold.
Prediction CSVs are saved by default in `outputs/` for auditability.

For seed ensembling, train several independent models and average their
independent-test probabilities:

```bash
!python3 scripts/run_replication.py \
  --run ensemble-independent \
  --epochs 30 \
  --batch-size 4 \
  --device cuda \
  --ensemble-seeds 42,123,2025
```

This is more expensive than a single independent run because it trains one
model per seed. Start with three seeds on free Colab; use five seeds if the
session has enough time:

```bash
!python3 scripts/run_replication.py \
  --run ensemble-independent \
  --epochs 30 \
  --batch-size 4 \
  --device cuda \
  --ensemble-seeds 42,123,2025,3407,777
```

For optimizer modernization, switch from the paper's SGD setup to AdamW:

```bash
!python3 scripts/run_replication.py \
  --run independent \
  --epochs 30 \
  --batch-size 4 \
  --device cuda \
  --optimizer adamw \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --scheduler linear \
  --warmup-ratio 0.1
```

If that is promising, combine AdamW with the ensemble mode:

```bash
!python3 scripts/run_replication.py \
  --run ensemble-independent \
  --epochs 30 \
  --batch-size 4 \
  --device cuda \
  --optimizer adamw \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --scheduler linear \
  --warmup-ratio 0.1 \
  --ensemble-seeds 42,123,2025
```

Free Colab GPU memory may be tight because ProtBert is large. If the session
runs out of memory, first try reducing `--batch-size` to `2` or `1`.

## Scientific Notes

The upstream data matches the paper's sample counts:

- Training set: 1720 Kla and 1767 non-Kla records.
- Test set: 177 Kla and 177 non-Kla records.
- Every sequence window has length 51.
- The encoded lysine position is 26 using 1-based indexing.

There is train/test sequence overlap in the provided data. The data-check
command reports this explicitly. We should document this in any replication
notes because overlap can inflate independent-test performance.

## Default Modeling Choices

The default runner is faithful to the paper/notebook:

- `Rostlab/prot_bert`
- first 4 encoder layers retained
- physicochemical feature vector dimension 27
- SGD optimizer
- learning rate `0.003`
- weight decay `0.0`
- no scheduler
- batch size `4`
- 30 epochs
- 5-fold shuffled KFold with seed `42`
- fixed decision threshold `0.5` unless `--calibrate-threshold` is provided

For stricter follow-up experiments, prefer `--splitter stratified` and later a
homology-aware split.
