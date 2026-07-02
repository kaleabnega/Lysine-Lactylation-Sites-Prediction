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
- batch size `4`
- 30 epochs
- 5-fold shuffled KFold with seed `42`

For stricter follow-up experiments, prefer `--splitter stratified` and later a
homology-aware split.
