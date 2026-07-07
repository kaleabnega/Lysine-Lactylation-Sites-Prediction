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

For the proposed architecture experiment, keep the same benchmark data and
inputs but replace the static CLS-plus-feature classifier with site-aware token
attention and gated fusion:

```bash
!python3 scripts/run_replication.py \
  --run independent \
  --architecture token_gated \
  --epochs 30 \
  --batch-size 4 \
  --device cuda \
  --optimizer adamw \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --scheduler linear \
  --warmup-ratio 0.1
```

If `token_gated` improves ranking metrics but not threshold metrics, try the
more conservative hybrid variant. It preserves the original CLS/global ProtBert
context and adds site-aware token pooling:

```bash
!python3 scripts/run_replication.py \
  --run independent \
  --architecture hybrid_gated \
  --epochs 30 \
  --batch-size 4 \
  --device cuda \
  --optimizer adamw \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --scheduler linear \
  --warmup-ratio 0.1
```

To test whether the token-gated framework transfers to another protein language
model, replace ProtBert with ESM-2. Start with a Colab-friendly ESM-2 checkpoint:

```bash
!python3 scripts/run_replication.py \
  --run independent \
  --architecture token_gated \
  --model-name facebook/esm2_t12_35M_UR50D \
  --epochs 30 \
  --batch-size 4 \
  --device cuda \
  --optimizer sgd \
  --learning-rate 0.003 \
  --weight-decay 0.0 \
  --scheduler none
```

If memory allows, try the stronger 150M checkpoint:

```bash
!python3 scripts/run_replication.py \
  --run independent \
  --architecture token_gated \
  --model-name facebook/esm2_t30_150M_UR50D \
  --epochs 30 \
  --batch-size 4 \
  --device cuda \
  --optimizer sgd \
  --learning-rate 0.003 \
  --weight-decay 0.0 \
  --scheduler none
```

The runner uses `--sequence-format auto` by default: ProtBert receives spaced
residues, ProtT5 receives spaced residues with rare amino acids mapped to `X`,
and ESM-style models receive raw amino-acid strings.

To add a T5-family protein language model to the backbone comparison table, use
the encoder-only half-precision ProtT5 checkpoint. Start with the encoder frozen
so Colab trains only the token-gated prediction head while using ProtT5 as a
feature extractor:

```bash
!python3 scripts/run_replication.py \
  --run independent \
  --architecture token_gated \
  --model-name Rostlab/prot_t5_xl_half_uniref50-enc \
  --encoder-layers 4 \
  --freeze-encoder \
  --epochs 30 \
  --batch-size 2 \
  --device cuda \
  --optimizer adamw \
  --learning-rate 1e-4 \
  --weight-decay 0.01 \
  --scheduler linear \
  --warmup-ratio 0.1
```

If memory allows and the frozen-encoder result is promising, try fine-tuning a
small truncated ProtT5 encoder. Keep the batch size small:

```bash
!python3 scripts/run_replication.py \
  --run independent \
  --architecture token_gated \
  --model-name Rostlab/prot_t5_xl_half_uniref50-enc \
  --encoder-layers 4 \
  --epochs 30 \
  --batch-size 1 \
  --device cuda \
  --optimizer adamw \
  --learning-rate 1e-5 \
  --weight-decay 0.01 \
  --scheduler linear \
  --warmup-ratio 0.1
```

If the single-seed run is promising, evaluate the same architecture as a
three-seed ensemble:

```bash
!python3 scripts/run_replication.py \
  --run ensemble-independent \
  --architecture token_gated \
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

## Proposed Architecture Variant

The `--architecture token_gated` model is designed as a deliberate
same-dataset architectural change:

```text
51-aa sequence
  -> ProtBert token embeddings
  -> site-aware attention pooling around the central lysine
  -> sequence representation

27 physicochemical features
  -> feature projection
  -> physicochemical representation

sequence representation + physicochemical representation
  -> learned gated fusion
  -> residual MLP classifier
  -> Kla probability
```

The justification is:

- Kla is site-specific, so token-level residue information around the central
  lysine should be more informative than a single static pooled vector.
- The physicochemical vector is a second modality, so learned gating is a more
  controlled fusion mechanism than blind concatenation.
- The dataset and feature sources remain unchanged, preserving fair comparison
  with PCBert-Kla.

Recommended ablation order:

```text
1. baseline + SGD
2. baseline + AdamW
3. token_gated + AdamW
4. hybrid_gated + AdamW
5. token_gated + SGD
6. token_gated + SGD + alternate PLM backbone, such as ESM-2
7. best architecture + optimizer setting + seed ensemble
```

`hybrid_gated` is the conservative follow-up variant: it keeps the baseline CLS
embedding and augments it with the central-lysine-aware token-pooled embedding
before gated fusion with physicochemical features.

For backbone ablations, keep the dataset, physicochemical features, architecture,
optimizer, and evaluation protocol fixed; only change `--model-name`.
