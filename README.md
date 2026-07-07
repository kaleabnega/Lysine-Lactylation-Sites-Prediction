# Lysine Lactylation Site Prediction

This repository contains a research implementation for lysine lactylation
site prediction based on the PCBert-Kla benchmark. The project keeps the
original baseline code separate from the cleaned and extended experimental
framework so that results can be reproduced and audited.

## Repository Layout

```text
baselines/
├── PCBert-Kla-original/          # original cloned PCBert-Kla reference code/data

experiment/
└── pcbert_kla_clean/             # active reproducible benchmark and proposed method
    ├── README.md                 # detailed commands and experiment notes
    ├── requirements.txt
    ├── scripts/run_replication.py
    └── src/pcbert_kla_clean/
        ├── backbones.py          # ProtBert, ESM-2, ProtT5, Ankh loading helpers
        ├── data.py               # dataset parsing and audit utilities
        ├── datasets.py           # PyTorch dataset/collate logic
        ├── experiments.py        # CV, independent test, ensemble runners
        ├── metrics.py            # ACC, AUC, AUPRC, F1, MCC, Pre, Rec, SP
        ├── model.py              # PCBert-Kla baseline and token-gated architectures
        ├── training.py           # training/evaluation and optimizer setup
        └── utils.py

experiment-1/
└── structure_aware_kla/          # earlier dataset provenance / structure-aware exploration

research-documents/               # papers, diagrams, and research notes
```

## Main Experiment

The active work is in:

```text
experiment/pcbert_kla_clean/
```

This framework supports:

- original PCBert-Kla style replication,
- five-fold cross-validation,
- independent test evaluation,
- seed ensembles,
- the proposed token-gated architecture,
- protein language model backbone comparisons using ProtBert, ESM-2, ProtT5,
  and Ankh.

The current proposed architecture keeps the same benchmark data for fair
comparison, but changes the downstream neural architecture by using
site-aware token attention and gated fusion with physicochemical features.

## Quick Start

From the repository root:

```bash
cd experiment/pcbert_kla_clean
pip install -r requirements.txt
python3 scripts/run_replication.py --run data-check
```

The data check does not download any protein language model. It only verifies
the benchmark data and prints class counts, feature dimensions, duplicate
sequence counts, and train/test overlap.

## Colab Setup

After cloning the repository in Google Colab:

```bash
%cd /content/Lysine-Lactylation-Sites-Prediction/experiment/pcbert_kla_clean
!pip install -r requirements.txt
!python3 scripts/run_replication.py --run data-check
```

## Representative Commands

Token-gated ProtBert independent test:

```bash
!python3 scripts/run_replication.py \
  --run independent \
  --architecture token_gated \
  --epochs 30 \
  --batch-size 4 \
  --device cuda \
  --optimizer sgd \
  --learning-rate 0.003 \
  --weight-decay 0.0 \
  --scheduler none
```

ESM-2 35M backbone:

```bash
!python3 scripts/run_replication.py \
  --run independent \
  --architecture token_gated \
  --model-name facebook/esm2_t12_35M_UR50D \
  --epochs 30 \
  --batch-size 4 \
  --device cuda \
  --optimizer adamw \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --scheduler linear \
  --warmup-ratio 0.1
```

Ankh-base frozen backbone:

```bash
!python3 scripts/run_replication.py \
  --run independent \
  --architecture token_gated \
  --model-name ElnaggarLab/ankh-base \
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

## Notes

- The original baseline folder should be treated as a reference copy.
- The active proposed-method code is in `experiment/pcbert_kla_clean/`.
- The `experiment-1/structure_aware_kla/` folder is exploratory provenance work
  and is not required for the current token-gated benchmark runs.
- Detailed commands and additional backbone experiments are documented in
  `experiment/pcbert_kla_clean/README.md`.
