# Lysine Lactylation Site Prediction

This repository contains a reproducible research framework for predicting
lysine lactylation (Kla) sites using the PCBert-Kla benchmark dataset. The
project keeps the original baseline code and data separate from the cleaned
experimental implementation so that the results can be audited, rerun, and
compared fairly against the published PCBert-Kla model.

## Research Goal

The goal is to improve Kla site prediction while using the same training and
independent test data as the baseline paper. This makes the comparison focused
on the modeling framework rather than on changes to the dataset.

The active proposed model combines:

- a protein language model backbone, primarily ProtBert,
- central-lysine-guided token attention pooling,
- 27 physicochemical and residue-composition features,
- gated fusion between sequence and physicochemical representations,
- a residual MLP classifier for binary Kla prediction.

The repository also supports backbone comparisons with ESM-2, ProtT5, and Ankh.

## Main Entry Points

```text
experiment/pcbert_kla_clean/main.ipynb
```

The notebook is the main research workbook. It is intended for Google Colab or
a local Jupyter environment and includes:

- dataset checks,
- baseline replication,
- proposed token-gated model runs,
- protein language model backbone comparison,
- ablation study,
- metric tables generated from executed cells,
- ROC and precision-recall curves,
- confusion matrix,
- t-SNE visualization,
- biological interpretation plots.

```text
experiment/pcbert_kla_clean/README.md
```

This contains the detailed command-line instructions and experiment notes.

## Repository Layout

```text
baselines/
└── PCBert-Kla-original/          # original PCBert-Kla reference code/data

experiment/
└── pcbert_kla_clean/             # active reproducible implementation
    ├── main.ipynb                # main research notebook
    ├── README.md                 # detailed experiment commands
    ├── requirements.txt
    ├── scripts/run_replication.py
    └── src/pcbert_kla_clean/
        ├── backbones.py          # ProtBert, ESM-2, ProtT5, Ankh helpers
        ├── data.py               # dataset parsing and audits
        ├── datasets.py           # PyTorch Dataset/DataLoader logic
        ├── experiments.py        # CV, independent test, ensemble runners
        ├── metrics.py            # ACC, AUC, AUPRC, F1, MCC, Pre, Rec, SP
        ├── model.py              # baseline and token-gated architectures
        ├── training.py           # training/evaluation loops
        └── utils.py              # reproducibility utilities

experiment-1/
└── structure_aware_kla/          # earlier exploratory provenance work
```

## Dataset

The active experiments use the original PCBert-Kla benchmark files:

```text
baselines/PCBert-Kla-original/data/train.csv
baselines/PCBert-Kla-original/data/test.csv
baselines/PCBert-Kla-original/data/feature_train.csv
baselines/PCBert-Kla-original/data/feature_test.csv
```

The sequence input is a 51-amino-acid window centered on a candidate lysine.
The tabular input contains 27 physicochemical and residue-composition features.
The same train/test split is retained for fair comparison with the baseline
paper.

## Representative Independent-Test Results

The strongest single-model result so far is the ProtBert token-gated model
trained with SGD.

| Model / Setting | ACC | AUC | AUPRC | F1 | MCC | Pre | Rec | SP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PCBert-Kla paper | 0.9497 | 0.9646 | 0.9523 | 0.9505 | 0.8999 | 0.9364 | 0.9650 | 0.9345 |
| Original architecture + SGD replication | 0.9407 | 0.9836 | 0.9818 | 0.9428 | 0.8837 | 0.9105 | 0.9774 | 0.9040 |
| Original architecture + AdamW | 0.9548 | 0.9801 | 0.9659 | 0.9556 | 0.9101 | 0.9399 | 0.9718 | 0.9379 |
| ProtBert token-gated + SGD | 0.9633 | 0.9715 | 0.9585 | 0.9636 | 0.9267 | 0.9556 | 0.9718 | 0.9548 |
| Ankh-base token-gated frozen | 0.9605 | 0.9746 | 0.9699 | 0.9607 | 0.9210 | 0.9553 | 0.9661 | 0.9548 |

The paper-reported metrics used for comparison are stored in:

```text
experiment/pcbert_kla_clean/outputs/paper_reported_metrics.json
```

## Quick Start

From the repository root:

```bash
cd experiment/pcbert_kla_clean
pip install -r requirements.txt
python3 scripts/run_replication.py --run data-check
```

The data check does not download a protein language model. It validates the
benchmark files and prints class counts, feature dimensions, duplicate sequence
counts, and train/test overlap information.

## Best Current Single-Model Run

```bash
cd experiment/pcbert_kla_clean
python3 scripts/run_replication.py \
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

## Google Colab

After cloning the repository in Colab:

```bash
%cd /content/Lysine-Lactylation-Sites-Prediction/experiment/pcbert_kla_clean
!pip install -r requirements.txt
!python3 scripts/run_replication.py --run data-check
```

Then open and run:

```text
experiment/pcbert_kla_clean/main.ipynb
```

The notebook is designed so that metric tables are generated from executed
experiment cells rather than hardcoded manually.

## Biological Interpretation

The notebook includes interpretation plots for:

- amino-acid enrichment around the central lysine,
- physicochemical feature differences between lactylated and non-lactylated
  samples,
- model attention across the lysine-centered sequence window.

These analyses are intended as supportive biological interpretation. They show
that the model uses local sequence context and physicochemical differences that
are consistent with Kla-site variation, but they should not be interpreted as
causal biological proof.

## Notes

- The active proposed-method code is in `experiment/pcbert_kla_clean/`.
- The original cloned baseline is retained in `baselines/PCBert-Kla-original/`.
- `experiment-1/structure_aware_kla/` contains earlier exploratory work and is
  not required for the current token-gated benchmark experiments.
- Research documents and local prompt files are intentionally ignored by git.
