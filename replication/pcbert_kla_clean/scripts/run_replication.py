from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from pcbert_kla_clean.data import (  # noqa: E402
    KlaSplit,
    describe_split,
    load_split,
    train_test_overlap_report,
)


def ensure_ml_deps() -> None:
    global AutoTokenizer
    global BertTokenizer
    global DataLoader
    global KFold
    global MinMaxScaler
    global PCBertKla
    global StratifiedKFold
    global compute_binary_metrics
    global joblib
    global nn
    global summarize_metric_rows
    global torch
    global tqdm
    global train_test_split

    import joblib
    import torch
    from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from torch import nn
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer, BertTokenizer

    from pcbert_kla_clean.metrics import compute_binary_metrics, summarize_metric_rows
    from pcbert_kla_clean.model import PCBertKla


class KlaDataset:
    def __init__(
        self,
        sequences: list[str],
        features: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.sequences = sequences
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[str, np.ndarray, np.float32]:
        return self.sequences[index], self.features[index], self.labels[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_protbert_sequence(sequence: str) -> str:
    return " ".join(sequence)


def make_collate_fn(tokenizer, max_length: int):
    def collate(batch):
        sequences, features, labels = zip(*batch)
        encoded = tokenizer(
            [format_protbert_sequence(sequence) for sequence in sequences],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        return {
            "encoded": encoded,
            "features": torch.tensor(np.asarray(features), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.float32),
        }

    return collate


def train_one_epoch(
    model: PCBertKla,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in tqdm(loader, leave=False, desc="train"):
        encoded = {key: value.to(device) for key, value in batch["encoded"].items()}
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(features=features, **encoded)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses))


def evaluate(
    model: PCBertKla,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    losses: list[float] = []
    labels_all: list[float] = []
    scores_all: list[float] = []

    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc="eval"):
            encoded = {key: value.to(device) for key, value in batch["encoded"].items()}
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(features=features, **encoded)
            loss = criterion(outputs, labels)
            losses.append(float(loss.detach().cpu()))
            labels_all.extend(labels.detach().cpu().numpy().tolist())
            scores_all.extend(outputs.detach().cpu().numpy().tolist())

    metrics = compute_binary_metrics(np.asarray(labels_all), np.asarray(scores_all))
    return float(np.mean(losses)), metrics


def make_loader(
    split: KlaSplit,
    indices: np.ndarray,
    scaled_features: np.ndarray,
    tokenizer,
    batch_size: int,
    shuffle: bool,
    max_length: int,
) -> DataLoader:
    dataset = KlaDataset(
        sequences=[split.sequences[index] for index in indices],
        features=scaled_features[indices],
        labels=split.labels[indices],
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_collate_fn(tokenizer, max_length=max_length),
    )


def build_model(args: argparse.Namespace, feature_dim: int, device: torch.device) -> PCBertKla:
    model = PCBertKla(
        model_name=args.model_name,
        feature_dim=feature_dim,
        encoder_layers=args.encoder_layers,
        cache_dir=args.cache_dir,
    )
    return model.to(device)


def load_tokenizer(model_name: str, cache_dir: str | None):
    if model_name == "Rostlab/prot_bert":
        return BertTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            do_lower_case=False,
        )

    return AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        use_fast=False,
    )


def run_data_check(train_split: KlaSplit, test_split: KlaSplit) -> None:
    report = {
        "train": describe_split(train_split),
        "test": describe_split(test_split),
        "train_test_overlap": train_test_overlap_report(train_split, test_split),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def run_cv(args: argparse.Namespace, train_split: KlaSplit) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = load_tokenizer(args.model_name, args.cache_dir)
    labels = train_split.labels

    if args.splitter == "stratified":
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        fold_iter = splitter.split(train_split.features, labels)
    else:
        splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        fold_iter = splitter.split(train_split.features)

    fold_metrics: list[dict[str, float]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for fold_id, (train_idx, val_idx) in enumerate(fold_iter, start=1):
        print(f"\nFold {fold_id}/{args.folds}")
        scaler = MinMaxScaler()
        scaled = np.zeros_like(train_split.features, dtype=np.float32)
        scaled[train_idx] = scaler.fit_transform(train_split.features[train_idx])
        scaled[val_idx] = scaler.transform(train_split.features[val_idx])

        train_loader = make_loader(
            train_split, train_idx, scaled, tokenizer, args.batch_size, True, args.max_length
        )
        val_loader = make_loader(
            train_split, val_idx, scaled, tokenizer, args.batch_size, False, args.max_length
        )
        model = build_model(args, feature_dim=train_split.features.shape[1], device=device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

        best_metrics: dict[str, float] | None = None
        best_score = -np.inf
        patience_left = args.patience

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, metrics = evaluate(model, val_loader, criterion, device)
            print(
                f"epoch={epoch:02d} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} ACC={metrics['ACC']:.4f} "
                f"MCC={metrics['MCC']:.4f} AUC={metrics['AUC']:.4f}"
            )

            monitor_score = metrics[args.monitor_metric]
            if monitor_score > best_score:
                best_score = monitor_score
                best_metrics = metrics
                patience_left = args.patience
                if args.save_models:
                    torch.save(
                        model.state_dict(),
                        args.output_dir / f"pcbert_kla_fold_{fold_id}.pt",
                    )
                    joblib.dump(
                        scaler,
                        args.output_dir / f"feature_scaler_fold_{fold_id}.joblib",
                    )
            else:
                patience_left -= 1
                if args.patience > 0 and patience_left <= 0:
                    print(f"early stopping at epoch {epoch}")
                    break

        if best_metrics is None:
            raise RuntimeError("No validation metrics were produced")
        fold_metrics.append(best_metrics)
        print("best:", json.dumps(best_metrics, sort_keys=True))

    print("\nCV mean:")
    print(json.dumps(summarize_metric_rows(fold_metrics), indent=2, sort_keys=True))


def run_independent(args: argparse.Namespace, train_split: KlaSplit, test_split: KlaSplit) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = load_tokenizer(args.model_name, args.cache_dir)

    if args.validation_fraction > 0:
        train_idx, val_idx = train_test_split(
            np.arange(len(train_split.labels)),
            test_size=args.validation_fraction,
            random_state=args.seed,
            stratify=train_split.labels,
        )
    else:
        train_idx = np.arange(len(train_split.labels))
        val_idx = np.array([], dtype=int)

    scaler = MinMaxScaler()
    scaled_train = np.zeros_like(train_split.features, dtype=np.float32)
    scaled_train[train_idx] = scaler.fit_transform(train_split.features[train_idx])
    if len(val_idx):
        scaled_train[val_idx] = scaler.transform(train_split.features[val_idx])
    scaled_test = scaler.transform(test_split.features).astype(np.float32)

    train_loader = make_loader(
        train_split, train_idx, scaled_train, tokenizer, args.batch_size, True, args.max_length
    )
    val_loader = (
        make_loader(train_split, val_idx, scaled_train, tokenizer, args.batch_size, False, args.max_length)
        if len(val_idx)
        else None
    )
    test_indices = np.arange(len(test_split.labels))
    test_loader = make_loader(
        test_split, test_indices, scaled_test, tokenizer, args.batch_size, False, args.max_length
    )

    model = build_model(args, feature_dim=train_split.features.shape[1], device=device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    best_state = None
    best_val_score = -np.inf
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        if val_loader is not None:
            val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
            print(
                f"epoch={epoch:02d} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} ACC={val_metrics['ACC']:.4f} "
                f"MCC={val_metrics['MCC']:.4f} AUC={val_metrics['AUC']:.4f}"
            )
            val_score = val_metrics[args.monitor_metric]
            if val_score > best_val_score:
                best_val_score = val_score
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                patience_left = args.patience
            else:
                patience_left -= 1
                if args.patience > 0 and patience_left <= 0:
                    print(f"early stopping at epoch {epoch}")
                    break
        else:
            print(f"epoch={epoch:02d} train_loss={train_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    print("\nIndependent test:")
    print(json.dumps({"loss": test_loss, **test_metrics}, indent=2, sort_keys=True))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_models:
        torch.save(model.state_dict(), args.output_dir / "pcbert_kla_independent.pt")
        joblib.dump(scaler, args.output_dir / "feature_scaler_independent.joblib")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean PCBert-Kla replication runner")
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=PROJECT_DIR.parents[1] / "baselines" / "PCBert-Kla-original",
    )
    parser.add_argument(
        "--run",
        choices=["data-check", "cv", "independent"],
        default="data-check",
    )
    parser.add_argument("--model-name", default="Rostlab/prot_bert")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--splitter", choices=["kfold", "stratified"], default="kfold")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--monitor-metric", default="ACC")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--save-models", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_split = load_split(
        args.baseline_root / "data" / "train.csv",
        args.baseline_root / "data" / "feature_train.csv",
    )
    test_split = load_split(
        args.baseline_root / "data" / "test.csv",
        args.baseline_root / "data" / "feature_test.csv",
    )

    if args.run == "data-check":
        run_data_check(train_split, test_split)
    elif args.run == "cv":
        ensure_ml_deps()
        if args.device == "auto":
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        run_cv(args, train_split)
    elif args.run == "independent":
        ensure_ml_deps()
        if args.device == "auto":
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        run_independent(args, train_split, test_split)
    else:
        raise ValueError(args.run)


if __name__ == "__main__":
    main()
