from __future__ import annotations

import argparse
import csv
import gc
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
    global HybridTokenGatedPCBertKla
    global TokenGatedPCBertKla
    global StratifiedKFold
    global compute_binary_metrics
    global find_best_threshold
    global joblib
    global get_linear_schedule_with_warmup
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
    from transformers import AutoTokenizer, BertTokenizer, get_linear_schedule_with_warmup

    from pcbert_kla_clean.metrics import (
        compute_binary_metrics,
        find_best_threshold,
        summarize_metric_rows,
    )
    from pcbert_kla_clean.model import PCBertKla, HybridTokenGatedPCBertKla, TokenGatedPCBertKla


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


def infer_sequence_format(model_name: str) -> str:
    if model_name == "Rostlab/prot_bert":
        return "protbert_spaced"
    return "raw"


def format_sequence(sequence: str, sequence_format: str) -> str:
    if sequence_format == "raw":
        return sequence
    if sequence_format == "protbert_spaced":
        return " ".join(sequence)
    raise ValueError(f"Unsupported sequence format: {sequence_format}")


def format_protbert_sequence(sequence: str) -> str:
    return " ".join(sequence)


def make_collate_fn(tokenizer, max_length: int, sequence_format: str):
    def collate(batch):
        sequences, features, labels = zip(*batch)
        encoded = tokenizer(
            [format_sequence(sequence, sequence_format) for sequence in sequences],
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
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler=None,
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
        if scheduler is not None:
            scheduler.step()
        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses))


def predict(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
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

    return float(np.mean(losses)), np.asarray(labels_all), np.asarray(scores_all)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[float, dict[str, float]]:
    loss, labels, scores = predict(model, loader, criterion, device)
    metrics = compute_binary_metrics(labels, scores, threshold=threshold)
    return loss, metrics


def make_loader(
    split: KlaSplit,
    indices: np.ndarray,
    scaled_features: np.ndarray,
    tokenizer,
    batch_size: int,
    shuffle: bool,
    max_length: int,
    sequence_format: str,
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
        collate_fn=make_collate_fn(
            tokenizer,
            max_length=max_length,
            sequence_format=sequence_format,
        ),
    )


def build_model(args: argparse.Namespace, feature_dim: int, device: torch.device) -> nn.Module:
    if args.architecture == "baseline":
        model = PCBertKla(
            model_name=args.model_name,
            feature_dim=feature_dim,
            encoder_layers=args.encoder_layers,
            cache_dir=args.cache_dir,
        )
    elif args.architecture == "token_gated":
        model = TokenGatedPCBertKla(
            model_name=args.model_name,
            feature_dim=feature_dim,
            encoder_layers=args.encoder_layers,
            fusion_dim=args.fusion_dim,
            attention_dim=args.attention_dim,
            dropout=args.arch_dropout,
            site_token_index=args.site_token_index,
            cache_dir=args.cache_dir,
        )
    elif args.architecture == "hybrid_gated":
        model = HybridTokenGatedPCBertKla(
            model_name=args.model_name,
            feature_dim=feature_dim,
            encoder_layers=args.encoder_layers,
            fusion_dim=args.fusion_dim,
            attention_dim=args.attention_dim,
            dropout=args.arch_dropout,
            site_token_index=args.site_token_index,
            cache_dir=args.cache_dir,
        )
    else:
        raise ValueError(f"Unsupported architecture: {args.architecture}")
    return model.to(device)


def build_optimizer(args: argparse.Namespace, model: nn.Module):
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.sgd_momentum,
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def build_scheduler(args: argparse.Namespace, optimizer, steps_per_epoch: int):
    if args.scheduler == "none":
        return None
    if args.scheduler == "linear":
        total_steps = max(1, steps_per_epoch * args.epochs)
        warmup_steps = int(total_steps * args.warmup_ratio)
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")


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


def save_predictions(
    path: Path,
    split: KlaSplit,
    indices: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "name",
                "position",
                "sequence",
                "true_label",
                "score",
                "threshold",
                "predicted_label",
            ],
        )
        writer.writeheader()
        for index, score in zip(indices, scores):
            record = split.records[int(index)]
            writer.writerow(
                {
                    "name": record.name,
                    "position": record.position,
                    "sequence": record.sequence,
                    "true_label": record.label,
                    "score": float(score),
                    "threshold": threshold,
                    "predicted_label": int(float(score) >= threshold),
                }
            )


def run_cv(args: argparse.Namespace, train_split: KlaSplit) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = load_tokenizer(args.model_name, args.cache_dir)
    sequence_format = args.sequence_format
    if sequence_format == "auto":
        sequence_format = infer_sequence_format(args.model_name)
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
            train_split,
            train_idx,
            scaled,
            tokenizer,
            args.batch_size,
            True,
            args.max_length,
            sequence_format,
        )
        val_loader = make_loader(
            train_split,
            val_idx,
            scaled,
            tokenizer,
            args.batch_size,
            False,
            args.max_length,
            sequence_format,
        )
        model = build_model(args, feature_dim=train_split.features.shape[1], device=device)
        criterion = nn.BCELoss()
        optimizer = build_optimizer(args, model)
        scheduler = build_scheduler(args, optimizer, steps_per_epoch=len(train_loader))

        best_metrics: dict[str, float] | None = None
        best_score = -np.inf
        patience_left = args.patience

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scheduler=scheduler,
            )
            val_loss, metrics = evaluate(
                model,
                val_loader,
                criterion,
                device,
                threshold=args.decision_threshold,
            )
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
                        args.output_dir / f"{args.architecture}_fold_{fold_id}.pt",
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


def train_and_predict_independent(
    args: argparse.Namespace,
    train_split: KlaSplit,
    test_split: KlaSplit,
    seed: int,
    tokenizer,
    device: torch.device,
    output_dir: Path | None = None,
) -> dict[str, object]:
    set_seed(seed)
    device = torch.device(args.device)
    sequence_format = args.sequence_format
    if sequence_format == "auto":
        sequence_format = infer_sequence_format(args.model_name)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.validation_fraction > 0:
        train_idx, val_idx = train_test_split(
            np.arange(len(train_split.labels)),
            test_size=args.validation_fraction,
            random_state=seed,
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
        train_split,
        train_idx,
        scaled_train,
        tokenizer,
        args.batch_size,
        True,
        args.max_length,
        sequence_format,
    )
    val_loader = (
        make_loader(
            train_split,
            val_idx,
            scaled_train,
            tokenizer,
            args.batch_size,
            False,
            args.max_length,
            sequence_format,
        )
        if len(val_idx)
        else None
    )
    test_indices = np.arange(len(test_split.labels))
    test_loader = make_loader(
        test_split,
        test_indices,
        scaled_test,
        tokenizer,
        args.batch_size,
        False,
        args.max_length,
        sequence_format,
    )

    model = build_model(args, feature_dim=train_split.features.shape[1], device=device)
    criterion = nn.BCELoss()
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer, steps_per_epoch=len(train_loader))

    best_state = None
    best_val_score = -np.inf
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scheduler=scheduler,
        )
        if val_loader is not None:
            val_loss, val_metrics = evaluate(
                model,
                val_loader,
                criterion,
                device,
                threshold=args.decision_threshold,
            )
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

    threshold = args.decision_threshold
    threshold_source = "fixed"
    validation_threshold_metrics = None

    if args.calibrate_threshold != "none":
        if val_loader is None:
            raise ValueError(
                "--calibrate-threshold requires --validation-fraction greater than 0"
            )
        val_loss, val_labels, val_scores = predict(model, val_loader, criterion, device)
        threshold, validation_threshold_metrics = find_best_threshold(
            val_labels,
            val_scores,
            metric=args.calibrate_threshold,
            min_threshold=args.threshold_min,
            max_threshold=args.threshold_max,
            steps=args.threshold_steps,
        )
        threshold_source = f"validation_{args.calibrate_threshold}"
        print("\nCalibrated threshold:")
        print(
            json.dumps(
                {
                    "threshold": threshold,
                    "source": threshold_source,
                    "validation_loss": val_loss,
                    **validation_threshold_metrics,
                },
                indent=2,
                sort_keys=True,
            )
        )
        if args.save_predictions and output_dir is not None:
            save_predictions(
                output_dir / "validation_predictions.csv",
                train_split,
                val_idx,
                val_scores,
                threshold,
            )

    test_loss, test_labels, test_scores = predict(model, test_loader, criterion, device)
    test_metrics = compute_binary_metrics(test_labels, test_scores, threshold=threshold)

    if args.save_predictions and output_dir is not None:
        save_predictions(
            output_dir / "independent_test_predictions.csv",
            test_split,
            test_indices,
            test_scores,
            threshold,
        )
    if args.save_models and output_dir is not None:
        torch.save(model.state_dict(), output_dir / f"{args.architecture}_independent.pt")
        joblib.dump(scaler, output_dir / "feature_scaler_independent.joblib")

    result = {
        "seed": seed,
        "loss": test_loss,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "metrics": test_metrics,
        "test_labels": test_labels,
        "test_scores": test_scores,
        "test_indices": test_indices,
        "validation_threshold_metrics": validation_threshold_metrics,
    }

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return result


def run_independent(args: argparse.Namespace, train_split: KlaSplit, test_split: KlaSplit) -> None:
    device = torch.device(args.device)
    tokenizer = load_tokenizer(args.model_name, args.cache_dir)
    result = train_and_predict_independent(
        args,
        train_split,
        test_split,
        seed=args.seed,
        tokenizer=tokenizer,
        device=device,
        output_dir=args.output_dir,
    )

    print("\nIndependent test:")
    print(
        json.dumps(
            {
                "loss": result["loss"],
                "threshold": result["threshold"],
                "threshold_source": result["threshold_source"],
                **result["metrics"],
            },
            indent=2,
            sort_keys=True,
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)


def parse_seed_list(seed_text: str) -> list[int]:
    seeds = [int(seed.strip()) for seed in seed_text.split(",") if seed.strip()]
    if not seeds:
        raise ValueError("--ensemble-seeds must contain at least one seed")
    return seeds


def save_ensemble_predictions(
    path: Path,
    split: KlaSplit,
    indices: np.ndarray,
    scores_by_seed: dict[int, np.ndarray],
    ensemble_scores: np.ndarray,
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "position",
        "sequence",
        "true_label",
        "ensemble_score",
        "threshold",
        "predicted_label",
    ]
    fieldnames.extend(f"score_seed_{seed}" for seed in scores_by_seed)

    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row_offset, index in enumerate(indices):
            record = split.records[int(index)]
            row = {
                "name": record.name,
                "position": record.position,
                "sequence": record.sequence,
                "true_label": record.label,
                "ensemble_score": float(ensemble_scores[row_offset]),
                "threshold": threshold,
                "predicted_label": int(float(ensemble_scores[row_offset]) >= threshold),
            }
            for seed, scores in scores_by_seed.items():
                row[f"score_seed_{seed}"] = float(scores[row_offset])
            writer.writerow(row)


def run_ensemble_independent(
    args: argparse.Namespace,
    train_split: KlaSplit,
    test_split: KlaSplit,
) -> None:
    seeds = parse_seed_list(args.ensemble_seeds)
    device = torch.device(args.device)
    tokenizer = load_tokenizer(args.model_name, args.cache_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    scores_by_seed: dict[int, np.ndarray] = {}
    labels = None
    test_indices = None

    for seed in seeds:
        print(f"\nEnsemble member seed={seed}")
        member_output_dir = args.output_dir / f"seed_{seed}"
        result = train_and_predict_independent(
            args,
            train_split,
            test_split,
            seed=seed,
            tokenizer=tokenizer,
            device=device,
            output_dir=member_output_dir,
        )
        results.append(result)
        scores_by_seed[seed] = result["test_scores"]
        labels = result["test_labels"]
        test_indices = result["test_indices"]
        print(
            "member:",
            json.dumps(
                {
                    "seed": seed,
                    "loss": result["loss"],
                    "threshold": result["threshold"],
                    "threshold_source": result["threshold_source"],
                    **result["metrics"],
                },
                sort_keys=True,
            ),
        )

    if labels is None or test_indices is None:
        raise RuntimeError("No ensemble member results were produced")

    ensemble_scores = np.mean(np.stack(list(scores_by_seed.values()), axis=0), axis=0)
    threshold = args.decision_threshold
    threshold_source = "fixed"
    ensemble_metrics = compute_binary_metrics(labels, ensemble_scores, threshold=threshold)
    mean_member_metrics = summarize_metric_rows([result["metrics"] for result in results])

    print("\nEnsemble independent test:")
    print(
        json.dumps(
            {
                "seeds": seeds,
                "threshold": threshold,
                "threshold_source": threshold_source,
                **ensemble_metrics,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("\nMean single-seed member metrics:")
    print(json.dumps(mean_member_metrics, indent=2, sort_keys=True))

    if args.save_predictions:
        save_ensemble_predictions(
            args.output_dir / "ensemble_independent_test_predictions.csv",
            test_split,
            test_indices,
            scores_by_seed,
            ensemble_scores,
            threshold,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean PCBert-Kla replication runner")
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=PROJECT_DIR.parents[1] / "baselines" / "PCBert-Kla-original",
    )
    parser.add_argument(
        "--run",
        choices=["data-check", "cv", "independent", "ensemble-independent"],
        default="data-check",
    )
    parser.add_argument("--model-name", default="Rostlab/prot_bert")
    parser.add_argument(
        "--sequence-format",
        choices=["auto", "protbert_spaced", "raw"],
        default="auto",
        help=(
            "How to format amino-acid windows before tokenization. Use auto for "
            "ProtBert spaced residues and raw sequences for ESM-style models."
        ),
    )
    parser.add_argument(
        "--architecture",
        choices=["baseline", "token_gated", "hybrid_gated"],
        default="baseline",
        help=(
            "Model architecture. 'baseline' preserves PCBert-Kla; 'token_gated' "
            "uses site-aware token attention and gated physicochemical fusion; "
            "'hybrid_gated' preserves CLS context and adds site-aware pooling."
        ),
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ensemble-seeds", default="42,123,2025,3407,777")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--splitter", choices=["kfold", "stratified"], default="kfold")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--sgd-momentum", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--scheduler", choices=["none", "linear"], default="none")
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--fusion-dim", type=int, default=256)
    parser.add_argument("--attention-dim", type=int, default=256)
    parser.add_argument("--arch-dropout", type=float, default=0.2)
    parser.add_argument(
        "--site-token-index",
        type=int,
        default=26,
        help="Token index of the central lysine after ProtBert special tokens are added.",
    )
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--monitor-metric", default="ACC")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument(
        "--calibrate-threshold",
        choices=["none", "ACC", "Rec", "Pre", "MCC", "F1", "SP"],
        default="none",
        help="Choose a decision threshold on the validation subset before testing.",
    )
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-steps", type=int, default=901)
    parser.add_argument(
        "--save-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save validation/test probability CSV files for independent runs.",
    )
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
    elif args.run == "ensemble-independent":
        ensure_ml_deps()
        if args.device == "auto":
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        run_ensemble_independent(args, train_split, test_split)
    else:
        raise ValueError(args.run)


if __name__ == "__main__":
    main()
