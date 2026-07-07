from __future__ import annotations

import csv
import gc
import json
from argparse import Namespace
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn

from pcbert_kla_clean.backbones import infer_sequence_format, load_tokenizer
from pcbert_kla_clean.data import (
    KlaSplit,
    describe_split,
    train_test_overlap_report,
)
from pcbert_kla_clean.datasets import make_loader
from pcbert_kla_clean.metrics import (
    compute_binary_metrics,
    find_best_threshold,
    summarize_metric_rows,
)
from pcbert_kla_clean.training import (
    build_model,
    build_optimizer,
    build_scheduler,
    evaluate,
    predict,
    train_one_epoch,
)
from pcbert_kla_clean.utils import set_seed


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


def run_cv(args: Namespace, train_split: KlaSplit) -> None:
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
    args: Namespace,
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
                best_state = {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                }
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


def run_independent(args: Namespace, train_split: KlaSplit, test_split: KlaSplit) -> None:
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
    args: Namespace,
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
