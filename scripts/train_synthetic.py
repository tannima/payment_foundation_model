from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from payment_foundation_model.model import ModelConfig, TransactionTransformer
from payment_foundation_model.synthetic_data import (
    MAX_RESULT_TOKENS,
    RESULT_BOS_ID,
    RESULT_EOS_ID,
    RESULT_PAD_ID,
    RESULT_TOKENS,
    SyntheticRiskDataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a synthetic transaction foundation model.")
    parser.add_argument("--train-size", type=int, default=6000)
    parser.add_argument("--val-size", type=int, default=1200)
    parser.add_argument("--max-events", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--export-jsonl", action="store_true")
    parser.add_argument("--hard-synth", action="store_true")
    parser.add_argument("--use-entity-memory", action="store_true")
    parser.add_argument("--use-relation-bias", action="store_true")
    parser.add_argument("--run-name", type=str, default="default")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def compute_fraud_metrics(labels: torch.Tensor, probs: torch.Tensor) -> dict[str, float]:
    preds = (probs >= 0.5).float()
    accuracy = (preds == labels).float().mean().item()
    total_positive = labels.sum().item()
    top_k = max(1, int(0.1 * labels.numel()))
    top_idx = torch.topk(probs, k=top_k).indices
    top_labels = labels[top_idx]
    precision_at_10 = top_labels.mean().item()
    recall_at_10 = top_labels.sum().item() / max(total_positive, 1.0)
    return {
        "fraud_accuracy": accuracy,
        "fraud_precision_at_top10pct": precision_at_10,
        "fraud_recall_at_top10pct": recall_at_10,
    }


def compute_slice_metrics(
    labels: torch.Tensor,
    probs: torch.Tensor,
    hard_negative: torch.Tensor,
    switch_attack: torch.Tensor,
    shared_device_benign: torch.Tensor,
    travel_burst_benign: torch.Tensor,
) -> dict[str, float]:
    metrics: Dict[str, float] = {}
    hard_mask = hard_negative > 0.5
    if hard_mask.any():
        metrics["hard_negative_fp_rate"] = (probs[hard_mask] >= 0.5).float().mean().item()
        metrics["hard_negative_mean_score"] = probs[hard_mask].mean().item()
    else:
        metrics["hard_negative_fp_rate"] = 0.0
        metrics["hard_negative_mean_score"] = 0.0

    switch_mask = switch_attack > 0.5
    if switch_mask.any():
        metrics["switch_attack_mean_score"] = probs[switch_mask].mean().item()
        metrics["switch_attack_recall_at_0.5"] = (probs[switch_mask] >= 0.5).float().mean().item()
    else:
        metrics["switch_attack_mean_score"] = 0.0
        metrics["switch_attack_recall_at_0.5"] = 0.0

    shared_mask = shared_device_benign > 0.5
    if shared_mask.any():
        metrics["shared_device_benign_fp_rate"] = (probs[shared_mask] >= 0.5).float().mean().item()
        metrics["shared_device_benign_mean_score"] = probs[shared_mask].mean().item()
    else:
        metrics["shared_device_benign_fp_rate"] = 0.0
        metrics["shared_device_benign_mean_score"] = 0.0

    travel_mask = travel_burst_benign > 0.5
    if travel_mask.any():
        metrics["travel_burst_benign_fp_rate"] = (probs[travel_mask] >= 0.5).float().mean().item()
        metrics["travel_burst_benign_mean_score"] = probs[travel_mask].mean().item()
    else:
        metrics["travel_burst_benign_fp_rate"] = 0.0
        metrics["travel_burst_benign_mean_score"] = 0.0
    return metrics


def compute_result_metrics(
    model: TransactionTransformer,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    token_correct = 0.0
    token_total = 0.0
    seq_correct = 0.0
    seq_total = 0.0

    for batch in loader:
        batch = move_batch(batch, device)
        generated = model.generate_result_chain(
            batch,
            bos_token_id=RESULT_BOS_ID,
            eos_token_id=RESULT_EOS_ID,
            max_steps=MAX_RESULT_TOKENS,
        )
        target = torch.full(
            (batch["result_target_ids"].shape[0], MAX_RESULT_TOKENS + 1),
            RESULT_PAD_ID,
            dtype=torch.long,
            device=device,
        )
        target[:, 0] = RESULT_BOS_ID
        valid_len = batch["result_mask"].sum(dim=1)
        for idx in range(target.shape[0]):
            length = int(valid_len[idx].item())
            target[idx, 1 : length + 1] = batch["result_target_ids"][idx, :length]

        compare_len = min(generated.shape[1], target.shape[1])
        match = generated[:, :compare_len] == target[:, :compare_len]
        non_pad = target[:, :compare_len] != RESULT_PAD_ID
        token_correct += (match & non_pad).sum().item()
        token_total += non_pad.sum().item()

        for idx in range(target.shape[0]):
            target_seq = target[idx][target[idx] != RESULT_PAD_ID]
            gen_seq = generated[idx][: target_seq.shape[0]]
            if gen_seq.shape[0] == target_seq.shape[0] and torch.equal(gen_seq, target_seq):
                seq_correct += 1.0
            seq_total += 1.0

    return {
        "result_token_accuracy": token_correct / max(token_total, 1.0),
        "result_sequence_accuracy": seq_correct / max(seq_total, 1.0),
    }


def build_datasets(args: argparse.Namespace) -> tuple[SyntheticRiskDataset, SyntheticRiskDataset]:
    train_ds = SyntheticRiskDataset(
        size=args.train_size,
        max_events=args.max_events,
        seed=args.seed,
        hard_mode=args.hard_synth,
    )
    val_ds = SyntheticRiskDataset(
        size=args.val_size,
        max_events=args.max_events,
        seed=args.seed + 100000,
        hard_mode=args.hard_synth,
    )
    return train_ds, val_ds


def build_model(args: argparse.Namespace, device: torch.device) -> TransactionTransformer:
    model = TransactionTransformer(
        ModelConfig(
            max_events=args.max_events,
            max_result_tokens=MAX_RESULT_TOKENS,
            result_vocab_size=len(RESULT_TOKENS),
            d_model=args.d_model,
            num_layers=args.layers,
            num_heads=args.heads,
            dropout=args.dropout,
            use_entity_memory=args.use_entity_memory,
            use_relation_bias=args.use_relation_bias,
        )
    )
    return model.to(device)


def evaluate(
    model: TransactionTransformer,
    loader: DataLoader,
    device: torch.device,
    fraud_loss_fn: nn.Module,
    result_loss_fn: nn.Module,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    all_labels = []
    all_probs = []
    all_hard = []
    all_switch = []
    all_shared = []
    all_travel = []
    mse_tx_count = 0.0
    mse_distinct = 0.0
    mse_amount = 0.0
    mse_same_device = 0.0
    result_loss_total = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = model(batch)
            fraud_loss = fraud_loss_fn(outputs["fraud_logit"], batch["fraud_label"])
            tx_loss = nn.functional.mse_loss(outputs["tx_count_pred"], batch["tx_count_1d_log"])
            distinct_loss = nn.functional.mse_loss(
                outputs["distinct_card_pred"], batch["distinct_card_count_7d_log"]
            )
            amount_loss = nn.functional.mse_loss(outputs["amount_sum_pred"], batch["amount_sum_7d_log"])
            same_device_loss = nn.functional.mse_loss(
                outputs["same_device_distinct_card_pred"], batch["same_device_distinct_card_7d_log"]
            )
            result_loss = result_loss_fn(
                outputs["result_logits"].reshape(-1, outputs["result_logits"].shape[-1]),
                batch["result_target_ids"].reshape(-1),
            )
            loss = (
                0.30 * fraud_loss
                + 0.20 * (tx_loss + distinct_loss + amount_loss + same_device_loss)
                + result_loss
            )

            total_loss += loss.item()
            total_batches += 1
            all_labels.append(batch["fraud_label"].cpu())
            all_probs.append(torch.sigmoid(outputs["fraud_logit"]).cpu())
            all_hard.append(batch["is_hard_negative"].cpu())
            all_switch.append(batch["is_switch_attack"].cpu())
            all_shared.append(batch["is_shared_device_benign"].cpu())
            all_travel.append(batch["is_travel_burst_benign"].cpu())
            mse_tx_count += tx_loss.item()
            mse_distinct += distinct_loss.item()
            mse_amount += amount_loss.item()
            mse_same_device += same_device_loss.item()
            result_loss_total += result_loss.item()

    labels = torch.cat(all_labels)
    probs = torch.cat(all_probs)
    hard = torch.cat(all_hard)
    switch = torch.cat(all_switch)
    shared = torch.cat(all_shared)
    travel = torch.cat(all_travel)

    metrics = compute_fraud_metrics(labels, probs)
    metrics.update(compute_slice_metrics(labels, probs, hard, switch, shared, travel))
    metrics.update(compute_result_metrics(model, loader, device))
    metrics.update(
        {
            "loss": total_loss / max(total_batches, 1),
            "mse_tx_count": mse_tx_count / max(total_batches, 1),
            "mse_distinct_card": mse_distinct / max(total_batches, 1),
            "mse_amount_sum": mse_amount / max(total_batches, 1),
            "mse_same_device_distinct_card": mse_same_device / max(total_batches, 1),
            "result_ce_loss": result_loss_total / max(total_batches, 1),
        }
    )
    return metrics


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds = build_datasets(args)

    if getattr(args, "export_jsonl", False):
        train_ds.export_jsonl(args.out_dir / f"synthetic_train_{args.run_name}.jsonl")
        val_ds.export_jsonl(args.out_dir / f"synthetic_val_{args.run_name}.jsonl")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    model = build_model(args, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    fraud_loss_fn = nn.BCEWithLogitsLoss()
    result_loss_fn = nn.CrossEntropyLoss(ignore_index=RESULT_PAD_ID)

    best_score = -1.0
    best_metrics: Dict[str, Any] = {}
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = move_batch(batch, device)
            outputs = model(batch)
            fraud_loss = fraud_loss_fn(outputs["fraud_logit"], batch["fraud_label"])
            tx_loss = nn.functional.mse_loss(outputs["tx_count_pred"], batch["tx_count_1d_log"])
            distinct_loss = nn.functional.mse_loss(
                outputs["distinct_card_pred"], batch["distinct_card_count_7d_log"]
            )
            amount_loss = nn.functional.mse_loss(outputs["amount_sum_pred"], batch["amount_sum_7d_log"])
            same_device_loss = nn.functional.mse_loss(
                outputs["same_device_distinct_card_pred"], batch["same_device_distinct_card_7d_log"]
            )
            result_loss = result_loss_fn(
                outputs["result_logits"].reshape(-1, outputs["result_logits"].shape[-1]),
                batch["result_target_ids"].reshape(-1),
            )
            loss = (
                0.30 * fraud_loss
                + 0.20 * (tx_loss + distinct_loss + amount_loss + same_device_loss)
                + result_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        metrics = evaluate(model, val_loader, device, fraud_loss_fn, result_loss_fn)
        avg_train_loss = running_loss / max(len(train_loader), 1)
        print(
            f"run={args.run_name} epoch={epoch} train_loss={avg_train_loss:.4f} "
            f"val_loss={metrics['loss']:.4f} fraud_r@10={metrics['fraud_recall_at_top10pct']:.4f} "
            f"hard_fp={metrics['hard_negative_fp_rate']:.4f} switch_r@0.5={metrics['switch_attack_recall_at_0.5']:.4f} "
            f"result_seq={metrics['result_sequence_accuracy']:.4f} same_dev_mse={metrics['mse_same_device_distinct_card']:.4f}"
        )

        score = (
            metrics["fraud_recall_at_top10pct"]
            + metrics["result_sequence_accuracy"]
            - metrics["hard_negative_fp_rate"]
        )
        if score > best_score:
            best_score = score
            best_metrics = {"epoch": epoch, **metrics}
            ckpt_path = args.out_dir / f"synthetic_model_{args.run_name}.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": vars(args),
                    "metrics": best_metrics,
                },
                ckpt_path,
            )

    results = {
        "run_name": args.run_name,
        "use_entity_memory": bool(args.use_entity_memory),
        "use_relation_bias": bool(args.use_relation_bias),
        "hard_synth": bool(args.hard_synth),
        "best_selection_score": best_score,
        "best_metrics": best_metrics,
    }
    results_path = args.out_dir / f"metrics_{args.run_name}.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return results


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
