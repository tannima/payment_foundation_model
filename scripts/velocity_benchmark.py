from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from payment_foundation_model.model import ModelConfig, TransactionTransformer
from payment_foundation_model.synthetic_data import MAX_RESULT_TOKENS, RESULT_TOKENS, SyntheticRiskDataset


TARGET_SPECS = {
    "tx_count_1d": {
        "pred_key": "tx_count_pred",
        "target_key": "tx_count_1d_log",
        "kind": "count",
        "buckets": [0.5, 1.5, 2.5, 4.5],
    },
    "distinct_card_count_7d": {
        "pred_key": "distinct_card_pred",
        "target_key": "distinct_card_count_7d_log",
        "kind": "count",
        "buckets": [0.5, 1.5, 2.5, 4.5],
    },
    "same_device_distinct_card_7d": {
        "pred_key": "same_device_distinct_card_pred",
        "target_key": "same_device_distinct_card_7d_log",
        "kind": "count",
        "buckets": [0.5, 1.5, 2.5, 4.5],
    },
    "amount_sum_7d": {
        "pred_key": "amount_sum_pred",
        "target_key": "amount_sum_7d_log",
        "kind": "amount",
        "buckets": [50.0, 150.0, 400.0, 800.0, 1500.0, 3000.0],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run velocity benchmark on a trained synthetic checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-size", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed-offset", type=int, default=200000)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def build_model_from_checkpoint(ckpt: Dict[str, Any], device: torch.device) -> TransactionTransformer:
    cfg = ckpt["config"]
    model = TransactionTransformer(
        ModelConfig(
            max_events=cfg["max_events"],
            max_result_tokens=MAX_RESULT_TOKENS,
            result_vocab_size=len(RESULT_TOKENS),
            d_model=cfg["d_model"],
            num_layers=cfg["layers"],
            num_heads=cfg["heads"],
            dropout=cfg["dropout"],
            use_entity_memory=cfg.get("use_entity_memory", False),
            use_relation_bias=cfg.get("use_relation_bias", False),
        )
    )
    model.load_state_dict(ckpt["model_state"])
    return model.to(device)


def inverse_log1p(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x).clamp(min=0.0)


def bucketize(values: torch.Tensor, boundaries: list[float]) -> torch.Tensor:
    bucket = torch.zeros_like(values, dtype=torch.long)
    for idx, boundary in enumerate(boundaries):
        bucket = bucket + (values > boundary).long()
    return bucket


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.std(unbiased=False) * y.std(unbiased=False)
    if denom.item() == 0.0:
        return 0.0
    return float((x * y).mean() / denom)


def rankdata(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(order.numel(), dtype=torch.float32, device=x.device)
    return ranks


def spearman_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    return pearson_corr(rankdata(x), rankdata(y))


def pairwise_ranking_accuracy(pred: torch.Tensor, target: torch.Tensor, max_pairs: int = 20000) -> float:
    n = pred.numel()
    if n < 2:
        return 0.0
    device = pred.device
    pair_count = min(max_pairs, n * (n - 1) // 2)
    idx_i = torch.randint(0, n, (pair_count,), device=device)
    idx_j = torch.randint(0, n, (pair_count,), device=device)
    valid = idx_i != idx_j
    idx_i = idx_i[valid]
    idx_j = idx_j[valid]
    if idx_i.numel() == 0:
        return 0.0
    pred_diff = pred[idx_i] - pred[idx_j]
    target_diff = target[idx_i] - target[idx_j]
    non_tie = target_diff != 0
    if non_tie.sum().item() == 0:
        return 0.0
    pred_sign = torch.sign(pred_diff[non_tie])
    target_sign = torch.sign(target_diff[non_tie])
    return float((pred_sign == target_sign).float().mean())


def collect_predictions(
    model: TransactionTransformer,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    storage: Dict[str, list[torch.Tensor]] = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch)
            for name, spec in TARGET_SPECS.items():
                storage.setdefault(f"{name}_pred", []).append(outputs[spec["pred_key"]].cpu())
                storage.setdefault(f"{name}_target", []).append(batch[spec["target_key"]].cpu())

    return {key: torch.cat(value) for key, value in storage.items()}


def evaluate_target(name: str, pred_log: torch.Tensor, target_log: torch.Tensor, spec: Dict[str, Any]) -> Dict[str, float]:
    mse_log = float(torch.mean((pred_log - target_log) ** 2))
    mae_log = float(torch.mean((pred_log - target_log).abs()))
    pred_value = inverse_log1p(pred_log)
    target_value = inverse_log1p(target_log)

    metrics = {
        "mse_log": mse_log,
        "mae_log": mae_log,
        "pearson": pearson_corr(pred_value, target_value),
        "spearman": spearman_corr(pred_value, target_value),
        "pairwise_ranking_acc": pairwise_ranking_accuracy(pred_value, target_value),
    }

    pred_bucket = bucketize(pred_value, spec["buckets"])
    target_bucket = bucketize(target_value, spec["buckets"])
    metrics["bucket_accuracy"] = float((pred_bucket == target_bucket).float().mean())

    if spec["kind"] == "count":
        pred_round = torch.round(pred_value)
        target_round = torch.round(target_value)
        metrics["exact_match_accuracy"] = float((pred_round == target_round).float().mean())
        metrics["off_by_one_accuracy"] = float(((pred_round - target_round).abs() <= 1).float().mean())
    else:
        rel_err = (pred_value - target_value).abs() / torch.clamp(target_value, min=1.0)
        metrics["within_10pct_accuracy"] = float((rel_err <= 0.10).float().mean())
        metrics["within_20pct_accuracy"] = float((rel_err <= 0.20).float().mean())

    return metrics


def run_velocity_benchmark(
    checkpoint: Path,
    test_size: int = 1600,
    seed_offset: int = 200000,
    out: Path | None = None,
) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_checkpoint(ckpt, device)

    dataset = SyntheticRiskDataset(
        size=test_size,
        max_events=cfg["max_events"],
        seed=cfg["seed"] + seed_offset,
        hard_mode=cfg.get("hard_synth", False),
    )
    loader = DataLoader(dataset, batch_size=cfg.get("batch_size", 256), shuffle=False)
    tensors = collect_predictions(model, loader, device)

    results: Dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "config": {
            "use_entity_memory": cfg.get("use_entity_memory", False),
            "use_relation_bias": cfg.get("use_relation_bias", False),
            "hard_synth": cfg.get("hard_synth", False),
            "test_size": test_size,
        },
        "targets": {},
    }
    for name, spec in TARGET_SPECS.items():
        results["targets"][name] = evaluate_target(
            name=name,
            pred_log=tensors[f"{name}_pred"],
            target_log=tensors[f"{name}_target"],
            spec=spec,
        )

    if out is None:
        out = checkpoint.with_name(checkpoint.stem + "_velocity_benchmark.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return results


def main() -> None:
    args = parse_args()
    results = run_velocity_benchmark(
        checkpoint=args.checkpoint,
        test_size=args.test_size,
        seed_offset=args.seed_offset,
        out=args.out,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
