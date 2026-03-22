from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Dict, List

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from train_synthetic import build_model, run_experiment
from velocity_benchmark import run_velocity_benchmark


def count_parameters(args: SimpleNamespace) -> int:
    device = torch.device("cpu")
    model = build_model(args, device)
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def velocity_score(benchmark: Dict[str, Any]) -> Dict[str, float]:
    targets = benchmark["targets"]
    count_exact_mean = (
        targets["tx_count_1d"]["exact_match_accuracy"]
        + targets["distinct_card_count_7d"]["exact_match_accuracy"]
        + targets["same_device_distinct_card_7d"]["exact_match_accuracy"]
    ) / 3.0
    overall_score = (
        count_exact_mean
        + targets["amount_sum_7d"]["bucket_accuracy"]
        + targets["amount_sum_7d"]["pairwise_ranking_acc"]
    ) / 3.0
    return {
        "count_exact_mean": count_exact_mean,
        "amount_bucket_accuracy": targets["amount_sum_7d"]["bucket_accuracy"],
        "amount_pairwise_ranking_acc": targets["amount_sum_7d"]["pairwise_ranking_acc"],
        "overall_velocity_score": overall_score,
    }


def load_json(path: Path) -> Dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def save_summary(rows: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)


def main() -> None:
    out_dir = ROOT / "artifacts" / "scaling"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "scaling_summary.json"

    base = {
        "max_events": 32,
        "batch_size": 200,
        "epochs": 2,
        "lr": 3e-4,
        "heads": 4,
        "dropout": 0.1,
        "seed": 31,
        "out_dir": out_dir,
        "export_jsonl": False,
        "hard_synth": True,
        "use_entity_memory": True,
        "use_relation_bias": True,
    }

    sample_sizes = [1000, 2000, 4000, 8000, 12000]
    model_scales = [
        {"scale_name": "xs", "d_model": 48, "layers": 2},
        {"scale_name": "s", "d_model": 64, "layers": 2},
        {"scale_name": "m", "d_model": 96, "layers": 3},
        {"scale_name": "l", "d_model": 128, "layers": 4},
        {"scale_name": "xl", "d_model": 160, "layers": 5},
    ]

    existing_rows: Dict[str, Dict[str, Any]] = {}
    if summary_path.exists():
        for row in load_json(summary_path):
            existing_rows[row["run_name"]] = row

    all_rows: List[Dict[str, Any]] = list(existing_rows.values())

    for model_scale in model_scales:
        for train_size in sample_sizes:
            val_size = max(400, train_size // 5)
            run_name = f"scale_{model_scale['scale_name']}_n{train_size}"
            checkpoint = out_dir / f"synthetic_model_{run_name}.pt"
            benchmark_path = out_dir / f"velocity_benchmark_{run_name}.json"

            if run_name in existing_rows and benchmark_path.exists() and checkpoint.exists():
                print(f"skip {run_name}: already complete")
                continue

            args = SimpleNamespace(
                **base,
                run_name=run_name,
                train_size=train_size,
                val_size=val_size,
                d_model=model_scale["d_model"],
                layers=model_scale["layers"],
            )
            param_count = count_parameters(args)
            print(
                f"running {run_name}: train_size={args.train_size} val_size={args.val_size} "
                f"d_model={args.d_model} layers={args.layers} params={param_count}"
            )

            run_experiment(args)
            benchmark = run_velocity_benchmark(
                checkpoint=checkpoint,
                test_size=1600,
                seed_offset=300000,
                out=benchmark_path,
            )
            score = velocity_score(benchmark)
            row = {
                "run_name": run_name,
                "train_size": args.train_size,
                "val_size": args.val_size,
                "d_model": args.d_model,
                "layers": args.layers,
                "parameter_count": param_count,
                **score,
                "benchmark": benchmark["targets"],
            }
            existing_rows[run_name] = row
            all_rows = sorted(existing_rows.values(), key=lambda item: (item["parameter_count"], item["train_size"]))
            save_summary(all_rows, summary_path)
            print(
                f"{run_name}: overall_velocity_score={score['overall_velocity_score']:.4f} "
                f"count_exact_mean={score['count_exact_mean']:.4f} "
                f"amount_bucket_accuracy={score['amount_bucket_accuracy']:.4f}"
            )

    print(json.dumps(all_rows, indent=2))


if __name__ == "__main__":
    main()
