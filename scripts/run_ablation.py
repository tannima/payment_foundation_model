from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from train_synthetic import run_experiment


def main() -> None:
    out_dir = ROOT / "artifacts" / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)

    common = {
        "train_size": 2400,
        "val_size": 480,
        "max_events": 32,
        "batch_size": 160,
        "epochs": 2,
        "lr": 3e-4,
        "d_model": 96,
        "layers": 3,
        "heads": 4,
        "dropout": 0.1,
        "seed": 11,
        "out_dir": out_dir,
        "export_jsonl": False,
        "hard_synth": True,
    }

    configs = [
        {"run_name": "base", "use_entity_memory": False, "use_relation_bias": False},
        {"run_name": "memory_only", "use_entity_memory": True, "use_relation_bias": False},
        {"run_name": "bias_only", "use_entity_memory": False, "use_relation_bias": True},
        {"run_name": "memory_plus_bias", "use_entity_memory": True, "use_relation_bias": True},
    ]

    results = []
    for cfg in configs:
        args = SimpleNamespace(**common, **cfg)
        results.append(run_experiment(args))

    summary_path = out_dir / "ablation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    for row in results:
        metrics = row["best_metrics"]
        print(
            f"{row['run_name']}: "
            f"fraud_r@10={metrics['fraud_recall_at_top10pct']:.4f} "
            f"hard_fp={metrics['hard_negative_fp_rate']:.4f} "
            f"switch_r@0.5={metrics['switch_attack_recall_at_0.5']:.4f} "
            f"result_seq={metrics['result_sequence_accuracy']:.4f} "
            f"same_dev_mse={metrics['mse_same_device_distinct_card']:.4f}"
        )


if __name__ == "__main__":
    main()
