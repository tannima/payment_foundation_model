from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from payment_foundation_model.synthetic_data import SyntheticRuleVerifierDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export synthetic text-rule verifier samples.")
    parser.add_argument("--size", type=int, default=200, help="Number of base transaction samples to generate.")
    parser.add_argument("--pairs-per-sample", type=int, default=6, help="Maximum verifier pairs to emit per base sample.")
    parser.add_argument("--max-events", type=int, default=32, help="Maximum number of events to keep in transaction context.")
    parser.add_argument("--fraud-rate", type=float, default=0.45, help="Suspicious sample rate for base synthetic samples.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts") / "synthetic_rule_verifier.jsonl",
        help="Output JSONL path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = SyntheticRuleVerifierDataset(
        size=args.size,
        max_events=args.max_events,
        fraud_rate=args.fraud_rate,
        seed=args.seed,
        hard_mode=True,
        max_pairs_per_sample=args.pairs_per_sample,
    )
    dataset.export_jsonl(args.output)

    difficulty_counter = Counter(row["meta"]["difficulty"] for row in dataset.rows)
    match_counter = Counter(int(row["labels"]["overall_match"]) for row in dataset.rows)
    rule_counter = Counter(row["rule"]["rule_id"] for row in dataset.rows)

    print(f"exported_pairs={len(dataset.rows)}")
    print(f"output={args.output}")
    print(f"match_0={match_counter.get(0, 0)} match_1={match_counter.get(1, 0)}")
    print("difficulty_counts:")
    for key in sorted(difficulty_counter):
        print(f"  {key}={difficulty_counter[key]}")
    print("top_rules:")
    for rule_id, count in rule_counter.most_common(8):
        print(f"  {rule_id}={count}")


if __name__ == "__main__":
    main()
