from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List

import torch
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from payment_foundation_model.model import ModelConfig, TransactionRuleVerifier
from payment_foundation_model.synthetic_data import CLAUSE_TYPES, SyntheticRuleVerifierDataset


class RuleCharTokenizer:
    def __init__(self, texts: Iterable[str], max_length: int = 64) -> None:
        self.max_length = max_length
        vocab = {"<pad>": 0, "<unk>": 1}
        for text in texts:
            for ch in self._normalize(text):
                if ch not in vocab:
                    vocab[ch] = len(vocab)
        self.vocab = vocab

    @staticmethod
    def _normalize(text: str) -> List[str]:
        return [ch for ch in text if ch.strip()]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        chars = self._normalize(text)[: self.max_length]
        ids = [self.vocab.get(ch, 1) for ch in chars]
        padded = torch.zeros(self.max_length, dtype=torch.long)
        mask = torch.zeros(self.max_length, dtype=torch.bool)
        if ids:
            padded[: len(ids)] = torch.tensor(ids, dtype=torch.long)
            mask[: len(ids)] = True
        return padded, mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a synthetic text-rule verifier.")
    parser.add_argument("--train-size", type=int, default=3000)
    parser.add_argument("--val-size", type=int, default=600)
    parser.add_argument("--pairs-per-sample", type=int, default=6)
    parser.add_argument("--max-events", type=int, default=32)
    parser.add_argument("--max-rule-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fraud-rate", type=float, default=0.45)
    parser.add_argument("--run-name", type=str, default="rule_verifier")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--use-entity-memory", action="store_true")
    parser.add_argument("--use-relation-bias", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_datasets(args: argparse.Namespace) -> tuple[SyntheticRuleVerifierDataset, SyntheticRuleVerifierDataset]:
    train_ds = SyntheticRuleVerifierDataset(
        size=args.train_size,
        max_events=args.max_events,
        fraud_rate=args.fraud_rate,
        seed=args.seed,
        hard_mode=True,
        max_pairs_per_sample=args.pairs_per_sample,
    )
    val_ds = SyntheticRuleVerifierDataset(
        size=args.val_size,
        max_events=args.max_events,
        fraud_rate=args.fraud_rate,
        seed=args.seed + 100_003,
        hard_mode=True,
        max_pairs_per_sample=args.pairs_per_sample,
    )
    return train_ds, val_ds


def build_tokenizer(train_ds: SyntheticRuleVerifierDataset, val_ds: SyntheticRuleVerifierDataset, max_length: int) -> RuleCharTokenizer:
    texts: List[str] = []
    for row in train_ds.rows + val_ds.rows:
        texts.append(row["rule"]["raw_text"])
        texts.append(row["rule"]["canonical_text"])
        for clause in row["rule"]["clauses"]:
            texts.append(clause["text"])
            texts.append(clause["canonical_text"])
    return RuleCharTokenizer(texts, max_length=max_length)


def make_collate_fn(tokenizer: RuleCharTokenizer):
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        collated: Dict[str, Any] = {}
        tensor_keys = [key for key, value in batch[0].items() if isinstance(value, torch.Tensor)]
        for key in tensor_keys:
            collated[key] = torch.stack([row[key] for row in batch], dim=0)

        rule_ids = []
        rule_masks = []
        for row in batch:
            ids, mask = tokenizer.encode(row["verifier_rule_text"])
            rule_ids.append(ids)
            rule_masks.append(mask)
        collated["rule_input_ids"] = torch.stack(rule_ids, dim=0)
        collated["rule_attention_mask"] = torch.stack(rule_masks, dim=0)
        collated["verifier_rule_text"] = [row["verifier_rule_text"] for row in batch]
        collated["verifier_rule_id"] = [row["verifier_rule_id"] for row in batch]
        collated["verifier_pattern_subtype"] = [row["verifier_pattern_subtype"] for row in batch]
        return collated

    return collate


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def build_model(args: argparse.Namespace, tokenizer: RuleCharTokenizer, device: torch.device) -> TransactionRuleVerifier:
    tx_config = ModelConfig(
        max_events=args.max_events,
        d_model=args.d_model,
        num_layers=args.layers,
        num_heads=args.heads,
        dropout=args.dropout,
        use_entity_memory=args.use_entity_memory,
        use_relation_bias=args.use_relation_bias,
    )
    model = TransactionRuleVerifier(
        tx_config=tx_config,
        text_vocab_size=tokenizer.vocab_size,
        num_clauses=len(CLAUSE_TYPES),
        max_rule_tokens=args.max_rule_tokens,
        text_num_layers=max(1, args.layers // 2),
    )
    return model.to(device)


def compute_losses(outputs: Dict[str, torch.Tensor], batch: Dict[str, Any], pos_weight: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
    match_loss = nn.functional.binary_cross_entropy_with_logits(
        outputs["match_logit"],
        batch["verifier_overall_match"],
        pos_weight=pos_weight,
    )
    uncertainty_loss = nn.functional.binary_cross_entropy_with_logits(
        outputs["uncertainty_logit"],
        batch["verifier_uncertain"],
    )

    clause_loss_raw = nn.functional.binary_cross_entropy_with_logits(
        outputs["clause_logits"],
        batch["verifier_clause_target"],
        reduction="none",
    )
    clause_mask = batch["verifier_clause_mask"].float()
    clause_loss = (clause_loss_raw * clause_mask).sum() / clause_mask.sum().clamp(min=1.0)

    evidence_loss_raw = nn.functional.binary_cross_entropy_with_logits(
        outputs["evidence_logits"],
        batch["verifier_evidence_mask"].float(),
        reduction="none",
    )
    event_mask = batch["event_mask"].float()
    evidence_loss = (evidence_loss_raw * event_mask).sum() / event_mask.sum().clamp(min=1.0)

    total_loss = match_loss + 0.5 * clause_loss + 0.2 * evidence_loss + 0.05 * uncertainty_loss
    return {
        "loss": total_loss,
        "match_loss": match_loss,
        "clause_loss": clause_loss,
        "evidence_loss": evidence_loss,
        "uncertainty_loss": uncertainty_loss,
    }


def compute_metrics(outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, float]:
    probs = torch.sigmoid(outputs["match_logit"])
    preds = (probs >= 0.5).float()
    labels = batch["verifier_overall_match"]

    metrics: Dict[str, float] = {
        "match_accuracy": (preds == labels).float().mean().item(),
    }
    predicted_positive = preds > 0.5
    if predicted_positive.any():
        metrics["precision_at_0.5"] = labels[predicted_positive].mean().item()
    else:
        metrics["precision_at_0.5"] = 0.0

    positive_mask = labels > 0.5
    if positive_mask.any():
        metrics["positive_recall_at_0.5"] = preds[positive_mask].mean().item()
    else:
        metrics["positive_recall_at_0.5"] = 0.0

    hard_mask = batch["verifier_difficulty_id"].eq(1)
    if hard_mask.any():
        metrics["hard_negative_accuracy"] = (preds[hard_mask] == labels[hard_mask]).float().mean().item()
        metrics["hard_negative_mean_score"] = probs[hard_mask].mean().item()
    else:
        metrics["hard_negative_accuracy"] = 0.0
        metrics["hard_negative_mean_score"] = 0.0

    clause_pred = (torch.sigmoid(outputs["clause_logits"]) >= 0.5).float()
    clause_target = batch["verifier_clause_target"]
    clause_mask = batch["verifier_clause_mask"]
    if clause_mask.any():
        metrics["clause_accuracy"] = (clause_pred[clause_mask] == clause_target[clause_mask]).float().mean().item()
    else:
        metrics["clause_accuracy"] = 0.0

    evidence_logits = outputs["evidence_logits"].masked_fill(~batch["event_mask"], -1e4)
    top_k = min(3, evidence_logits.shape[1])
    top_idx = torch.topk(evidence_logits, k=top_k, dim=1).indices
    evidence_hit = []
    for idx in range(evidence_logits.shape[0]):
        target_mask = batch["verifier_evidence_mask"][idx]
        if target_mask.any():
            hit = target_mask[top_idx[idx]].any().float().item()
            evidence_hit.append(hit)
    metrics["evidence_hit_at_3"] = sum(evidence_hit) / max(len(evidence_hit), 1)
    return metrics


def evaluate(
    model: TransactionRuleVerifier,
    loader: DataLoader,
    device: torch.device,
    pos_weight: torch.Tensor | None = None,
) -> Dict[str, float]:
    model.eval()
    aggregate: Dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = model(batch)
            losses = compute_losses(outputs, batch, pos_weight=pos_weight)
            metrics = compute_metrics(outputs, batch)
            row = {
                "loss": losses["loss"].item(),
                "match_loss": losses["match_loss"].item(),
                "clause_loss": losses["clause_loss"].item(),
                "evidence_loss": losses["evidence_loss"].item(),
                **metrics,
            }
            for key, value in row.items():
                aggregate[key] = aggregate.get(key, 0.0) + value
            count += 1
    return {key: value / max(count, 1) for key, value in aggregate.items()}


def run_experiment(args: argparse.Namespace) -> Dict[str, float]:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds = build_datasets(args)
    tokenizer = build_tokenizer(train_ds, val_ds, max_length=args.max_rule_tokens)
    collate_fn = make_collate_fn(tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    model = build_model(args, tokenizer, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_positive = sum(int(row["labels"]["overall_match"]) for row in train_ds.rows)
    train_negative = max(len(train_ds.rows) - train_positive, 1)
    if train_positive > 0:
        pos_weight = torch.tensor(float(train_negative) / float(train_positive), device=device)
    else:
        pos_weight = None

    best_metrics: Dict[str, float] = {}
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            losses = compute_losses(outputs, batch, pos_weight=pos_weight)
            losses["loss"].backward()
            optimizer.step()

        val_metrics = evaluate(model, val_loader, device, pos_weight=pos_weight)
        print(
            f"epoch={epoch} val_loss={val_metrics['loss']:.4f} match_acc={val_metrics['match_accuracy']:.4f} "
            f"precision={val_metrics['precision_at_0.5']:.4f} recall={val_metrics['positive_recall_at_0.5']:.4f} "
            f"hard_acc={val_metrics['hard_negative_accuracy']:.4f} clause_acc={val_metrics['clause_accuracy']:.4f} "
            f"evidence_hit@3={val_metrics['evidence_hit_at_3']:.4f}"
        )
        score = (
            0.40 * val_metrics["positive_recall_at_0.5"]
            + 0.25 * val_metrics["hard_negative_accuracy"]
            + 0.20 * val_metrics["clause_accuracy"]
            + 0.15 * val_metrics["precision_at_0.5"]
        )
        if score > best_score:
            best_score = score
            best_metrics = val_metrics
            args.out_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = args.out_dir / f"rule_verifier_{args.run_name}.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": {
                        "max_events": args.max_events,
                        "d_model": args.d_model,
                        "layers": args.layers,
                        "heads": args.heads,
                        "dropout": args.dropout,
                        "use_entity_memory": args.use_entity_memory,
                        "use_relation_bias": args.use_relation_bias,
                        "max_rule_tokens": args.max_rule_tokens,
                        "text_vocab_size": tokenizer.vocab_size,
                        "num_clauses": len(CLAUSE_TYPES),
                    },
                    "tokenizer_vocab": tokenizer.vocab,
                    "pos_weight": None if pos_weight is None else float(pos_weight.item()),
                    "metrics": val_metrics,
                },
                ckpt_path,
            )

    metrics_path = args.out_dir / f"rule_verifier_metrics_{args.run_name}.json"
    metrics_path.write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    return best_metrics


def main() -> None:
    args = parse_args()
    metrics = run_experiment(args)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
