"""Minimal prototype package for a transaction foundation model."""

from .synthetic_data import SyntheticRiskDataset, SyntheticRuleVerifierDataset

__all__ = [
    "SyntheticRiskDataset",
    "SyntheticRuleVerifierDataset",
]
