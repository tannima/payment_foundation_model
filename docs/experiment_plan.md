# Experiment Plan For The First Prototype

## Objective

Validate that raw event sequences with local entity ids can learn core risk
patterns that previously depended on hand-crafted velocity features.

## Main Questions

1. Can the model predict synthetic fraud labels from raw event history?
2. Can the same backbone learn atomic quantities such as recent transaction
   count, distinct card count, and recent amount sum?
3. Can the same model learn the payment decision chain with autoregressive
   decoding?
4. Does local entity re-indexing materially help the model learn switch and
   burst patterns?

## Baselines

### Baseline A

Small MLP on hand-crafted aggregate features:

- transaction count in 1 day
- distinct card count in 7 days
- amount sum in 7 days
- same card as previous transaction

### Baseline B

Transformer on event sequence without local entity ids.

### Proposed

Transformer on event sequence with local entity ids and relation-to-anchor
flags, plus an autoregressive result-chain decoder.

## Datasets

### Train

- synthetic anchors: 20,000 to 50,000

### Validation

- synthetic anchors: 4,000 to 10,000

### Domain setup

- `AE`: full fields available
- `Antom`: some device fields missing
- `AlipayHK`: some billing fields missing

## Metrics

### Classification

- loss
- average precision proxy: `precision_at_top_10_percent`
- `recall_at_top_10_percent`
- accuracy at threshold `0.5`

### Atomic tasks

- MSE for each target
- Pearson correlation between predicted and target logs

### Result chain

- token accuracy
- exact sequence accuracy
- cross-entropy

## Ablations

1. Remove local entity ids
2. Remove relation-to-anchor flags
3. Replace autoregressive result-chain decoding with parallel result heads
4. Train only fraud classification without atomic losses
5. Train only on one domain and evaluate on all domains

## Success Criteria

The prototype is useful if:

- fraud training converges stably
- top-ranked recall is clearly above random
- atomic targets are predicted with meaningful correlation
- removing local entity ids causes a measurable drop

## Failure Criteria

The current design should be reconsidered if:

- fraud recall stays near random after convergence
- atomic target errors remain high and uncorrelated
- the model only succeeds when explicit atomic targets are directly fed in

## Next Steps After Success

1. Add entity memory tokens
2. Add typed attention bias for same-entity relations
3. Add a retrieval interface instead of synthetic history assembly
4. Replace synthetic labels with real business samples
