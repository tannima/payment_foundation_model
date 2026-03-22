# Transaction Foundation Model Sample Protocol

## Goal

This protocol defines the first training sample format for the foundation model
prototype. It is intentionally smaller than the final production design, but it
keeps the main ideas:

- raw event sequence instead of precomputed velocity features
- local entity re-indexing inside each sample
- relative-time encoding to the anchor transaction
- multitask supervision on fraud label, atomic risk quantities, and an
  autoregressive result chain

## Sample Unit

One training sample corresponds to one anchor transaction and all retrieved
historical events visible before that anchor time.

Each sample contains:

- `domain_id`: business domain identifier
- `events`: ordered sequence of history events plus the anchor event
- `event_mask`: valid event positions
- `targets`: downstream, atomic, and result-chain supervision labels

## Event Ordering

Events are sorted by event time in ascending order.

- history events come first
- anchor event is the last valid event
- padded positions are appended at the tail

## Event Features

Each event is represented by typed fields.

### Numeric features

- `amount_log`: `log1p(event_amount)`
- `delta_hours_log`: `log1p(hours_between_anchor_and_event)`

### Local entity ids

Local ids are assigned inside one sample after retrieval and deduplication.
Missing values are encoded as `0`.

- `user_local_id`
- `card_local_id`
- `device_local_id`
- `email_local_id`
- `address_local_id`
- `ip_local_id`

### Low-cardinality categorical features

- `domain_id`
- `card_country_id`
- `issuer_id`
- `item_category_id`

### Relation-to-anchor flags

- `is_anchor`
- `same_user_as_anchor`
- `same_card_as_anchor`
- `same_device_as_anchor`
- `same_email_as_anchor`
- `same_address_as_anchor`
- `same_ip_as_anchor`

### Availability flags

- `has_device`
- `has_email`
- `has_address`
- `has_ip`

## Targets

The first prototype uses one downstream label, three atomic targets, and one
autoregressive result chain target.

### Fraud label

- `fraud_label`: `0` or `1`

### Atomic targets

- `tx_count_1d_log`: `log1p(number of history events in the last 1 day)`
- `distinct_card_count_7d_log`: `log1p(number of distinct cards in the last 7 days)`
- `amount_sum_7d_log`: `log1p(sum of historical event amounts in the last 7 days)`

These targets are derived only from the history before the anchor event.

### Result chain target

The result chain is modeled as a causal sequence rather than parallel heads.

Example sequence:

- `<bos> risk_allow 3ds_yes bank_approve pay_success fraud_yes cb_no <eos>`

Shorter sequences are valid when the payment flow terminates earlier:

- `<bos> risk_reject <eos>`
- `<bos> risk_allow 3ds_no bank_decline <eos>`

Stored tensors:

- `result_input_ids`
- `result_target_ids`
- `result_mask`

## Tensor Shapes

For batch size `B` and max event length `T`:

- categorical tensors: `[B, T]`
- numeric tensors: `[B, T]`
- boolean flags: `[B, T]`
- event mask: `[B, T]`
- labels: `[B]`
- result-chain tensors: `[B, R]` where `R` is max decoder length

## First Prototype Assumptions

- only one anchor transaction per sample
- no explicit entity memory tokens yet
- no online retrieval service yet; retrieval is simulated in synthetic data
- no text encoder yet; `item_name` is mapped to a compact item category
- the payment decision chain is trained with teacher forcing autoregression

## Upgrade Path

This protocol is designed to extend cleanly toward the full foundation model:

- add retrieved entity memory tokens
- add more atomic targets from velocity templates
- replace item category with text encoder output
- support multiple business domains with different missing-field patterns
- replace synthetic retrieval with the online context engine
