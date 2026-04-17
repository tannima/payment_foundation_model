from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    max_events: int = 32
    max_result_tokens: int = 8
    result_vocab_size: int = 32
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.1
    local_entity_vocab: int = 32
    domain_vocab: int = 8
    country_vocab: int = 8
    issuer_vocab: int = 16
    item_vocab: int = 16
    use_entity_memory: bool = False
    use_relation_bias: bool = False


class RelationAwareEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        x_norm = self.norm1(x)
        attn_mask = None
        if attn_bias is not None:
            batch_size, seq_len, _ = attn_bias.shape
            expanded = attn_bias.unsqueeze(1).expand(batch_size, self.num_heads, seq_len, seq_len)
            attn_mask = expanded.reshape(batch_size * self.num_heads, seq_len, seq_len)
        attn_out, _ = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = residual + self.dropout1(attn_out)
        x = x.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        residual = x
        x_norm = self.norm2(x)
        x = residual + self.dropout2(self.ffn(x_norm))
        x = x.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        return x


class RelationAwareEncoder(nn.Module):
    def __init__(self, d_model: int, num_layers: int, num_heads: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                RelationAwareEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor, attn_bias: torch.Tensor | None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, valid_mask=valid_mask, attn_bias=attn_bias)
        return self.norm(x)


class TransactionTransformer(nn.Module):
    entity_fields = [
        "user_local_id",
        "card_local_id",
        "device_local_id",
        "email_local_id",
        "address_local_id",
        "ip_local_id",
    ]

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        ent_dim = 16
        cat_dim = 12
        self.user_emb = nn.Embedding(config.local_entity_vocab, ent_dim)
        self.card_emb = nn.Embedding(config.local_entity_vocab, ent_dim)
        self.device_emb = nn.Embedding(config.local_entity_vocab, ent_dim)
        self.email_emb = nn.Embedding(config.local_entity_vocab, ent_dim)
        self.address_emb = nn.Embedding(config.local_entity_vocab, ent_dim)
        self.ip_emb = nn.Embedding(config.local_entity_vocab, ent_dim)
        self.domain_emb = nn.Embedding(config.domain_vocab, cat_dim)
        self.country_emb = nn.Embedding(config.country_vocab, cat_dim)
        self.issuer_emb = nn.Embedding(config.issuer_vocab, cat_dim)
        self.item_emb = nn.Embedding(config.item_vocab, cat_dim)

        numeric_dim = 2
        relation_dim = 11
        input_dim = ent_dim * 6 + cat_dim * 4 + numeric_dim + relation_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.event_position_emb = nn.Embedding(config.max_events + 1, config.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.cls_domain_proj = nn.Linear(cat_dim, config.d_model)

        self.entity_type_emb = nn.Embedding(len(self.entity_fields), config.d_model)
        self.entity_memory_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

        self.encoder = RelationAwareEncoder(
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
        )

        # same user, same card, same device, same email, same address, same ip, recent time
        self.relation_weights = nn.Parameter(torch.zeros(7))
        self.memory_relation_weight = nn.Parameter(torch.tensor(0.3))

        self.result_token_emb = nn.Embedding(config.result_vocab_size, config.d_model)
        self.result_position_emb = nn.Embedding(config.max_result_tokens + 1, config.d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_model * config.mlp_ratio,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.result_decoder = nn.TransformerDecoder(decoder_layer, num_layers=max(2, config.num_layers // 2))
        self.result_norm = nn.LayerNorm(config.d_model)
        self.result_head = nn.Linear(config.d_model, config.result_vocab_size)

        self.fraud_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 1),
        )
        self.tx_count_head = nn.Linear(config.d_model, 1)
        self.distinct_card_head = nn.Linear(config.d_model, 1)
        self.amount_sum_head = nn.Linear(config.d_model, 1)
        self.same_device_distinct_card_head = nn.Linear(config.d_model, 1)

    def _build_event_tokens(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        event_mask = batch["event_mask"]
        batch_size, seq_len = event_mask.shape
        device = event_mask.device

        pieces = [
            self.user_emb(batch["user_local_id"]),
            self.card_emb(batch["card_local_id"]),
            self.device_emb(batch["device_local_id"]),
            self.email_emb(batch["email_local_id"]),
            self.address_emb(batch["address_local_id"]),
            self.ip_emb(batch["ip_local_id"]),
            self.domain_emb(batch["domain_id"]).unsqueeze(1).expand(batch_size, seq_len, -1),
            self.country_emb(batch["card_country_id"]),
            self.issuer_emb(batch["issuer_id"]),
            self.item_emb(batch["item_category_id"]),
        ]
        numeric = torch.stack([batch["amount_log"], batch["delta_hours_log"]], dim=-1)
        relations = torch.stack(
            [
                batch["is_anchor"],
                batch["same_user_as_anchor"],
                batch["same_card_as_anchor"],
                batch["same_device_as_anchor"],
                batch["same_email_as_anchor"],
                batch["same_address_as_anchor"],
                batch["same_ip_as_anchor"],
                batch["has_device"],
                batch["has_email"],
                batch["has_address"],
                batch["has_ip"],
            ],
            dim=-1,
        )
        x = torch.cat(pieces + [numeric, relations], dim=-1)
        x = self.input_proj(x)
        positions = torch.arange(1, seq_len + 1, device=device).unsqueeze(0).expand(batch_size, -1)
        return x + self.event_position_emb(positions)

    def _build_entity_memories(self, batch: dict[str, torch.Tensor], event_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.config.use_entity_memory:
            batch_size = event_tokens.shape[0]
            empty_tokens = event_tokens.new_zeros((batch_size, 0, self.config.d_model))
            empty_mask = torch.zeros((batch_size, 0), dtype=torch.bool, device=event_tokens.device)
            return empty_tokens, empty_mask

        max_local_entities = self.config.local_entity_vocab - 1
        memory_tokens = []
        memory_masks = []
        for type_idx, field_name in enumerate(self.entity_fields):
            ids = batch[field_name].clamp(min=0, max=max_local_entities)
            one_hot = F.one_hot(ids, num_classes=max_local_entities + 1).float()[:, :, 1:]
            counts = one_hot.sum(dim=1)
            agg = one_hot.transpose(1, 2) @ event_tokens
            agg = agg / counts.unsqueeze(-1).clamp(min=1.0)
            agg = agg + self.entity_type_emb.weight[type_idx].view(1, 1, -1)
            agg = self.entity_memory_proj(agg)
            memory_tokens.append(agg)
            memory_masks.append(counts > 0)
        return torch.cat(memory_tokens, dim=1), torch.cat(memory_masks, dim=1)

    def _build_relation_bias(
        self,
        batch: dict[str, torch.Tensor],
        total_valid_mask: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.config.use_relation_bias:
            return None

        event_mask = batch["event_mask"]
        batch_size, event_len = event_mask.shape
        device = event_mask.device
        cls_len = 1
        memory_len = memory_mask.shape[1]
        total_len = cls_len + event_len + memory_len
        bias = torch.zeros((batch_size, total_len, total_len), device=device)

        relation_fields = self.entity_fields
        relation_stack = []
        for field_name in relation_fields:
            field = batch[field_name]
            same = (field.unsqueeze(1) == field.unsqueeze(2)) & field.unsqueeze(1).gt(0) & field.unsqueeze(2).gt(0)
            relation_stack.append(same.float())

        time_delta = batch["delta_hours_log"].exp() - 1.0
        time_diff = (time_delta.unsqueeze(1) - time_delta.unsqueeze(2)).abs()
        recent_time = (time_diff <= 24.0).float()
        relation_stack.append(recent_time)
        stacked = torch.stack(relation_stack, dim=-1)
        event_bias = (stacked * self.relation_weights.view(1, 1, 1, -1)).sum(dim=-1)
        bias[:, cls_len : cls_len + event_len, cls_len : cls_len + event_len] = event_bias

        if self.config.use_entity_memory and memory_len > 0:
            max_local_entities = self.config.local_entity_vocab - 1
            for type_idx, field_name in enumerate(self.entity_fields):
                ids = batch[field_name]
                start = cls_len + event_len + type_idx * max_local_entities
                for local_id in range(1, max_local_entities + 1):
                    mem_index = start + (local_id - 1)
                    member = ids.eq(local_id).float()
                    bias[:, cls_len : cls_len + event_len, mem_index] += member * self.memory_relation_weight
                    bias[:, mem_index, cls_len : cls_len + event_len] += member * self.memory_relation_weight

        invalid_keys = ~total_valid_mask.unsqueeze(1)
        bias = bias.masked_fill(invalid_keys, -1e4)
        return bias

    def encode_events(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        event_tokens = self._build_event_tokens(batch)
        event_mask = batch["event_mask"]
        batch_size = event_tokens.shape[0]
        device = event_tokens.device

        memory_tokens, memory_mask = self._build_entity_memories(batch, event_tokens)
        cls = self.cls_token.expand(batch_size, -1, -1)
        cls = cls + self.cls_domain_proj(self.domain_emb(batch["domain_id"])).unsqueeze(1)

        x = torch.cat([cls, event_tokens, memory_tokens], dim=1)
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        total_valid_mask = torch.cat([cls_mask, event_mask, memory_mask], dim=1)
        attn_bias = self._build_relation_bias(batch, total_valid_mask=total_valid_mask, memory_mask=memory_mask)
        encoded = self.encoder(x, valid_mask=total_valid_mask, attn_bias=attn_bias)
        pooled = encoded[:, 0]
        return encoded, total_valid_mask, pooled

    def decode_result_chain(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        result_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = result_input_ids.shape
        device = result_input_ids.device
        positions = torch.arange(1, seq_len + 1, device=device).unsqueeze(0).expand(batch_size, -1)
        tgt = self.result_token_emb(result_input_ids) + self.result_position_emb(positions)
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=device, dtype=torch.bool),
            diagonal=1,
        )
        tgt_padding_mask = result_input_ids.eq(0)
        decoded = self.result_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=~memory_mask,
        )
        decoded = self.result_norm(decoded)
        return self.result_head(decoded)

    def generate_result_chain(
        self,
        batch: dict[str, torch.Tensor],
        bos_token_id: int,
        eos_token_id: int,
        max_steps: int,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            memory, memory_mask, _ = self.encode_events(batch)
            generated = torch.full(
                (memory.shape[0], 1),
                bos_token_id,
                dtype=torch.long,
                device=memory.device,
            )
            for _ in range(max_steps):
                logits = self.decode_result_chain(memory, memory_mask, generated)
                next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                if torch.all(next_token.squeeze(-1) == eos_token_id):
                    break
            return generated

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        memory, memory_mask, pooled = self.encode_events(batch)
        outputs = {
            "fraud_logit": self.fraud_head(pooled).squeeze(-1),
            "tx_count_pred": self.tx_count_head(pooled).squeeze(-1),
            "distinct_card_pred": self.distinct_card_head(pooled).squeeze(-1),
            "amount_sum_pred": self.amount_sum_head(pooled).squeeze(-1),
            "same_device_distinct_card_pred": self.same_device_distinct_card_head(pooled).squeeze(-1),
            "embedding": pooled,
        }
        if "result_input_ids" in batch:
            outputs["result_logits"] = self.decode_result_chain(memory, memory_mask, batch["result_input_ids"])
        return outputs


class TransactionRuleVerifier(nn.Module):
    def __init__(
        self,
        tx_config: ModelConfig,
        text_vocab_size: int,
        num_clauses: int,
        max_rule_tokens: int = 64,
        text_num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.tx_encoder = TransactionTransformer(tx_config)
        self.d_model = tx_config.d_model
        self.max_rule_tokens = max_rule_tokens

        self.rule_token_emb = nn.Embedding(text_vocab_size, self.d_model)
        self.rule_position_emb = nn.Embedding(max_rule_tokens + 1, self.d_model)
        text_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=tx_config.num_heads,
            dim_feedforward=self.d_model * tx_config.mlp_ratio,
            dropout=tx_config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.rule_encoder = nn.TransformerEncoder(text_layer, num_layers=text_num_layers)
        self.rule_norm = nn.LayerNorm(self.d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=tx_config.num_heads,
            dropout=tx_config.dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(self.d_model)

        fusion_dim = self.d_model * 4
        self.match_head = nn.Sequential(
            nn.Linear(fusion_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(tx_config.dropout),
            nn.Linear(self.d_model, 1),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(fusion_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(tx_config.dropout),
            nn.Linear(self.d_model, 1),
        )
        self.clause_head = nn.Sequential(
            nn.Linear(fusion_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(tx_config.dropout),
            nn.Linear(self.d_model, num_clauses),
        )
        self.evidence_head = nn.Sequential(
            nn.Linear(fusion_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(tx_config.dropout),
            nn.Linear(self.d_model, 1),
        )

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).float()
        denom = weights.sum(dim=1).clamp(min=1.0)
        return (x * weights).sum(dim=1) / denom

    def encode_rule_text(self, rule_input_ids: torch.Tensor, rule_attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = rule_input_ids.shape
        device = rule_input_ids.device
        positions = torch.arange(1, seq_len + 1, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.rule_token_emb(rule_input_ids) + self.rule_position_emb(positions)
        x = self.rule_encoder(x, src_key_padding_mask=~rule_attention_mask)
        x = self.rule_norm(x)
        pooled = self._masked_mean(x, rule_attention_mask)
        return x, pooled

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        memory, memory_mask, tx_pooled = self.tx_encoder.encode_events(batch)
        rule_hidden, _ = self.encode_rule_text(batch["rule_input_ids"], batch["rule_attention_mask"])
        cross_out, _ = self.cross_attn(
            query=rule_hidden,
            key=memory,
            value=memory,
            key_padding_mask=~memory_mask,
            need_weights=False,
        )
        rule_hidden = self.cross_norm(rule_hidden + cross_out)
        rule_pooled = self._masked_mean(rule_hidden, batch["rule_attention_mask"])

        fused = torch.cat(
            [
                tx_pooled,
                rule_pooled,
                tx_pooled * rule_pooled,
                torch.abs(tx_pooled - rule_pooled),
            ],
            dim=-1,
        )
        event_len = batch["event_mask"].shape[1]
        event_repr = memory[:, 1 : 1 + event_len]
        rule_expand = rule_pooled.unsqueeze(1).expand(-1, event_len, -1)
        event_fused = torch.cat(
            [
                event_repr,
                rule_expand,
                event_repr * rule_expand,
                torch.abs(event_repr - rule_expand),
            ],
            dim=-1,
        )

        return {
            "match_logit": self.match_head(fused).squeeze(-1),
            "uncertainty_logit": self.uncertainty_head(fused).squeeze(-1),
            "clause_logits": self.clause_head(fused),
            "evidence_logits": self.evidence_head(event_fused).squeeze(-1),
            "tx_embedding": tx_pooled,
            "rule_embedding": rule_pooled,
        }
