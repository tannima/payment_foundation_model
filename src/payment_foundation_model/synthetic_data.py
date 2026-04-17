from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


DOMAINS = ["AE", "Antom", "AlipayHK"]
CARD_COUNTRIES = ["US", "GB", "SG", "HK"]
ISSUERS = ["Chase", "BankOfAmerica", "Citi", "HSBC", "DBS"]
ITEM_CATEGORIES = ["daily", "grocery", "beauty", "electronics", "luxury", "gaming", "travel"]

DOMAIN_TO_ID = {name: idx + 1 for idx, name in enumerate(DOMAINS)}
COUNTRY_TO_ID = {name: idx + 1 for idx, name in enumerate(CARD_COUNTRIES)}
ISSUER_TO_ID = {name: idx + 1 for idx, name in enumerate(ISSUERS)}
ITEM_TO_ID = {name: idx + 1 for idx, name in enumerate(ITEM_CATEGORIES)}

RESULT_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "risk_reject",
    "risk_allow",
    "3ds_yes",
    "3ds_no",
    "bank_approve",
    "bank_decline",
    "pay_success",
    "pay_fail",
    "fraud_yes",
    "fraud_no",
    "cb_yes",
    "cb_no",
]
RESULT_TOKEN_TO_ID = {token: idx for idx, token in enumerate(RESULT_TOKENS)}
RESULT_PAD_ID = RESULT_TOKEN_TO_ID["<pad>"]
RESULT_BOS_ID = RESULT_TOKEN_TO_ID["<bos>"]
RESULT_EOS_ID = RESULT_TOKEN_TO_ID["<eos>"]
MAX_RESULT_TOKENS = 8

PATTERN_SUBTYPE_TO_FLAGS = {
    "benign_stable": {
        "fraud_label": 0,
        "is_hard_negative": 0,
        "is_switch_attack": 0,
        "is_shared_device_benign": 0,
        "is_travel_burst_benign": 0,
    },
    "benign_card_refresh": {
        "fraud_label": 0,
        "is_hard_negative": 1,
        "is_switch_attack": 0,
        "is_shared_device_benign": 0,
        "is_travel_burst_benign": 0,
    },
    "benign_travel_burst": {
        "fraud_label": 0,
        "is_hard_negative": 1,
        "is_switch_attack": 0,
        "is_shared_device_benign": 0,
        "is_travel_burst_benign": 1,
    },
    "benign_shared_device": {
        "fraud_label": 0,
        "is_hard_negative": 1,
        "is_switch_attack": 0,
        "is_shared_device_benign": 1,
        "is_travel_burst_benign": 0,
    },
    "suspicious_card_theft": {
        "fraud_label": 1,
        "is_hard_negative": 0,
        "is_switch_attack": 1,
        "is_shared_device_benign": 0,
        "is_travel_burst_benign": 0,
    },
    "suspicious_ato": {
        "fraud_label": 1,
        "is_hard_negative": 0,
        "is_switch_attack": 1,
        "is_shared_device_benign": 0,
        "is_travel_burst_benign": 0,
    },
    "suspicious_device_ring": {
        "fraud_label": 1,
        "is_hard_negative": 0,
        "is_switch_attack": 0,
        "is_shared_device_benign": 0,
        "is_travel_burst_benign": 0,
    },
}

CLAUSE_SPECS: Dict[str, Dict[str, Any]] = {
    "recent_multi_card_3d_ge_2": {
        "type": "distinct_card_count",
        "canonical_text": "用户近3天更换过多张卡",
        "templates": [
            "用户近3天更换过多张卡",
            "最近3天这位用户出现多卡切换",
            "近3天存在明显换卡行为",
        ],
        "window": "3d",
        "operator": ">=",
        "threshold": 2,
    },
    "same_device_multi_card_7d_ge_3": {
        "type": "same_device_distinct_card_count",
        "canonical_text": "同一设备近7天关联至少3张不同卡",
        "templates": [
            "同一设备近7天关联至少3张不同卡",
            "这台设备最近挂过多张卡",
            "设备侧出现了明显的一机多卡模式",
        ],
        "window": "7d",
        "operator": ">=",
        "threshold": 3,
    },
    "billing_profile_changed_7d": {
        "type": "billing_profile_changed",
        "canonical_text": "近7天账单邮箱或账单地址发生变化",
        "templates": [
            "近7天账单邮箱或账单地址发生变化",
            "账单身份近期出现变更",
            "账单邮箱或地址最近不稳定",
        ],
        "window": "7d",
    },
    "anchor_new_card": {
        "type": "new_card",
        "canonical_text": "当前交易使用了新的卡片",
        "templates": [
            "当前交易使用了新的卡片",
            "这笔支付切到了新卡",
            "当前卡片此前未在该用户上下文中出现",
        ],
    },
    "anchor_new_device": {
        "type": "new_device",
        "canonical_text": "当前交易使用了新的设备",
        "templates": [
            "当前交易使用了新的设备",
            "这笔交易发生在此前未见过的设备上",
            "设备侧发生了切换",
        ],
    },
    "anchor_new_email_or_address": {
        "type": "new_billing_identity",
        "canonical_text": "当前交易使用了新的账单邮箱或账单地址",
        "templates": [
            "当前交易使用了新的账单邮箱或账单地址",
            "账单身份在当前交易中发生了切换",
            "邮箱或地址与历史记录相比是新的",
        ],
    },
    "high_freq_high_amount_1d": {
        "type": "burst_high_amount",
        "canonical_text": "近1天出现高频高额交易",
        "templates": [
            "近1天出现高频高额交易",
            "短时间内连续出现高金额支付",
            "最近1天交易频率和金额都明显抬升",
        ],
        "window": "1d",
    },
    "travel_ip_burst_2d": {
        "type": "travel_like_ip_burst",
        "canonical_text": "近2天单卡在多个IP下出现密集交易",
        "templates": [
            "近2天单卡在多个IP下出现密集交易",
            "单张卡近期在多个IP环境中连续交易",
            "最近2天更像旅行场景下的多IP连续支付",
        ],
        "window": "2d",
    },
    "stable_device_recent": {
        "type": "stable_device",
        "canonical_text": "近期设备保持稳定",
        "templates": [
            "近期设备保持稳定",
            "设备侧没有明显变化",
            "最近一段时间设备基本一致",
        ],
        "window": "30d",
    },
    "stable_billing_recent": {
        "type": "stable_billing",
        "canonical_text": "近期账单身份保持稳定",
        "templates": [
            "近期账单身份保持稳定",
            "账单邮箱和地址没有明显变化",
            "最近一段时间账单身份较稳定",
        ],
        "window": "30d",
    },
    "same_card_recent_7d": {
        "type": "same_card_recent",
        "canonical_text": "近7天主要使用同一张卡",
        "templates": [
            "近7天主要使用同一张卡",
            "最近7天没有明显换卡",
            "近期支付工具整体保持一致",
        ],
        "window": "7d",
    },
}
CLAUSE_TYPES = list(CLAUSE_SPECS.keys())
CLAUSE_TYPE_TO_ID = {name: idx for idx, name in enumerate(CLAUSE_TYPES)}

RULE_SPECS: List[Dict[str, Any]] = [
    {
        "rule_id": "rule_recent_card_switch",
        "logic": "AND",
        "clauses": ["recent_multi_card_3d_ge_2"],
        "canonical_text": "用户近3天更换过多张卡",
        "templates": [
            "用户近3天更换过多张卡",
            "最近3天这位用户有明显换卡行为",
            "这笔交易前的上下文显示近3天存在多卡切换",
        ],
    },
    {
        "rule_id": "rule_device_multi_card",
        "logic": "AND",
        "clauses": ["same_device_multi_card_7d_ge_3"],
        "canonical_text": "同一设备近7天关联至少3张不同卡",
        "templates": [
            "同一设备近7天关联至少3张不同卡",
            "这台设备最近挂过很多卡",
            "设备侧出现了一机多卡模式",
        ],
    },
    {
        "rule_id": "rule_card_theft_like",
        "logic": "AND",
        "clauses": ["recent_multi_card_3d_ge_2", "billing_profile_changed_7d", "high_freq_high_amount_1d"],
        "canonical_text": "用户近3天切换过多张卡，且账单身份近期变化，并伴随近1天高频高额交易",
        "templates": [
            "用户近3天切换过多张卡，且账单身份近期变化，并伴随近1天高频高额交易",
            "卡片和账单身份同时变化，短时间内又出现高额密集支付",
            "更像盗卡后连续换卡并快速打高额单的模式",
        ],
    },
    {
        "rule_id": "rule_ato_like",
        "logic": "AND",
        "clauses": ["same_card_recent_7d", "billing_profile_changed_7d", "high_freq_high_amount_1d"],
        "canonical_text": "近期仍使用同一卡片，但账单身份近期变化，并伴随近1天高频高额交易",
        "templates": [
            "近期仍使用同一卡片，但账单身份近期变化，并伴随近1天高频高额交易",
            "卡没换，但账单身份突然变化，同时交易金额和频率抬升",
            "更像账户接管：同卡、账单身份变化、短时高额连续支付",
        ],
    },
    {
        "rule_id": "rule_device_ring_like",
        "logic": "AND",
        "clauses": ["same_device_multi_card_7d_ge_3", "high_freq_high_amount_1d"],
        "canonical_text": "同一设备近7天关联多张卡，且近1天出现高频高额交易",
        "templates": [
            "同一设备近7天关联多张卡，且近1天出现高频高额交易",
            "设备侧多卡共享，同时出现短时高额密集支付",
            "这更像设备团伙模式：一机多卡叠加高频高额下单",
        ],
    },
    {
        "rule_id": "rule_benign_card_refresh",
        "logic": "AND",
        "clauses": ["recent_multi_card_3d_ge_2", "stable_device_recent", "stable_billing_recent"],
        "canonical_text": "近期虽然有换卡，但设备和账单身份保持稳定",
        "templates": [
            "近期虽然有换卡，但设备和账单身份保持稳定",
            "更像正常补卡或换卡，其他身份介质没有明显变化",
            "虽然最近换了卡，但设备和账单档案都很稳定",
        ],
    },
    {
        "rule_id": "rule_benign_travel_burst",
        "logic": "AND",
        "clauses": ["same_card_recent_7d", "travel_ip_burst_2d", "stable_billing_recent"],
        "canonical_text": "近7天主要使用同一卡片，近2天在多个IP下连续交易，但账单身份稳定",
        "templates": [
            "近7天主要使用同一卡片，近2天在多个IP下连续交易，但账单身份稳定",
            "更像正常旅行导致的多IP密集支付，账单身份并未变化",
            "虽然IP环境变化较多，但卡和账单身份整体稳定，更接近旅行场景",
        ],
    },
    {
        "rule_id": "rule_benign_shared_device",
        "logic": "AND",
        "clauses": ["same_device_multi_card_7d_ge_3", "stable_billing_recent"],
        "canonical_text": "同一设备近7天关联多张卡，但账单身份近期保持稳定",
        "templates": [
            "同一设备近7天关联多张卡，但账单身份近期保持稳定",
            "虽然设备侧是一机多卡，但账单身份没有明显漂移",
            "更像家庭共享设备，而不是账单身份同步漂移的攻击模式",
        ],
    },
]

VERIFIER_DIFFICULTY_TO_ID = {
    "positive": 0,
    "hard_negative": 1,
    "semantic_negative": 2,
    "easy_negative": 3,
}


@dataclass(frozen=True)
class Event:
    timestamp: datetime
    amount: float
    user_id: str
    card_no: str
    device_id: Optional[str]
    bill_email: Optional[str]
    bill_address: Optional[str]
    ip_address: Optional[str]
    card_country: str
    issuer: str
    item_category: str


def _make_id(prefix: str, rng: random.Random, length: int = 16) -> str:
    alphabet = "0123456789abcdef"
    return prefix + "_" + "".join(rng.choice(alphabet) for _ in range(length))


def _choice(rng: random.Random, values: List[str]) -> str:
    return values[rng.randrange(len(values))]


def _make_event(
    when: datetime,
    amount: float,
    user_id: str,
    card_no: str,
    device_id: Optional[str],
    bill_email: Optional[str],
    bill_address: Optional[str],
    ip_address: Optional[str],
    card_country: str,
    issuer: str,
    item_category: str,
) -> Event:
    return Event(
        timestamp=when,
        amount=amount,
        user_id=user_id,
        card_no=card_no,
        device_id=device_id,
        bill_email=bill_email,
        bill_address=bill_address,
        ip_address=ip_address,
        card_country=card_country,
        issuer=issuer,
        item_category=item_category,
    )


def _domain_missing_fields(
    domain: str,
    device_id: Optional[str],
    bill_email: Optional[str],
    bill_address: Optional[str],
    ip_address: Optional[str],
    rng: random.Random,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    if domain == "AE":
        return device_id, bill_email, bill_address, ip_address
    if domain == "Antom":
        if rng.random() < 0.40:
            device_id = None
        if rng.random() < 0.25:
            bill_address = None
        return device_id, bill_email, bill_address, ip_address
    if domain == "AlipayHK":
        if rng.random() < 0.45:
            bill_email = None
        if rng.random() < 0.55:
            bill_address = None
        return device_id, bill_email, bill_address, ip_address
    return device_id, bill_email, bill_address, ip_address


def _localize_ids(events: List[Event], field_name: str) -> Dict[str, int]:
    values = []
    for event in events:
        value = getattr(event, field_name)
        if value is not None:
            values.append(value)
    unique_values = sorted(set(values))
    return {value: idx + 1 for idx, value in enumerate(unique_values)}


def _event_to_json(event: Event) -> Dict[str, Any]:
    payload = asdict(event)
    payload["timestamp"] = event.timestamp.isoformat()
    payload["amount"] = round(event.amount, 2)
    return payload


def _events_from_raw_sample(raw_sample: Dict[str, Any]) -> List[Event]:
    events = [
        Event(
            timestamp=datetime.fromisoformat(payload["timestamp"]),
            amount=float(payload["amount"]),
            user_id=payload["user_id"],
            card_no=payload["card_no"],
            device_id=payload["device_id"],
            bill_email=payload["bill_email"],
            bill_address=payload["bill_address"],
            ip_address=payload["ip_address"],
            card_country=payload["card_country"],
            issuer=payload["issuer"],
            item_category=payload["item_category"],
        )
        for payload in raw_sample["events"]
    ]
    return sorted(events, key=lambda event: event.timestamp)


def _window_events(events: List[Event], anchor: Event, hours: float) -> List[Tuple[int, Event]]:
    rows: List[Tuple[int, Event]] = []
    for idx, event in enumerate(events):
        delta_hours = max((anchor.timestamp - event.timestamp).total_seconds() / 3600.0, 0.0)
        if delta_hours <= hours:
            rows.append((idx, event))
    return rows


def _distinct_non_null(values: List[Optional[str]]) -> int:
    return len({value for value in values if value is not None})


def _is_stable(values: List[Optional[str]]) -> bool:
    observed = [value for value in values if value is not None]
    return len(observed) > 0 and len(set(observed)) <= 1


def _compute_verifier_features(raw_sample: Dict[str, Any]) -> Dict[str, Any]:
    events = _events_from_raw_sample(raw_sample)
    anchor = events[-1]
    history = events[:-1]

    recent_1d = _window_events(events, anchor, 24.0)
    recent_2d = _window_events(events, anchor, 48.0)
    recent_3d = _window_events(events, anchor, 24.0 * 3.0)
    recent_7d = _window_events(events, anchor, 24.0 * 7.0)
    recent_30d = _window_events(events, anchor, 24.0 * 30.0)

    recent_1d_events = [event for _, event in recent_1d]
    recent_2d_events = [event for _, event in recent_2d]
    recent_3d_events = [event for _, event in recent_3d]
    recent_7d_events = [event for _, event in recent_7d]
    recent_30d_events = [event for _, event in recent_30d]

    history_cards = {event.card_no for event in history}
    history_devices = {event.device_id for event in history if event.device_id is not None}
    history_emails = {event.bill_email for event in history if event.bill_email is not None}
    history_addresses = {event.bill_address for event in history if event.bill_address is not None}

    distinct_cards_3d = len({event.card_no for event in recent_3d_events})
    distinct_cards_7d = len({event.card_no for event in recent_7d_events})
    same_device_distinct_cards_7d = len(
        {
            event.card_no
            for event in recent_7d_events
            if anchor.device_id is not None and event.device_id == anchor.device_id
        }
    )
    distinct_ips_2d = _distinct_non_null([event.ip_address for event in recent_2d_events])
    distinct_emails_7d = _distinct_non_null([event.bill_email for event in recent_7d_events])
    distinct_addresses_7d = _distinct_non_null([event.bill_address for event in recent_7d_events])
    distinct_devices_30d = _distinct_non_null([event.device_id for event in recent_30d_events])
    distinct_emails_30d = _distinct_non_null([event.bill_email for event in recent_30d_events])
    distinct_addresses_30d = _distinct_non_null([event.bill_address for event in recent_30d_events])
    has_email_30d = any(event.bill_email is not None for event in recent_30d_events)
    has_address_30d = any(event.bill_address is not None for event in recent_30d_events)
    tx_count_1d = max(len(recent_1d_events) - 1, 0)
    high_amount_count_1d = sum(1 for event in recent_1d_events if event.amount >= 250.0)

    feature_values = {
        "recent_multi_card_3d_ge_2": distinct_cards_3d >= 2,
        "same_device_multi_card_7d_ge_3": same_device_distinct_cards_7d >= 3,
        "billing_profile_changed_7d": distinct_emails_7d >= 2 or distinct_addresses_7d >= 2,
        "anchor_new_card": anchor.card_no not in history_cards,
        "anchor_new_device": anchor.device_id is not None and anchor.device_id not in history_devices,
        "anchor_new_email_or_address": (
            (anchor.bill_email is not None and anchor.bill_email not in history_emails)
            or (anchor.bill_address is not None and anchor.bill_address not in history_addresses)
        ),
        "high_freq_high_amount_1d": tx_count_1d >= 3 and high_amount_count_1d >= 2,
        "travel_ip_burst_2d": distinct_cards_7d <= 1 and distinct_ips_2d >= 3 and tx_count_1d >= 3,
        "stable_device_recent": _is_stable([event.device_id for event in recent_30d_events]),
        "stable_billing_recent": (
            (distinct_emails_30d <= 1 if has_email_30d else True)
            and (distinct_addresses_30d <= 1 if has_address_30d else True)
            and (has_email_30d or has_address_30d)
        ),
        "same_card_recent_7d": distinct_cards_7d <= 1,
    }

    known = {
        "recent_multi_card_3d_ge_2": True,
        "same_device_multi_card_7d_ge_3": anchor.device_id is not None and any(
            event.device_id is not None for event in recent_7d_events
        ),
        "billing_profile_changed_7d": any(
            event.bill_email is not None or event.bill_address is not None for event in recent_7d_events
        ),
        "anchor_new_card": True,
        "anchor_new_device": anchor.device_id is not None and len(history_devices) > 0,
        "anchor_new_email_or_address": (
            (anchor.bill_email is not None and len(history_emails) > 0)
            or (anchor.bill_address is not None and len(history_addresses) > 0)
        ),
        "high_freq_high_amount_1d": True,
        "travel_ip_burst_2d": anchor.ip_address is not None and any(
            event.ip_address is not None for event in recent_2d_events
        ),
        "stable_device_recent": any(event.device_id is not None for event in recent_30d_events),
        "stable_billing_recent": any(
            event.bill_email is not None or event.bill_address is not None for event in recent_30d_events
        ),
        "same_card_recent_7d": True,
    }

    evidence = {
        "recent_multi_card_3d_ge_2": [idx for idx, _ in recent_3d],
        "same_device_multi_card_7d_ge_3": [
            idx for idx, event in recent_7d if anchor.device_id is not None and event.device_id == anchor.device_id
        ],
        "billing_profile_changed_7d": [
            idx
            for idx, event in recent_7d
            if (
                anchor.bill_email is not None
                and event.bill_email is not None
                and event.bill_email != anchor.bill_email
            )
            or (
                anchor.bill_address is not None
                and event.bill_address is not None
                and event.bill_address != anchor.bill_address
            )
        ],
        "anchor_new_card": [len(events) - 1],
        "anchor_new_device": [len(events) - 1],
        "anchor_new_email_or_address": [len(events) - 1],
        "high_freq_high_amount_1d": [
            idx for idx, event in recent_1d if event.amount >= 250.0 or idx == len(events) - 1
        ],
        "travel_ip_burst_2d": [idx for idx, _ in recent_2d],
        "stable_device_recent": [idx for idx, _ in recent_30d],
        "stable_billing_recent": [idx for idx, _ in recent_30d],
        "same_card_recent_7d": [idx for idx, _ in recent_7d],
    }

    metrics = {
        "distinct_cards_3d": distinct_cards_3d,
        "distinct_cards_7d": distinct_cards_7d,
        "same_device_distinct_cards_7d": same_device_distinct_cards_7d,
        "distinct_ips_2d": distinct_ips_2d,
        "distinct_emails_7d": distinct_emails_7d,
        "distinct_addresses_7d": distinct_addresses_7d,
        "tx_count_1d": tx_count_1d,
        "high_amount_count_1d": high_amount_count_1d,
    }

    return {
        "events": events,
        "anchor": anchor,
        "feature_values": feature_values,
        "known": known,
        "evidence": evidence,
        "metrics": metrics,
    }


def _select_template(templates: List[str], rng: random.Random) -> str:
    return templates[rng.randrange(len(templates))]


def _classify_verifier_difficulty(clause_labels: Dict[str, int], overall_match: int) -> str:
    if overall_match == 1:
        return "positive"
    positives = sum(clause_labels.values())
    total = len(clause_labels)
    if total > 1 and positives == total - 1:
        return "hard_negative"
    if positives > 0:
        return "semantic_negative"
    return "easy_negative"


def _build_rule_pair_rows(
    raw_sample: Dict[str, Any],
    max_pairs_per_sample: int,
    seed: int,
) -> List[Dict[str, Any]]:
    features = _compute_verifier_features(raw_sample)
    value_map = features["feature_values"]
    known_map = features["known"]
    evidence_map = features["evidence"]
    metrics = features["metrics"]
    rng = random.Random(seed)
    candidate_rows: List[Dict[str, Any]] = []

    for rule_spec in RULE_SPECS:
        clause_ids = rule_spec["clauses"]
        if not all(known_map.get(clause_id, False) for clause_id in clause_ids):
            continue

        clause_labels = {clause_id: int(bool(value_map[clause_id])) for clause_id in clause_ids}
        overall_match = int(all(clause_labels.values()))
        evidence_indices = sorted(
            {
                index
                for clause_id in clause_ids
                for index in evidence_map.get(clause_id, [])
            }
        )
        candidate_rows.append(
            {
                "sample_id": f"verifier_{raw_sample['sample_id']}_{rule_spec['rule_id']}",
                "transaction_context": {
                    "sample_id": raw_sample["sample_id"],
                    "domain": raw_sample["domain"],
                    "pattern_subtype": raw_sample["pattern_subtype"],
                    "events": raw_sample["events"],
                },
                "rule": {
                    "rule_id": rule_spec["rule_id"],
                    "raw_text": _select_template(rule_spec["templates"], rng),
                    "canonical_text": rule_spec["canonical_text"],
                    "logic": rule_spec["logic"],
                    "clauses": [
                        {
                            "clause_id": clause_id,
                            "type": CLAUSE_SPECS[clause_id]["type"],
                            "text": _select_template(CLAUSE_SPECS[clause_id]["templates"], rng),
                            "canonical_text": CLAUSE_SPECS[clause_id]["canonical_text"],
                            "window": CLAUSE_SPECS[clause_id].get("window"),
                            "operator": CLAUSE_SPECS[clause_id].get("operator"),
                            "threshold": CLAUSE_SPECS[clause_id].get("threshold"),
                        }
                        for clause_id in clause_ids
                    ],
                },
                "labels": {
                    "overall_match": overall_match,
                    "uncertain": 0,
                    "clause_labels": clause_labels,
                    "evidence_event_indices": evidence_indices,
                },
                "meta": {
                    "source": "synthetic_rule_builder",
                    "domain": raw_sample["domain"],
                    "pattern_subtype": raw_sample["pattern_subtype"],
                    "difficulty": _classify_verifier_difficulty(clause_labels, overall_match),
                    "metrics": metrics,
                },
            }
        )

    grouped: Dict[str, List[Dict[str, Any]]] = {
        "positive": [],
        "hard_negative": [],
        "semantic_negative": [],
        "easy_negative": [],
    }
    for row in candidate_rows:
        grouped[row["meta"]["difficulty"]].append(row)

    selected: List[Dict[str, Any]] = []
    quotas = [
        ("positive", 2),
        ("hard_negative", 2),
        ("semantic_negative", 1),
        ("easy_negative", 1),
    ]
    for difficulty, limit in quotas:
        rows = grouped[difficulty]
        rng.shuffle(rows)
        selected.extend(rows[:limit])

    if len(selected) < max_pairs_per_sample:
        remaining = [row for row in candidate_rows if row not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: max_pairs_per_sample - len(selected)])

    return selected[:max_pairs_per_sample]


def _build_result_chain_for_pattern(
    pattern_subtype: str,
    rng: random.Random,
) -> Tuple[List[str], Dict[str, Any]]:
    chain = ["<bos>"]
    details: Dict[str, Any] = {
        "risk_decision": None,
        "three_ds": None,
        "bank_decision": None,
        "payment_outcome": None,
        "is_fraud": None,
        "is_cb": None,
    }

    if pattern_subtype in {"suspicious_card_theft", "suspicious_ato", "suspicious_device_ring"}:
        risk_reject_prob = {
            "suspicious_card_theft": 0.18,
            "suspicious_ato": 0.10,
            "suspicious_device_ring": 0.12,
        }[pattern_subtype]
        if rng.random() < risk_reject_prob:
            chain.extend(["risk_reject", "<eos>"])
            details["risk_decision"] = "reject"
            return chain, details

        chain.append("risk_allow")
        details["risk_decision"] = "allow"
        three_ds = "3ds_yes" if rng.random() < 0.45 else "3ds_no"
        chain.append(three_ds)
        details["three_ds"] = "yes" if three_ds == "3ds_yes" else "no"

        if rng.random() < 0.18:
            chain.extend(["bank_decline", "<eos>"])
            details["bank_decision"] = "decline"
            return chain, details

        chain.append("bank_approve")
        details["bank_decision"] = "approve"

        if rng.random() < 0.08:
            chain.extend(["pay_fail", "<eos>"])
            details["payment_outcome"] = "fail"
            return chain, details

        chain.append("pay_success")
        details["payment_outcome"] = "success"

        fraud_yes_prob = {
            "suspicious_card_theft": 0.88,
            "suspicious_ato": 0.75,
            "suspicious_device_ring": 0.82,
        }[pattern_subtype]
        fraud_yes = rng.random() < fraud_yes_prob
        chain.append("fraud_yes" if fraud_yes else "fraud_no")
        details["is_fraud"] = bool(fraud_yes)
        cb_yes = fraud_yes and rng.random() < 0.40
        chain.append("cb_yes" if cb_yes else "cb_no")
        details["is_cb"] = bool(cb_yes)
        chain.append("<eos>")
        return chain, details

    risk_reject_prob = {
        "benign_stable": 0.02,
        "benign_card_refresh": 0.03,
        "benign_travel_burst": 0.04,
        "benign_shared_device": 0.05,
    }[pattern_subtype]
    if rng.random() < risk_reject_prob:
        chain.extend(["risk_reject", "<eos>"])
        details["risk_decision"] = "reject"
        return chain, details

    chain.append("risk_allow")
    details["risk_decision"] = "allow"
    three_ds_prob = {
        "benign_stable": 0.12,
        "benign_card_refresh": 0.20,
        "benign_travel_burst": 0.35,
        "benign_shared_device": 0.22,
    }[pattern_subtype]
    three_ds = "3ds_yes" if rng.random() < three_ds_prob else "3ds_no"
    chain.append(three_ds)
    details["three_ds"] = "yes" if three_ds == "3ds_yes" else "no"

    bank_decline_prob = {
        "benign_stable": 0.08,
        "benign_card_refresh": 0.12,
        "benign_travel_burst": 0.15,
        "benign_shared_device": 0.12,
    }[pattern_subtype]
    if rng.random() < bank_decline_prob:
        chain.extend(["bank_decline", "<eos>"])
        details["bank_decision"] = "decline"
        return chain, details

    chain.append("bank_approve")
    details["bank_decision"] = "approve"

    pay_fail_prob = {
        "benign_stable": 0.06,
        "benign_card_refresh": 0.06,
        "benign_travel_burst": 0.10,
        "benign_shared_device": 0.08,
    }[pattern_subtype]
    if rng.random() < pay_fail_prob:
        chain.extend(["pay_fail", "<eos>"])
        details["payment_outcome"] = "fail"
        return chain, details

    chain.extend(["pay_success", "fraud_no", "cb_no", "<eos>"])
    details["payment_outcome"] = "success"
    details["is_fraud"] = False
    details["is_cb"] = False
    return chain, details


def _pad_result_chain(chain_tokens: List[str]) -> Dict[str, torch.Tensor]:
    token_ids = [RESULT_TOKEN_TO_ID[token] for token in chain_tokens]
    input_ids = token_ids[:-1]
    target_ids = token_ids[1:]
    valid_len = len(input_ids)

    padded_input = torch.full((MAX_RESULT_TOKENS,), RESULT_PAD_ID, dtype=torch.long)
    padded_target = torch.full((MAX_RESULT_TOKENS,), RESULT_PAD_ID, dtype=torch.long)
    result_mask = torch.zeros(MAX_RESULT_TOKENS, dtype=torch.bool)

    padded_input[:valid_len] = torch.tensor(input_ids, dtype=torch.long)
    padded_target[:valid_len] = torch.tensor(target_ids, dtype=torch.long)
    result_mask[:valid_len] = True
    return {
        "result_input_ids": padded_input,
        "result_target_ids": padded_target,
        "result_mask": result_mask,
    }


def _anchor_time() -> datetime:
    return datetime(2026, 3, 22, 7, 18, 43)


def _generate_benign_stable(domain: str, rng: random.Random) -> List[Event]:
    anchor_time = _anchor_time()
    user_id = _make_id("usr", rng, 24)
    card_1 = _make_id("card", rng, 24)
    card_2 = _make_id("card", rng, 24)
    device_1 = _make_id("dev", rng, 20)
    device_2 = _make_id("dev", rng, 20)
    email_1 = "alex.home@example.com"
    email_2 = "alex.travel@example.com"
    address_1 = "1209 Maple Ave, Columbus, OH"
    address_2 = "88 Lakeshore Rd, Chicago, IL"
    ip_1 = "73.162.54.91"
    ip_2 = "73.162.54.114"
    country = _choice(rng, CARD_COUNTRIES)
    issuer = _choice(rng, ISSUERS)

    months_back = [370, 240, 140, 45, 10, 4]
    history = []
    for idx, days in enumerate(months_back):
        when = anchor_time - timedelta(days=days, hours=rng.randint(0, 23))
        card = card_1 if idx < 4 else card_2
        device = device_1 if idx != 4 else device_2
        email = email_1 if idx < 5 else email_2
        address = address_1 if idx < 5 else address_2
        ip = ip_1 if idx < 5 else ip_2
        device, email, address, ip = _domain_missing_fields(domain, device, email, address, ip, rng)
        amount = rng.uniform(6.0, 120.0)
        item = _choice(rng, ["daily", "grocery", "beauty", "electronics"])
        history.append(
            _make_event(
                when, amount, user_id, card, device, email, address, ip, country, issuer, item
            )
        )

    device, email, address, ip = _domain_missing_fields(domain, device_2, email_2, address_2, ip_2, rng)
    history.append(
        _make_event(
            anchor_time,
            rng.uniform(30.0, 180.0),
            user_id,
            card_2,
            device,
            email,
            address,
            ip,
            country,
            issuer,
            _choice(rng, ["daily", "grocery", "beauty", "electronics"]),
        )
    )
    return history


def _generate_benign_card_refresh(domain: str, rng: random.Random) -> List[Event]:
    anchor_time = _anchor_time()
    user_id = _make_id("usr", rng, 24)
    old_card = _make_id("card", rng, 24)
    new_card = _make_id("card", rng, 24)
    device = _make_id("dev", rng, 20)
    email = "alex.refresh@example.com"
    address = "1209 Maple Ave, Columbus, OH"
    ip = "73.162.54.91"
    country = "US"
    issuer = "Chase"
    events = []

    for days in [300, 160, 40, 8]:
        d, e, a, i = _domain_missing_fields(domain, device, email, address, ip, rng)
        events.append(
            _make_event(
                anchor_time - timedelta(days=days),
                rng.uniform(12.0, 90.0),
                user_id,
                old_card,
                d,
                e,
                a,
                i,
                country,
                issuer,
                _choice(rng, ["daily", "grocery", "beauty"]),
            )
        )

    for hours in [60, 32, 12]:
        d, e, a, i = _domain_missing_fields(domain, device, email, address, ip, rng)
        events.append(
            _make_event(
                anchor_time - timedelta(hours=hours),
                rng.uniform(45.0, 180.0),
                user_id,
                new_card,
                d,
                e,
                a,
                i,
                country,
                issuer,
                _choice(rng, ["grocery", "electronics"]),
            )
        )

    d, e, a, i = _domain_missing_fields(domain, device, email, address, ip, rng)
    events.append(
        _make_event(
            anchor_time,
            rng.uniform(50.0, 220.0),
            user_id,
            new_card,
            d,
            e,
            a,
            i,
            country,
            issuer,
            _choice(rng, ["electronics", "daily"]),
        )
    )
    return events


def _generate_benign_travel_burst(domain: str, rng: random.Random) -> List[Event]:
    anchor_time = _anchor_time()
    user_id = _make_id("usr", rng, 24)
    card = _make_id("card", rng, 24)
    device = _make_id("dev", rng, 20)
    email = "alex.travel@example.com"
    address = "88 Lakeshore Rd, Chicago, IL"
    ips = ["34.201.118.19", "34.201.118.23", "52.17.211.8", "18.162.10.4"]
    country = "US"
    issuer = "Citi"
    events = []

    for days in [330, 210, 80]:
        d, e, a, i = _domain_missing_fields(domain, device, email, address, ips[0], rng)
        events.append(
            _make_event(
                anchor_time - timedelta(days=days),
                rng.uniform(20.0, 120.0),
                user_id,
                card,
                d,
                e,
                a,
                i,
                country,
                issuer,
                _choice(rng, ["daily", "grocery", "beauty"]),
            )
        )

    for idx, hours in enumerate([48, 30, 18, 8]):
        d, e, a, i = _domain_missing_fields(domain, device, email, address, ips[idx % len(ips)], rng)
        events.append(
            _make_event(
                anchor_time - timedelta(hours=hours),
                rng.uniform(180.0, 450.0),
                user_id,
                card,
                d,
                e,
                a,
                i,
                country,
                issuer,
                _choice(rng, ["travel", "electronics"]),
            )
        )

    d, e, a, i = _domain_missing_fields(domain, device, email, address, ips[-1], rng)
    events.append(
        _make_event(
            anchor_time,
            rng.uniform(220.0, 520.0),
            user_id,
            card,
            d,
            e,
            a,
            i,
            country,
            issuer,
            _choice(rng, ["travel", "electronics"]),
        )
    )
    return events


def _generate_benign_shared_device(domain: str, rng: random.Random) -> List[Event]:
    anchor_time = _anchor_time()
    user_id = _make_id("usr", rng, 24)
    card_1 = _make_id("card", rng, 24)
    card_2 = _make_id("card", rng, 24)
    card_3 = _make_id("card", rng, 24)
    shared_device = _make_id("dev", rng, 20)
    email_1 = "alex.family@example.com"
    email_2 = "alex.wallet@example.com"
    address = "210 Parkside Dr, Seattle, WA"
    ips = ["73.162.54.91", "73.162.54.114"]
    country = "US"
    issuer = "BankOfAmerica"
    events = []

    for idx, days in enumerate([260, 160, 60, 14]):
        card = [card_1, card_2, card_1, card_3][idx]
        email = email_1 if idx < 2 else email_2
        d, e, a, i = _domain_missing_fields(domain, shared_device, email, address, ips[idx % 2], rng)
        events.append(
            _make_event(
                anchor_time - timedelta(days=days),
                rng.uniform(15.0, 110.0),
                user_id,
                card,
                d,
                e,
                a,
                i,
                country,
                issuer,
                _choice(rng, ["daily", "grocery", "beauty", "electronics"]),
            )
        )

    for idx, hours in enumerate([30, 10]):
        d, e, a, i = _domain_missing_fields(domain, shared_device, email_2, address, ips[idx], rng)
        events.append(
            _make_event(
                anchor_time - timedelta(hours=hours),
                rng.uniform(80.0, 260.0),
                user_id,
                card_3 if idx == 0 else card_2,
                d,
                e,
                a,
                i,
                country,
                issuer,
                _choice(rng, ["electronics", "gaming"]),
            )
        )

    d, e, a, i = _domain_missing_fields(domain, shared_device, email_2, address, ips[-1], rng)
    events.append(
        _make_event(
            anchor_time,
            rng.uniform(120.0, 300.0),
            user_id,
            card_2,
            d,
            e,
            a,
            i,
            country,
            issuer,
            _choice(rng, ["electronics", "gaming"]),
        )
    )
    return events


def _generate_suspicious_card_theft(domain: str, rng: random.Random) -> List[Event]:
    anchor_time = _anchor_time()
    user_id = _make_id("usr", rng, 24)
    old_card = _make_id("card", rng, 24)
    new_card = _make_id("card", rng, 24)
    base_device = _make_id("dev", rng, 20)
    old_email = "alex.home@example.net"
    new_email = "alex.payments@example.com"
    old_address = "1209 Maple Ave, Columbus, OH"
    new_address = "742 Evergreen Terrace, Springfield, IL"
    old_ip = "73.162.54.91"
    burst_ips = ["34.201.118.19", "34.201.118.23", "34.201.118.27", "34.201.118.44"]
    events = []

    for days in [430, 290, 120]:
        d, e, a, i = _domain_missing_fields(domain, base_device, old_email, old_address, old_ip, rng)
        events.append(
            _make_event(
                anchor_time - timedelta(days=days, hours=rng.randint(0, 23)),
                rng.uniform(5.0, 28.0),
                user_id,
                old_card,
                d,
                e,
                a,
                i,
                "US",
                "BankOfAmerica",
                _choice(rng, ["daily", "grocery", "beauty"]),
            )
        )

    for idx, hours_before in enumerate([52, 45, 43, 40, 31, 28, 24, 7]):
        d, e, a, i = _domain_missing_fields(domain, base_device, new_email, new_address, burst_ips[idx % 4], rng)
        events.append(
            _make_event(
                anchor_time - timedelta(hours=hours_before),
                [329.0, 499.0, 899.0, 1249.0, 799.0, 1599.0, 999.0, 129.0][idx],
                user_id,
                new_card,
                d,
                e,
                a,
                i,
                "US",
                "Chase",
                ["electronics", "gaming", "electronics", "electronics", "electronics", "luxury", "electronics", "electronics"][idx],
            )
        )

    d, e, a, i = _domain_missing_fields(domain, base_device, new_email, new_address, burst_ips[-1], rng)
    events.append(
        _make_event(
            anchor_time,
            129.0,
            user_id,
            new_card,
            d,
            e,
            a,
            i,
            "US",
            "Chase",
            "electronics",
        )
    )
    return events


def _generate_suspicious_ato(domain: str, rng: random.Random) -> List[Event]:
    anchor_time = _anchor_time()
    user_id = _make_id("usr", rng, 24)
    old_card = _make_id("card", rng, 24)
    same_card = old_card
    trusted_device = _make_id("dev", rng, 20)
    risky_device = _make_id("dev", rng, 20)
    old_email = "alex.member@example.com"
    new_email = "alex.takeover@example.com"
    old_address = "910 Elm St, Austin, TX"
    new_address = "742 Evergreen Terrace, Springfield, IL"
    ips = ["73.162.54.91", "34.201.118.44", "34.201.118.27"]
    events = []

    for days in [360, 180, 40]:
        d, e, a, i = _domain_missing_fields(domain, trusted_device, old_email, old_address, ips[0], rng)
        events.append(
            _make_event(
                anchor_time - timedelta(days=days),
                rng.uniform(12.0, 75.0),
                user_id,
                old_card,
                d,
                e,
                a,
                i,
                "US",
                "Citi",
                _choice(rng, ["daily", "grocery", "beauty"]),
            )
        )

    for idx, hours in enumerate([36, 18, 5]):
        d, e, a, i = _domain_missing_fields(domain, risky_device, new_email, new_address, ips[(idx + 1) % len(ips)], rng)
        events.append(
            _make_event(
                anchor_time - timedelta(hours=hours),
                rng.uniform(280.0, 920.0),
                user_id,
                same_card,
                d,
                e,
                a,
                i,
                "US",
                "Citi",
                _choice(rng, ["electronics", "luxury"]),
            )
        )

    d, e, a, i = _domain_missing_fields(domain, risky_device, new_email, new_address, ips[-1], rng)
    events.append(
        _make_event(
            anchor_time,
            rng.uniform(180.0, 760.0),
            user_id,
            same_card,
            d,
            e,
            a,
            i,
            "US",
            "Citi",
            _choice(rng, ["electronics", "luxury"]),
        )
    )
    return events


def _generate_suspicious_device_ring(domain: str, rng: random.Random) -> List[Event]:
    anchor_time = _anchor_time()
    user_id = _make_id("usr", rng, 24)
    card_1 = _make_id("card", rng, 24)
    card_2 = _make_id("card", rng, 24)
    card_3 = _make_id("card", rng, 24)
    ring_device = _make_id("dev", rng, 20)
    email_1 = "alex.ring1@example.com"
    email_2 = "alex.ring2@example.com"
    address_1 = "31 Harbor Rd, Miami, FL"
    address_2 = "742 Evergreen Terrace, Springfield, IL"
    ips = ["34.201.118.19", "34.201.118.23", "34.201.118.27"]
    events = []

    for idx, days in enumerate([200, 120]):
        d, e, a, i = _domain_missing_fields(domain, ring_device, email_1, address_1, ips[0], rng)
        events.append(
            _make_event(
                anchor_time - timedelta(days=days),
                rng.uniform(10.0, 40.0),
                user_id,
                card_1,
                d,
                e,
                a,
                i,
                "US",
                "HSBC",
                _choice(rng, ["daily", "beauty"]),
            )
        )

    sequence = [
        (card_1, email_1, address_1, 30, 249.0, "electronics"),
        (card_2, email_2, address_2, 18, 699.0, "electronics"),
        (card_3, email_2, address_2, 12, 1189.0, "luxury"),
        (card_2, email_2, address_2, 6, 850.0, "gaming"),
    ]
    for idx, (card, email, address, hours, amount, item) in enumerate(sequence):
        d, e, a, i = _domain_missing_fields(domain, ring_device, email, address, ips[idx % len(ips)], rng)
        events.append(
            _make_event(
                anchor_time - timedelta(hours=hours),
                amount,
                user_id,
                card,
                d,
                e,
                a,
                i,
                "US",
                "HSBC",
                item,
            )
        )

    d, e, a, i = _domain_missing_fields(domain, ring_device, email_2, address_2, ips[-1], rng)
    events.append(
        _make_event(
            anchor_time,
            499.0,
            user_id,
            card_3,
            d,
            e,
            a,
            i,
            "US",
            "HSBC",
            "electronics",
        )
    )
    return events


def _build_sample_tensors(
    raw_sample: Dict[str, Any],
    max_events: int,
) -> Dict[str, torch.Tensor]:
    events = _events_from_raw_sample(raw_sample)
    anchor = events[-1]
    history = events[:-1]
    if len(events) > max_events:
        history = history[-(max_events - 1) :]
        events = history + [anchor]
    valid_len = len(events)

    local_maps = {
        "user_id": _localize_ids(events, "user_id"),
        "card_no": _localize_ids(events, "card_no"),
        "device_id": _localize_ids(events, "device_id"),
        "bill_email": _localize_ids(events, "bill_email"),
        "bill_address": _localize_ids(events, "bill_address"),
        "ip_address": _localize_ids(events, "ip_address"),
    }

    amount_log = torch.zeros(max_events, dtype=torch.float32)
    delta_hours_log = torch.zeros(max_events, dtype=torch.float32)
    user_local = torch.zeros(max_events, dtype=torch.long)
    card_local = torch.zeros(max_events, dtype=torch.long)
    device_local = torch.zeros(max_events, dtype=torch.long)
    email_local = torch.zeros(max_events, dtype=torch.long)
    address_local = torch.zeros(max_events, dtype=torch.long)
    ip_local = torch.zeros(max_events, dtype=torch.long)
    card_country_id = torch.zeros(max_events, dtype=torch.long)
    issuer_id = torch.zeros(max_events, dtype=torch.long)
    item_id = torch.zeros(max_events, dtype=torch.long)
    is_anchor = torch.zeros(max_events, dtype=torch.float32)
    same_user = torch.zeros(max_events, dtype=torch.float32)
    same_card = torch.zeros(max_events, dtype=torch.float32)
    same_device = torch.zeros(max_events, dtype=torch.float32)
    same_email = torch.zeros(max_events, dtype=torch.float32)
    same_address = torch.zeros(max_events, dtype=torch.float32)
    same_ip = torch.zeros(max_events, dtype=torch.float32)
    has_device = torch.zeros(max_events, dtype=torch.float32)
    has_email = torch.zeros(max_events, dtype=torch.float32)
    has_address = torch.zeros(max_events, dtype=torch.float32)
    has_ip = torch.zeros(max_events, dtype=torch.float32)
    event_mask = torch.zeros(max_events, dtype=torch.bool)

    recent_1d_count = 0.0
    recent_7d_cards = set()
    recent_7d_amount = 0.0
    same_device_cards_7d = set()

    for idx, event in enumerate(events):
        delta_hours = (anchor.timestamp - event.timestamp).total_seconds() / 3600.0
        amount_log[idx] = math.log1p(event.amount)
        delta_hours_log[idx] = math.log1p(max(delta_hours, 0.0))
        user_local[idx] = local_maps["user_id"].get(event.user_id, 0)
        card_local[idx] = local_maps["card_no"].get(event.card_no, 0)
        if event.device_id is not None:
            device_local[idx] = local_maps["device_id"].get(event.device_id, 0)
            has_device[idx] = 1.0
        if event.bill_email is not None:
            email_local[idx] = local_maps["bill_email"].get(event.bill_email, 0)
            has_email[idx] = 1.0
        if event.bill_address is not None:
            address_local[idx] = local_maps["bill_address"].get(event.bill_address, 0)
            has_address[idx] = 1.0
        if event.ip_address is not None:
            ip_local[idx] = local_maps["ip_address"].get(event.ip_address, 0)
            has_ip[idx] = 1.0
        card_country_id[idx] = COUNTRY_TO_ID[event.card_country]
        issuer_id[idx] = ISSUER_TO_ID[event.issuer]
        item_id[idx] = ITEM_TO_ID[event.item_category]
        is_anchor[idx] = 1.0 if idx == valid_len - 1 else 0.0
        same_user[idx] = 1.0 if event.user_id == anchor.user_id else 0.0
        same_card[idx] = 1.0 if event.card_no == anchor.card_no else 0.0
        same_device[idx] = 1.0 if event.device_id is not None and event.device_id == anchor.device_id else 0.0
        same_email[idx] = 1.0 if event.bill_email is not None and event.bill_email == anchor.bill_email else 0.0
        same_address[idx] = 1.0 if event.bill_address is not None and event.bill_address == anchor.bill_address else 0.0
        same_ip[idx] = 1.0 if event.ip_address is not None and event.ip_address == anchor.ip_address else 0.0
        event_mask[idx] = True

        if idx != valid_len - 1:
            if delta_hours <= 24.0:
                recent_1d_count += 1.0
            if delta_hours <= 24.0 * 7.0:
                recent_7d_cards.add(event.card_no)
                recent_7d_amount += event.amount
                if anchor.device_id is not None and event.device_id == anchor.device_id:
                    same_device_cards_7d.add(event.card_no)

    chain = _pad_result_chain(raw_sample["result_chain"])
    flags = PATTERN_SUBTYPE_TO_FLAGS[raw_sample["pattern_subtype"]]

    return {
        "amount_log": amount_log,
        "delta_hours_log": delta_hours_log,
        "user_local_id": user_local,
        "card_local_id": card_local,
        "device_local_id": device_local,
        "email_local_id": email_local,
        "address_local_id": address_local,
        "ip_local_id": ip_local,
        "card_country_id": card_country_id,
        "issuer_id": issuer_id,
        "item_category_id": item_id,
        "is_anchor": is_anchor,
        "same_user_as_anchor": same_user,
        "same_card_as_anchor": same_card,
        "same_device_as_anchor": same_device,
        "same_email_as_anchor": same_email,
        "same_address_as_anchor": same_address,
        "same_ip_as_anchor": same_ip,
        "has_device": has_device,
        "has_email": has_email,
        "has_address": has_address,
        "has_ip": has_ip,
        "event_mask": event_mask,
        "domain_id": torch.tensor(DOMAIN_TO_ID[raw_sample["domain"]], dtype=torch.long),
        "fraud_label": torch.tensor(float(flags["fraud_label"]), dtype=torch.float32),
        "tx_count_1d_log": torch.tensor(math.log1p(recent_1d_count), dtype=torch.float32),
        "distinct_card_count_7d_log": torch.tensor(math.log1p(float(len(recent_7d_cards))), dtype=torch.float32),
        "amount_sum_7d_log": torch.tensor(math.log1p(recent_7d_amount), dtype=torch.float32),
        "same_device_distinct_card_7d_log": torch.tensor(math.log1p(float(len(same_device_cards_7d))), dtype=torch.float32),
        "is_hard_negative": torch.tensor(float(flags["is_hard_negative"]), dtype=torch.float32),
        "is_switch_attack": torch.tensor(float(flags["is_switch_attack"]), dtype=torch.float32),
        "is_shared_device_benign": torch.tensor(float(flags["is_shared_device_benign"]), dtype=torch.float32),
        "is_travel_burst_benign": torch.tensor(float(flags["is_travel_burst_benign"]), dtype=torch.float32),
        **chain,
    }


def _build_raw_sample(sample_id: int, seed: int, pattern_subtype: str, domain: str) -> Dict[str, Any]:
    rng = random.Random(seed)
    if pattern_subtype == "benign_stable":
        events = _generate_benign_stable(domain, rng)
    elif pattern_subtype == "benign_card_refresh":
        events = _generate_benign_card_refresh(domain, rng)
    elif pattern_subtype == "benign_travel_burst":
        events = _generate_benign_travel_burst(domain, rng)
    elif pattern_subtype == "benign_shared_device":
        events = _generate_benign_shared_device(domain, rng)
    elif pattern_subtype == "suspicious_card_theft":
        events = _generate_suspicious_card_theft(domain, rng)
    elif pattern_subtype == "suspicious_ato":
        events = _generate_suspicious_ato(domain, rng)
    elif pattern_subtype == "suspicious_device_ring":
        events = _generate_suspicious_device_ring(domain, rng)
    else:
        raise ValueError(f"Unsupported pattern subtype: {pattern_subtype}")

    result_chain, outcome_details = _build_result_chain_for_pattern(pattern_subtype, rng)
    raw_sample = {
        "sample_id": sample_id,
        "domain": domain,
        "pattern_subtype": pattern_subtype,
        "pattern_type": "suspicious" if PATTERN_SUBTYPE_TO_FLAGS[pattern_subtype]["fraud_label"] == 1 else "benign",
        "fraud_label": PATTERN_SUBTYPE_TO_FLAGS[pattern_subtype]["fraud_label"],
        "result_chain": result_chain,
        "outcome_details": outcome_details,
        "events": [_event_to_json(event) for event in events],
    }
    return raw_sample


class SyntheticRiskDataset(Dataset):
    def __init__(
        self,
        size: int,
        max_events: int = 32,
        fraud_rate: float = 0.45,
        seed: int = 7,
        hard_mode: bool = True,
    ) -> None:
        self.size = size
        self.max_events = max_events
        self.fraud_rate = fraud_rate
        self.seed = seed
        self.hard_mode = hard_mode
        self.raw_samples = [self._make_raw_sample(idx) for idx in range(size)]
        self.samples = [_build_sample_tensors(raw_sample, max_events=self.max_events) for raw_sample in self.raw_samples]

    def _pick_pattern(self, rng: random.Random, suspicious: bool) -> str:
        if suspicious:
            candidates = ["suspicious_card_theft", "suspicious_ato", "suspicious_device_ring"]
            weights = [0.45, 0.30, 0.25]
        elif self.hard_mode:
            candidates = ["benign_stable", "benign_card_refresh", "benign_travel_burst", "benign_shared_device"]
            weights = [0.30, 0.25, 0.20, 0.25]
        else:
            candidates = ["benign_stable"]
            weights = [1.0]

        cutoff = rng.random() * sum(weights)
        running = 0.0
        for pattern, weight in zip(candidates, weights):
            running += weight
            if cutoff <= running:
                return pattern
        return candidates[-1]

    def _make_raw_sample(self, idx: int) -> Dict[str, Any]:
        rng = random.Random(self.seed + idx * 9973)
        domain = _choice(rng, DOMAINS)
        suspicious = rng.random() < self.fraud_rate
        pattern_subtype = self._pick_pattern(rng, suspicious=suspicious)
        return _build_raw_sample(
            sample_id=idx,
            seed=self.seed + idx * 9973 + 17,
            pattern_subtype=pattern_subtype,
            domain=domain,
        )

    def export_jsonl(self, path: str | Path, limit: Optional[int] = None) -> None:
        export_path = Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.raw_samples if limit is None else self.raw_samples[:limit]
        with export_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.samples[index]


class SyntheticRuleVerifierDataset(Dataset):
    def __init__(
        self,
        size: int,
        max_events: int = 32,
        fraud_rate: float = 0.45,
        seed: int = 7,
        hard_mode: bool = True,
        max_pairs_per_sample: int = 6,
    ) -> None:
        self.base_dataset = SyntheticRiskDataset(
            size=size,
            max_events=max_events,
            fraud_rate=fraud_rate,
            seed=seed,
            hard_mode=hard_mode,
        )
        self.max_events = max_events
        self.max_pairs_per_sample = max_pairs_per_sample
        self.rows: List[Dict[str, Any]] = []
        self.samples: List[Dict[str, Any]] = []

        for raw_sample, tx_sample in zip(self.base_dataset.raw_samples, self.base_dataset.samples):
            pair_rows = _build_rule_pair_rows(
                raw_sample,
                max_pairs_per_sample=self.max_pairs_per_sample,
                seed=seed + raw_sample["sample_id"] * 104729,
            )
            for row in pair_rows:
                self.rows.append(row)
                sample = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in tx_sample.items()}
                clause_target = torch.zeros(len(CLAUSE_TYPES), dtype=torch.float32)
                clause_mask = torch.zeros(len(CLAUSE_TYPES), dtype=torch.bool)
                for clause_id, label in row["labels"]["clause_labels"].items():
                    clause_idx = CLAUSE_TYPE_TO_ID[clause_id]
                    clause_target[clause_idx] = float(label)
                    clause_mask[clause_idx] = True

                evidence_mask = torch.zeros(self.max_events, dtype=torch.bool)
                for event_idx in row["labels"]["evidence_event_indices"]:
                    if 0 <= event_idx < self.max_events:
                        evidence_mask[event_idx] = True

                sample.update(
                    {
                        "verifier_rule_text": row["rule"]["raw_text"],
                        "verifier_rule_canonical_text": row["rule"]["canonical_text"],
                        "verifier_rule_id": row["rule"]["rule_id"],
                        "verifier_overall_match": torch.tensor(float(row["labels"]["overall_match"]), dtype=torch.float32),
                        "verifier_uncertain": torch.tensor(float(row["labels"]["uncertain"]), dtype=torch.float32),
                        "verifier_clause_target": clause_target,
                        "verifier_clause_mask": clause_mask,
                        "verifier_evidence_mask": evidence_mask,
                        "verifier_difficulty_id": torch.tensor(
                            VERIFIER_DIFFICULTY_TO_ID[row["meta"]["difficulty"]], dtype=torch.long
                        ),
                        "verifier_pattern_subtype": row["meta"]["pattern_subtype"],
                    }
                )
                self.samples.append(sample)

    def export_jsonl(self, path: str | Path, limit: Optional[int] = None) -> None:
        export_path = Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.rows if limit is None else self.rows[:limit]
        with export_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.samples[index]
