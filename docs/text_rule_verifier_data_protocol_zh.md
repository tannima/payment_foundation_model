# 文本规则匹配验证器样本协议

## 1. 目标

本文档定义 `text rule verifier` 的训练样本协议，重点解决四件事：

1. 一条文本规则如何表示
2. 一笔交易上下文如何与规则组成训练样本
3. 正样本、负样本和 hard negative 如何构造
4. 如何用 synthetic 数据补足边界条件与组合模式

目标是让模型学到“条件满足”，而不是只学到“语义近似”。

---

## 2. 样本单位

一个训练样本对应一个 `(transaction_context, rule_text)` pair。

推荐结构如下：

```json
{
  "sample_id": "verifier_000001",
  "transaction_context": {
    "anchor_event": {},
    "history_events": [],
    "entity_memories": []
  },
  "rule": {
    "rule_id": "rule_001",
    "raw_text": "卡国与 IP 国不一致，且用户近 3 天失败后换卡超过 2 次",
    "canonical_text": "卡国与 IP 国不一致，且用户近 3 天存在失败后换卡重试行为",
    "logic": "AND",
    "clauses": []
  },
  "labels": {
    "overall_match": 1,
    "uncertain": 0,
    "clause_labels": {},
    "evidence_event_indices": [3, 7],
    "evidence_entity_ids": ["card_2", "device_1"]
  },
  "meta": {
    "source": "rule_engine",
    "domain": "AE",
    "difficulty": "hard_negative"
  }
}
```

---

## 3. 规则表示协议

## 3.1 Rule Object

每条规则建议保留以下字段：

- `rule_id`
- `raw_text`
- `canonical_text`
- `logic`
- `clauses`
- `tags`
- `language`
- `rule_source`

示例：

```json
{
  "rule_id": "rule_country_switch_3d_v1",
  "raw_text": "卡国和 IP 国不一致，而且用户最近 3 天失败后频繁换卡",
  "canonical_text": "卡国与 IP 国不一致，且用户近 3 天失败后换卡超过 2 次",
  "logic": "AND",
  "tags": ["country_mismatch", "switch_after_failure", "3d"],
  "language": "zh",
  "rule_source": "analyst"
}
```

## 3.2 Clause Object

建议 clause 保留如下字段：

- `clause_id`
- `type`
- `text`
- `subject`
- `object`
- `filter`
- `window`
- `operator`
- `threshold`
- `polarity`

示例：

```json
[
  {
    "clause_id": "c1",
    "type": "country_mismatch",
    "text": "卡国与 IP 国不一致",
    "subject": "card_country",
    "object": "ip_country",
    "operator": "!="
  },
  {
    "clause_id": "c2",
    "type": "switch_after_failure_count",
    "text": "用户近 3 天失败后换卡超过 2 次",
    "subject": "user_id",
    "object": "card_no",
    "filter": "after_failed_payment",
    "window": "3d",
    "operator": ">",
    "threshold": 2
  }
]
```

---

## 4. 标签协议

建议标签至少包括三层：

## 4.1 Overall Match

- `overall_match ∈ {0,1}`

代表整条规则是否满足。

## 4.2 Clause Labels

对于每个 clause，提供：

- `clause_labels[clause_id] ∈ {0,1}`

如果存在文本模糊或数据不足，还可扩展为：

- `0 = not match`
- `1 = match`
- `2 = unknown`

## 4.3 Evidence Labels

如果样本能提供证据，建议标：

- `evidence_event_indices`
- `evidence_entity_ids`

第一版即使没有人工证据标注，也可以由规则引擎自动导出候选证据。

---

## 5. 正样本构造

正样本应来自“规则真实满足”的交易上下文。建议有三种主来源。

## 5.1 现有规则 / 特征平台自动标注

最可靠。

示例：

- DSL：`ip_country != card_country and user_distinct_card_3d > 5`
- 自动生成文本：`卡国与 IP 国不一致，且用户近 3 天更换多张卡`
- 满足 DSL 的历史样本标为 `overall_match=1`

优点：

- 标签稳定
- 可以大规模生产
- clause 真值可直接由现有引擎导出

## 5.2 基于结果链和行为模式自动生成

例如：

- 交易失败后换卡重试
- 风控未拒绝，银行通过，支付成功，后续报回欺诈
- 同设备关联多张卡

这类规则可由历史事件序列自动检测并生成正样本。

## 5.3 分析师案例 / 调查样本

价值最高，但规模较小。

用途：

- 补真实自然语言表达
- 补复杂模式
- 补现有规则难以覆盖的风险语义

---

## 6. 负样本构造

不能只随机采样负样本，否则模型会学成“风险 vs 非风险”。

建议负样本分三层。

## 6.1 Easy Negatives

完全不相关样本。

例如规则：

- `卡国与 IP 国不一致，且失败后换卡超过 2 次`

easy negative：

- 卡国与 IP 国一致
- 没有失败
- 没有换卡

作用：

- 快速建立基本分界

## 6.2 Semantic Negatives

语义接近但不满足。

例如：

- 有 country mismatch，但没有换卡
- 有换卡，但不是失败后换卡
- 有失败后换卡，但时间窗是 `7d` 不是 `3d`

作用：

- 防止模型只看局部关键词

## 6.3 Hard Negatives

最重要的一类。特征上很像，条件上只差一点。

例如：

- 次数阈值差 `1`
- 满足两个 clause 之一
- 行为模式相似，但逻辑因果不同
- 文本改写后引入边界歧义

这类样本最能训练 verifier 学习“条件满足”。

---

## 7. Hard Negative 构造策略

建议围绕以下五类构造。

## 7.1 阈值边界

规则：

- `近 3 天失败后换卡超过 2 次`

hard negatives：

- 恰好 `2` 次
- `1` 次
- `3` 次但发生在 `4` 天前

## 7.2 只满足部分 clause

规则：

- `A and B`

hard negatives：

- 只满足 `A`
- 只满足 `B`

## 7.3 逻辑关系干扰

规则：

- `失败后换卡`

hard negatives：

- 先换卡后失败
- 失败后换设备而不是换卡
- 多次失败与多张卡同时存在，但没有顺序关系

## 7.4 正常业务相似模式

例如：

- 正常旅行导致 `IP_country != card_country`
- 合法补卡导致短期换卡
- 家庭设备共享导致同设备多卡

这些是业务上非常关键的 hard negatives。

## 7.5 语言近义干扰

例如同一交易匹配：

- `设备不稳定`
- `支付工具频繁切换`

但目标规则是：

- `失败后换卡`

即语义相邻，但条件不同。

---

## 8. Clause 标签生成

为了让 verifier 不只是学整体匹配，建议 clause 标签尽量自动化生成。

规则生成流程：

1. 从 DSL / 规则模板得到 clause 定义
2. 对每条样本离线计算 clause 真值
3. 输出到 `clause_labels`

示例：

```json
{
  "clause_labels": {
    "c1": 1,
    "c2": 0
  },
  "overall_match": 0
}
```

对于 `AND` 规则：

- `overall_match = min(clause_labels)`

对于 `OR` 规则：

- `overall_match = max(clause_labels)`

对于复杂规则：

- 根据解析出的逻辑树执行

---

## 9. Evidence 标注策略

Evidence 不要求一开始全人工标注，可分三层获得：

1. `programmatic evidence`
   - 由 DSL 规则执行时产出的命中事件

2. `heuristic evidence`
   - 由局部窗口检索时的关键事件近似标注

3. `human evidence`
   - 分析师明确指出的支持事件

第一版建议优先使用第 1 类和第 2 类。

---

## 10. Synthetic 数据构造

Synthetic 数据很适合 verifier，因为它可以系统性覆盖边界条件和逻辑扰动。

## 10.1 适合 synthetic 的部分

- 原子 clause
- clause 组合
- 阈值边界
- 时间窗边界
- 正常相似行为 vs 攻击行为
- 失败后换卡 / 换设备 / 换邮箱 / 换地址的各种排列

## 10.2 不适合完全依赖 synthetic 的部分

- 分析师真实语言风格
- 模糊表达和隐喻表达
- 跨业务真实噪声

因此 synthetic 应作为：

- warm-up 数据
- hard negative 数据
- 组合泛化数据

而不是唯一数据来源。

## 10.3 推荐的 synthetic 生成层次

### 层 1：结构模式

直接控制：

- 事件数
- 时间间隔
- entity switch 次数
- 决策链

### 层 2：规则文本模板

例如：

- `卡国与 IP 国不一致`
- `用户近 {window} 失败后换卡超过 {k} 次`
- `同设备近 {window} 关联 {k}+ 张不同卡`

### 层 3：文本改写

例如：

- `刷不过就换卡再试`
- `同一设备近期挂了很多卡`
- `在异地 IP 下反复改卡尝试`

---

## 11. 文本来源分层

建议训练文本同时包含三层：

1. `canonical text`
   - 高精度
   - 由规则自动生成

2. `paraphrase text`
   - 同义改写
   - 由模板或 LLM 改写

3. `real analyst text`
   - 工单、调查备注、案例摘要

三层缺一不可：

- 只做 canonical，模型只会模板匹配
- 只做 analyst text，噪声太大

---

## 12. 训练采样策略

建议 mini-batch 内混合以下样本：

1. 正样本
2. easy negatives
3. semantic negatives
4. hard negatives

推荐一个起步比例：

- `positive : easy : semantic : hard = 1 : 1 : 1 : 2`

原因：

- verifier 最缺的是 hard negative 辨别能力

另外建议：

- 同一个 `transaction_context` 配多个不同 rule
- 同一个 `rule_text` 配多个交易上下文

这样才能真正学 pairwise verification。

---

## 13. 数据切分原则

必须避免以下泄漏：

1. 同一交易同时出现在 train / val / test
2. 同一条合成模板只在训练出现，验证全是同模板改写
3. 同一业务 case 的近似重复样本跨集合出现

建议切分维度包括：

- 时间切分
- 规则切分
- paraphrase 切分
- 模式切分

还应单独构造 `unseen composition test`：

- 训练只见过 `A`、`B`
- 测试看 `A and B`

---

## 14. 评估协议

除了整体匹配指标，还建议单独看：

1. `overall_match_auprc`
2. `hard_negative_accuracy`
3. `clause_f1`
4. `near_boundary_ranking_acc`
5. `evidence_hit@k`
6. `paraphrase_consistency`
7. `unseen_composition_generalization`

必须单独做的 slice：

1. `threshold boundary`
2. `partial clause satisfied`
3. `benign look-alike`
4. `language paraphrase`

---

## 15. 第一阶段推荐数据规模

如果做 PoC，建议规模如下：

### 规则数量

- `1k - 5k` 条 canonical rule

### 文本变体

- 每条 `5 - 20` 个 paraphrase

### 训练 pair

- `100万 - 500万` 个 `(transaction_context, rule_text)` pair

### 其中 hard negatives

- 至少占 `30% - 40%`

若进入第一版可用阶段，可扩展到：

- `1万 - 5万` 条规则
- `1000万 - 5000万` pair

---

## 16. 推荐落地顺序

建议按以下顺序推进：

1. 从现有 DSL / velocity 规则自动生成 canonical rule 和 clause 标签
2. 先构建 verifier 训练样本协议
3. 用 synthetic 补足边界条件和逻辑扰动
4. 引入 paraphrase 扩展语言泛化
5. 最后再接真实分析师文本

第一阶段最应验证的是：

- 模型是否能在 hard negative 上显著优于 embedding-only baseline
- clause 监督是否能提升条件验证能力
- synthetic + canonical 数据是否足以启动语言对齐

---

## 17. 总结

`text rule verifier` 的数据协议设计，关键不在于“多造一些文本”，而在于：

- 明确规则的 clause 结构
- 大量构造 hard negatives
- 把“是否满足条件”而不是“是否像风险”作为监督目标

如果这三点做对，verifier 才会真正成为“文本规则的匹配验证系统”，而不是另一个泛化模糊的风险分类器。
