# 文本规则匹配验证器设计

## 1. 目标

本文档定义 `text rule verifier` 的职责、模型结构、训练目标与线上推理链路。

该模块不是通用风险分类器，而是一个更窄的条件验证器：

- 输入：`交易上下文 + 一条文本规则`
- 输出：`这笔交易是否满足这条规则`

更准确地说，它解决的是 `rule satisfaction / textual entailment over transaction context`，而不是“语义是否相似”。

---

## 2. 为什么需要 verifier

单独使用 dual encoder 做文本规则召回，只能回答：

- 这笔交易和这段描述是否语义接近

但业务真正需要的是：

- 这笔交易是否真的满足这段描述里的条件

例如规则：

- `卡国与 IP 国不一致，且用户近 3 天失败后换卡超过 2 次`

可能出现以下近似样本：

1. 只满足 `卡国与 IP 国不一致`
2. 满足失败后换卡，但次数只有 `2`
3. 满足条件但时间窗是 `7 天` 而不是 `3 天`

这些样本在 embedding 空间可能都很接近规则文本，但不应被视为“满足”。因此系统应分成两阶段：

1. 文本规则召回：找候选规则或候选交易
2. 匹配验证：验证候选 pair 是否真的满足

---

## 3. 系统职责

建议将文本规则系统拆成三层：

1. `Rule Retrieval`
   - dual encoder
   - 用于大规模候选召回

2. `Rule Verification`
   - cross-encoder / co-attention verifier
   - 用于条件满足验证

3. `Optional Symbolic Check`
   - 对可解析为确定逻辑的规则做符号化校验
   - 用于 hard reject 或高置信策略

其中本文档聚焦第二层。

---

## 4. 输入与输出定义

## 4.1 输入

验证器输入由三部分组成：

1. `transaction_context`
   - 当前 anchor 交易
   - 历史事件上下文
   - entity memory token
   - 关系特征

2. `rule_text`
   - 分析师写的一句文本规则
   - 例如：`卡国与 IP 国不一致，且用户近 3 天失败后换卡超过 2 次`

3. `rule_parse`（可选）
   - 规则拆解后的 clause 结构
   - 若解析失败，可为空

## 4.2 输出

建议第一版至少输出以下字段：

1. `p_match`
   - 整条规则是否满足

2. `p_uncertain`
   - 是否信息不足或规则过于模糊

3. `clause_scores`
   - 每个 clause 的满足概率

4. `evidence_scores`
   - 每条事件 / 实体 token 对结果的贡献

5. `symbolic_consistency`（可选）
   - 若规则被成功解析，则输出神经验证与符号验证的一致性

---

## 5. 规则表示

## 5.1 原始文本

保留分析师写的原始文本：

- `卡国与 IP 国不一致，且用户近 3 天失败后换卡超过 2 次`

## 5.2 Canonical Text

建议同时保留规范化版本，避免语言空间过于发散。例如：

- 原始文本：`疑似用户刷不过就换卡继续试`
- canonical：`用户近 3 天存在交易失败后换卡重试行为`

## 5.3 Clause 结构

建议把规则拆解为 clause 列表：

```json
{
  "logic": "AND",
  "clauses": [
    {
      "clause_id": "c1",
      "type": "country_mismatch",
      "text": "卡国与 IP 国不一致"
    },
    {
      "clause_id": "c2",
      "type": "switch_after_failure_count",
      "text": "用户近 3 天失败后换卡超过 2 次",
      "window": "3d",
      "operator": ">",
      "threshold": 2
    }
  ]
}
```

第一版即使无法完全自动解析自由文本，也可以要求规则入库时生成一版结构化 clause。

---

## 6. 模型结构

## 6.1 整体结构

推荐结构：

```text
Transaction Encoder -> H_tx
Rule Text Encoder   -> H_rule
Clause Encoder      -> H_clause (optional)
Cross Attention / Co-Attention
Heads:
  - overall match
  - uncertainty
  - clause match
  - evidence pointer
```

### 交易侧

建议直接复用现有 `Transaction Transformer` 主干：

- 输入：anchor + history events + entity memory + relation bias
- 输出：`H_tx ∈ R^{N_tx × d}`

其中 `N_tx` 可以包括：

- `N_event`
- `N_entity_memory`
- `N_global`

例如：

- `N_event = 128`
- `N_entity = 48`
- `N_global = 4`
- 总长度 `N_tx = 180`

### 文本侧

规则文本可用轻量 text encoder：

- BERT / RoBERTa 级别 encoder
- 或复用现成中文文本 encoder

输出：

- `H_rule ∈ R^{N_rule × d}`

其中 `N_rule` 一般较短，例如 `16-64 token`。

### 子句侧

若 clause 可获得，可额外编码成 clause token：

- `H_clause ∈ R^{N_clause × d}`

这样 head 可以直接在 clause 级别输出匹配结果。

---

## 6.2 推荐实现：Cross-Encoder Verifier

不建议只用全局 embedding 做 MLP matching。更推荐：

1. `H_tx = TxEncoder(context)`
2. `H_rule = RuleEncoder(rule_text)`
3. 让文本 token 对交易 token 做 cross attention
4. 用 pooled representation 输出 `p_match`

形式如下：

```text
H_cross = CrossAttn(query=H_rule, key=H_tx, value=H_tx)
h_pool  = Pool(H_cross)
p_match = Head_match(h_pool)
```

若有 clause：

```text
H_clause_cross = CrossAttn(query=H_clause, key=H_tx, value=H_tx)
p_clause_i = Head_clause_i(H_clause_cross[i])
```

这样模型可以把“失败后换卡”“设备保持不变”“卡国与 IP 国不一致”等文本短语，直接对齐到具体事件和实体。

---

## 6.3 备选实现：双塔向量 + 验证头

可做一个更便宜的 baseline：

```text
h_tx   = Pool(H_tx)
h_rule = Pool(H_rule)
z = [h_tx, h_rule, h_tx * h_rule, |h_tx - h_rule|]
p_match = MLP(z)
```

优点：

- 训练简单
- 推理便宜
- 可直接复用召回编码

缺点：

- 很难验证复杂逻辑条件
- clause 解释能力弱
- 对 hard negative 的区分通常不如 cross-encoder

建议用途：

- 作为 baseline
- 或者作为召回后的轻量过滤层

---

## 7. Evidence 建模

验证器除了输出 `p_match`，还应尽量指出证据。

建议两种方式并行：

1. `attention-based evidence`
   - 从 rule token 到 transaction token 的 cross attention 权重中取 top-k

2. `event scoring head`
   - 对每个 event token 单独预测 `is_evidence`

第二种更稳，因为 attention 权重不一定等于解释。

建议输出：

```json
{
  "evidence_events": [3, 7, 9],
  "evidence_entities": ["card_2", "device_1"]
}
```

---

## 8. Loss 设计

建议总 loss 由四部分组成：

```text
L = λ1 * L_match
  + λ2 * L_clause
  + λ3 * L_rank
  + λ4 * L_evidence
```

## 8.1 Overall Match Loss

- `L_match = BCE(p_match, y_match)`

这是核心目标。

## 8.2 Clause Loss

- 每个 clause 一个 BCE
- 若 clause 有明确数值，也可做辅助回归

例如：

- `country_mismatch`
- `switch_after_failure_count_gt_2`

## 8.3 Ranking Loss

针对 hard negative 做 margin ranking：

```text
L_rank = max(0, margin - s_pos + s_neg)
```

作用：

- 拉开“接近满足但未满足”的样本
- 防止模型退化成粗糙的风险分类器

## 8.4 Evidence Loss

如果样本里有证据事件标注：

- 对 evidence event 做 BCE
- 或 pointer loss

若没有显式证据标注，可先不启用此项。

---

## 9. 推理链路

线上建议使用如下链路：

1. `Rule Retrieval`
   - 文本规则 embedding 检索 top-K 候选规则

2. `Verifier`
   - 对 top-K 的 `transaction-context, rule-text` pair 做验证

3. `Optional Symbolic Check`
   - 对可解析规则补一层 deterministic check

4. `Decision Layer`
   - 根据 `p_match`、`p_uncertain`、`clause_scores`、`symbolic_consistency` 做最终决策

---

## 10. 线上输出与使用方式

建议先分三类场景：

### A. 检索型规则

- 用于召回可疑交易
- 只依赖 retrieval + verifier

### B. 软决策规则

- 用于加分、复审、3DS
- 依赖 verifier 输出，不直接 hard reject

### C. 硬规则

- 用于直接拒绝
- 要求：
  - 文本能解析出 clause
  - clause 可以验证
  - 最好有符号化 backing

---

## 11. 与符号系统的关系

Verifier 不应孤立存在，而应与现有特征/规则能力协同。

建议把可解析规则转换成统一中间表示 IR：

```json
{
  "logic": "AND",
  "clauses": [
    {"type": "country_mismatch", "left": "card_country", "right": "ip_country"},
    {"type": "distinct_count", "subject": "user_id", "object": "card_no", "window": "3d", "op": ">", "value": 5}
  ]
}
```

然后可以有三条执行路径：

1. 纯神经验证
2. 纯符号校验
3. 神经 + 符号融合

最终线上决策建议优先使用第三种。

---

## 12. 评估指标

Verifier 的评估不应只看 AUC，建议至少包括：

1. `overall_match_auc / auprc`
2. `hard_negative_accuracy`
3. `clause_accuracy`
4. `near-boundary ranking accuracy`
5. `evidence hit@k`
6. `uncertainty calibration`

还应单独做三类 slice：

1. 只差阈值
2. 只满足部分 clause
3. 文本改写但语义不变

---

## 13. 第一阶段实现建议

如果现在开始做，建议采用下面这版最小可行方案：

1. 复用现有 `Transaction Transformer` 作为交易 encoder
2. 增加一个轻量中文 rule text encoder
3. 在其上实现 cross-attention verifier
4. 输出：
   - `p_match`
   - `clause_scores`
   - `evidence_event_scores`
5. 训练时重点加入 hard negative ranking loss

第一版不要追求：

- 完全自由文本解析
- 直接替代所有 DSL 硬规则
- 完全开放式 reasoning

第一版最应验证的是：

- 模型是否能区分“语义接近但条件不满足”的样本
- clause 级监督是否有效
- verifier 是否明显优于仅用 embedding similarity

---

## 14. 总结

`text rule verifier` 的本质不是另一个风险分类器，而是一个专门验证“交易上下文是否蕴含这条文本规则”的模型。

它的价值在于：

- 把文本规则从“语义召回”提升到“条件验证”
- 为文本规则走向真实线上策略提供可控中间层
- 让神经网络与符号规则形成协同，而不是相互替代
