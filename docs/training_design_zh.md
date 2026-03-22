# 信用卡支付风险基座模型训练设计

## 1. 目标

本设计文档聚焦训练体系，回答四个问题：

1. 一笔样本如何组织成模型输入
2. 基座模型预训练到底学什么
3. 后训练如何适配不同业务域
4. 第一阶段实验应如何验证方案有效性

目标不是直接定义最终 7B 全量实现，而是先给出一条可渐进扩展的训练路线。

---

## 2. 训练样本单位

### 2.1 样本定义

一个训练样本对应一笔 anchor 交易，以及在该交易发生之前、通过多实体检索召回并组装的历史上下文。

样本由三部分组成：

1. `anchor_event`
2. `history_events`
3. `targets`

其中：

- `anchor_event` 是当前待判断交易
- `history_events` 是该时点之前可见的上下文事件
- `targets` 包含预训练目标和后训练标签

### 2.2 时间约束

所有训练样本必须严格满足因果约束：

- 历史上下文只能包含 anchor 之前发生的事件
- 任何未来事件不可进入输入
- 事后标签如 `is_fraud/is_cb` 可以作为监督信号，但不能作为输入条件

---

## 3. 输入组织

### 3.1 标准字段

建议按统一 schema 组织每个事件，主字段包括：

- 时间：`gmt_occur`
- 金额：`event_amount`
- 用户：`user_id`
- 卡：`card_no`、`card_bin`、`issuer`、`card_country`
- 设备网络：`device_id`、`ip_address`、`ip_country`
- 账单：`bill_email`、`bill_address`
- 商品：`item_name`、`item_category`
- 业务域：`business_domain`
- 结果链：`is_reject`、`is_3ds`、`is_approve`、`is_success`、`is_fraud`、`is_cb`

### 3.2 样本内部局部实体编号

对高基数实体在单样本内重编号：

- `user_local_id`
- `card_local_id`
- `device_local_id`
- `email_local_id`
- `address_local_id`
- `ip_local_id`

作用：

- 显式保留“是否同一实体”
- 支持模型学习 `count distinct`
- 避免高基数词表问题

### 3.3 事件附加特征

建议为每个事件附加：

- 相对 anchor 时间差
- 与 anchor 是否同卡 / 同设备 / 同地址 / 同 IP
- 字段缺失掩码
- 是否首次出现某实体
- 是否是实体切换点

---

## 4. 预训练任务设计

预训练建议拆成三大类核心任务和一类长期任务。

### 4.1 风险原子任务

目标是显式学习过去 velocity 特征表达的核心统计能力。

建议覆盖：

- `count`
- `count distinct`
- `sum(amount)`
- `max(amount)`
- `avg(amount)`
- `time since last occurrence`
- `new entity appeared`
- `entity switch happened`
- `burst score in 1h/1d/7d`

标签生成方式：

- 从同一份上下文中，按照 `subject/object/filter/window/function` 自动离线计算

训练形式：

- 回归
- 分桶分类
- Pairwise ranking

建议不要直接把所有 velocity 配置原样映射成海量头，而应先抽象为原子算子，再用模板自动生成训练目标。

### 4.2 事件结果自回归预测

这是支付基座模型中最重要的过程建模任务之一。

目标不是把 `is_reject/is_approve/is_success/is_fraud/is_cb` 当作并列标签独立预测，而是学习支付流程中的因果决策链。

支付流程天然具有前后依赖关系：

1. 风控决策先发生
2. 只有风控不拒绝，银行决策才会发生
3. 只有风控和银行都未拒绝，支付成功结果才会发生
4. 只有支付成功，事后欺诈和拒付标签才有实际意义

因此建议设计一个结果序列：

- `<risk_decision>`
- `<bank_decision>`
- `<payment_outcome>`
- `<post_event_risk>`

例如可展开为：

- `reject=no`
- `3ds=yes`
- `approve=yes`
- `success=yes`
- `fraud=yes`
- `cb=no`

训练方式：

- 先编码结构化事件上下文
- 在其后拼接结果链 token 序列
- 使用 teacher forcing 进行自回归训练

这样做的好处：

- 条件依赖天然由自回归概率分解表达
- 能正确表示“前一步不发生，后一步就不存在”的业务因果结构
- 更适合未来扩展为更长的支付过程建模

建议概率分解写成：

```text
P(result_chain | context)
= P(risk_decision | context)
* P(bank_decision | context, risk_decision)
* P(payment_outcome | context, risk_decision, bank_decision)
* P(post_risk | context, risk_decision, bank_decision, payment_outcome)
```

这一步比并行多头更符合业务本质。

### 4.3 去噪与缺失鲁棒性任务

为了支持多业务迁移，建议加入：

- mask 字段恢复
- 删除部分事件后恢复行为关系
- 模拟 AE 有、其他域缺失的字段
- 跨域相似模式对比学习

目标是让 backbone 学会在字段缺失情况下仍然保持风险推理能力。

### 4.4 语言对齐任务

长期可引入结构化输入到自然语言问答的对齐任务，例如：

- “是否存在躲闪行为？”
- “风险原因是什么？”
- “为什么建议触发 3DS？”

当前阶段不作为优先项，建议放在原子任务和自回归决策链任务验证成功之后。

---

## 5. 预训练 Loss 组合

总损失建议写为：

```text
L = λ1 * L_atomic
  + λ2 * L_result_ar
  + λ3 * L_denoise
  + λ4 * L_domain_alignment
```

其中：

- `L_atomic`：风险原子任务 loss
- `L_result_ar`：事件结果链自回归 loss
- `L_denoise`：去噪和缺失鲁棒性 loss
- `L_domain_alignment`：跨域对齐或对比学习 loss

首版建议：

- `λ1 = 0.40`
- `λ2 = 0.35`
- `λ3 = 0.15`
- `λ4 = 0.10`

后续根据验证结果调整。

---

## 6. 后训练设计

后训练目标是适配不同业务域自己的标签定义和风险水位。

建议结构：

- 共享 foundation backbone
- 各业务独立 prediction head
- 各业务可拼接原有域内特征

输入：

- foundation embedding
- 业务独有实时特征
- 域内标签

输出：

- 各业务独立风险分
- 各业务独立 calibration

这种方式比统一大二分类头更稳妥，因为：

- 不同业务标签定义不同
- 不同业务可接受风险水位不同
- 现有业务资产可以逐步融合，而不是被迫整体替换

---

## 7. 第一阶段实验设计

### 7.1 实验目标

先验证三件事：

1. 原始事件上下文输入是否能学到有效风险模式
2. 模型是否能学会 `count distinct / switch / burst`
3. 自回归结果链任务是否优于并行多头预测

### 7.2 Baseline

建议至少做三类对比：

1. `XGB / MLP + 手工聚合特征`
2. `Transformer + 原始事件序列 + 并行结果头`
3. `Transformer + 原始事件序列 + 自回归结果链`

### 7.3 关键 Ablation

建议重点比较：

1. 去掉局部实体编号
2. 去掉 relation-to-anchor 特征
3. 去掉风险原子任务
4. 去掉自回归结果链，只保留并行结果预测
5. 单域训练 vs 多域预训练

### 7.4 评估指标

分类类：

- Recall@TopK
- Recall@Top1%
- PR-AUC
- FPR at fixed recall

原子任务类：

- MSE
- Pearson correlation
- 排序一致性

过程建模类：

- 结果链 token-level perplexity
- 链级别完整准确率
- 条件决策一致率

其中“条件决策一致率”建议单独统计，例如：

- 模型是否错误预测了一个本不该在当前条件下出现的后续结果

---

## 8. 首版工程落地建议

第一阶段训练原型建议保持克制：

1. 先做 `100M~300M` backbone
2. 先接 1 到 2 个主业务域
3. 先用统一 schema + 检索上下文输入
4. 先做三类任务：
   - 风险原子任务
   - 结果链自回归任务
   - 业务域后训练分类

不建议第一阶段就做：

- 7B 线上模型
- 完整语言任务
- 极复杂图检索结构

---

## 9. 结论

训练体系的核心转变是：

- 从“学习人工聚合后的风险特征”转向“学习风险行为过程”
- 从“单一二分类监督”转向“原子统计能力 + 自回归支付过程 + 多域后训练”

其中最关键的设计点有两个：

1. 用原始上下文事件取代手工 velocity 特征
2. 用自回归结果链取代并行多头结果预测

第二点尤其重要，因为支付流程天然存在因果顺序，只有自回归建模才能自然表达：

- 哪些决策先发生
- 哪些结果依赖前置结果
- 哪些标签只在特定条件下才有意义

这会使基座模型不仅能判断“风险高不高”，还能够学习“这个支付过程是如何一步步走到当前结果的”。
