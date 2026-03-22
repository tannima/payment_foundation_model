# 原型实现逻辑说明

## 1. 这版原型实现了什么

当前代码实现的是一个最小可运行版本，目标不是完整复现最终基座模型，而是把以下关键链路先跑通：

1. 生成结构化 synthetic transaction context
2. 将原始事件序列编码为事件级表示
3. 通过 Transformer encoder 学上下文风险表征
4. 通过自回归 decoder 学支付结果链
5. 同时学习若干 atomic risk targets
6. 导出 synthetic 数据文件，便于直接查看样本

---

## 2. synthetic 数据文件在哪里

训练脚本加上 `--export-jsonl` 后，会生成：

- `artifacts/synthetic_train.jsonl`
- `artifacts/synthetic_val.jsonl`

每一行都是一个完整样本，包含：

- 业务域
- 模式类型：`benign` 或 `suspicious`
- `fraud_label`
- `result_chain`
- `outcome_details`
- `events`

这类文件不是训练时直接读取的最终张量，而是训练前的可读原始样本表示。

---

## 3. 一条样本长什么样

以 `artifacts/synthetic_train.jsonl` 中的样本为例，一个高风险样本大致如下：

```json
{
  "sample_id": 1,
  "domain": "Antom",
  "pattern_type": "suspicious",
  "fraud_label": 1,
  "result_chain": [
    "<bos>",
    "risk_allow",
    "3ds_no",
    "bank_approve",
    "pay_success",
    "fraud_no",
    "cb_no",
    "<eos>"
  ],
  "events": [
    {
      "timestamp": "2025-01-15T14:18:43",
      "amount": 18.71,
      "card_no": "old_card_hash",
      "bill_email": "old_email"
    },
    {
      "timestamp": "2026-03-20T03:18:43",
      "amount": 329.0,
      "card_no": "new_card_hash",
      "bill_email": "new_email"
    },
    {
      "timestamp": "2026-03-22T07:18:43",
      "amount": 129.0,
      "card_no": "new_card_hash",
      "bill_email": "new_email"
    }
  ]
}
```

这类样本表达的是：

- 早期是低频低额小额交易
- 后期突然切换卡、邮箱、地址
- 短时间内高频大额交易
- 再配上支付链结果 token 序列

这里最重要的不是 token 文本本身，而是：

- 原始事件序列
- 同实体关系
- 时间差
- 决策链顺序

---

## 4. synthetic 数据如何生成

相关代码在：

- `src/payment_foundation_model/synthetic_data.py`

核心逻辑分为四层。

### 4.1 生成事件序列

Benign 样本：

- 长时间跨度内低频交易
- 金额较小
- 卡、设备、邮箱切换较少

Suspicious 样本：

- 早期是旧卡、小额、低频
- 后期切换到新卡、新邮箱、新地址
- 从 `2026-03-20` 左右开始短时高频大额交易

这部分在：

- `_generate_benign_events`
- `_generate_suspicious_events`

### 4.2 生成支付结果链

结果链由 `_build_result_chain` 生成。

它不再把结果字段视为并列标签，而是构造成因果顺序的 token 序列，例如：

- `<bos> risk_reject <eos>`
- `<bos> risk_allow 3ds_no bank_decline <eos>`
- `<bos> risk_allow 3ds_yes bank_approve pay_success fraud_yes cb_no <eos>`

这里的关键点是：

- 如果风控拒绝，链路直接结束
- 如果银行拒绝，支付和事后标签不会出现
- 只有支付成功，才会出现 `fraud/cb` 结果

这正是你要求的自回归条件建模。

### 4.3 生成局部实体编号

在 `_build_sample_tensors` 中，会对每个样本内部出现的实体做重编号：

- `user_local_id`
- `card_local_id`
- `device_local_id`
- `email_local_id`
- `address_local_id`
- `ip_local_id`

例如某个样本里只出现两张卡，则会被映射为：

- 第一张卡 -> `1`
- 第二张卡 -> `2`

这一步的作用是让模型显式理解：

- 哪些事件是同一张卡
- 一共换了几张卡
- 哪个实体是新出现的

### 4.4 构造成训练张量

最终每个样本会被转换为三类输入：

1. 事件序列张量
2. atomic supervision
3. result chain decoder 张量

其中 decoder 相关张量为：

- `result_input_ids`
- `result_target_ids`
- `result_mask`

如果结果链是：

```text
<bos> risk_allow 3ds_no bank_approve <eos>
```

那么：

- `result_input_ids = [<bos>, risk_allow, 3ds_no, bank_approve]`
- `result_target_ids = [risk_allow, 3ds_no, bank_approve, <eos>]`

这就是标准 teacher forcing 的自回归训练格式。

---

## 5. 模型结构如何实现

相关代码在：

- `src/payment_foundation_model/model.py`

这版模型是一个最小 `encoder + autoregressive decoder` 架构。

### 5.1 Event Encoder

输入部分把每个事件拆成多种 typed feature：

- 局部实体 id embedding
- 业务域 embedding
- 国家 / issuer / item category embedding
- 金额和相对时间数值
- relation-to-anchor flags

然后拼接后投影到统一维度 `d_model`。

即：

```text
event_features
-> concat(typed embeddings + numeric + relation flags)
-> linear projection
-> event token
```

再加上 event position embedding 和 `CLS` token，送入 Transformer encoder。

### 5.2 Encoder 输出什么

Encoder 输出两类东西：

1. 全部 event memory
2. `CLS` pooled 向量

用途分别是：

- 全部 event memory：给结果链 decoder 做 cross attention
- `CLS` pooled：给 fraud head 和 atomic heads 做预测

### 5.3 Result Decoder

结果链 decoder 使用 `nn.TransformerDecoder` 实现。

输入是：

- 上一步的结果 token
- encoder memory

并使用 causal mask 保证只能看见过去 token。

这样概率分解就是：

```text
P(result_chain | context)
= Π P(token_t | context, token_<t)
```

这正是支付结果链需要的建模方式。

### 5.4 多任务输出

当前模型同时输出：

- `fraud_logit`
- `tx_count_pred`
- `distinct_card_pred`
- `amount_sum_pred`
- `result_logits`

对应三类学习目标：

1. 风险分类辅助头
2. atomic risk targets
3. 自回归结果链

---

## 6. 训练脚本如何工作

相关代码在：

- `scripts/train_synthetic.py`

训练流程如下：

1. 构建 synthetic train / val dataset
2. 可选导出 JSONL 原始样本
3. 构建 `TransactionTransformer`
4. 计算三类 loss
5. 每个 epoch 做验证并保存最优 checkpoint

当前总 loss 结构是：

```text
loss
= result_chain_ce
+ 0.35 * fraud_bce
+ 0.25 * (tx_count_mse + distinct_card_mse + amount_sum_mse)
```

这里的意图是：

- 自回归结果链作为过程建模主任务
- fraud head 作为一个可观测的风险分类辅助目标
- atomic targets 作为 velocity 替代能力的显式监督

---

## 7. 为什么这版实现是合理的第一步

它虽然还远不是最终基座模型，但已经把几个关键设计点落地了：

1. 输入是原始事件上下文，而不是手工 velocity 数值
2. 高基数实体通过局部实体编号进入模型
3. 结果链采用自回归而不是并行多头
4. 同时保留 atomic task，让模型显式学习统计能力

这使它成为一个合适的“最小可证伪原型”。

---

## 8. 这版还缺什么

与最终目标相比，这版还故意省掉了很多东西：

1. 还没有 entity memory token
2. 还没有 typed attention bias
3. 还没有真实在线 context retrieval
4. 还没有多哈希桶全局 ID 编码
5. 还没有真实业务数据和多域后训练
6. 还没有 language task

也就是说，这版重点是先验证：

- 方案链路能否工作
- 数据协议是否合理
- 自回归结果链是否容易接到事件 backbone 上

---

## 9. 下一步建议

如果继续往前推进，建议按这个顺序演进：

1. 在 synthetic data 中加入更难的 hard negatives
2. 加入实体 memory token
3. 加入 same-card / same-device typed attention bias
4. 把结果链从固定 token 集扩展到更真实的支付流程事件
5. 接真实离线样本和真实 context assembler

这样可以从可跑原型，逐步过渡到真正可评估的基座模型系统。
