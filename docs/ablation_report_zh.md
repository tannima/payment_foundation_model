# synthetic data 消融实验记录

## 1. 实验目的

本轮实验的目标是验证两个新增结构是否在更难的 synthetic data 上带来收益：

1. `entity memory token`
2. `typed attention bias`

同时回答两个问题：

1. 更难的 synthetic data 是否比原先更适合作为结构消融基准
2. 当前实现下，这两个结构是否已经带来可测收益

---

## 2. synthetic data 改动

这一轮不再只生成简单的“正常用户 vs 盗卡用户”两类样本，而是加入了更多模式：

### Benign

- `benign_stable`
- `benign_card_refresh`
- `benign_travel_burst`
- `benign_shared_device`

其中后三类属于 `hard negatives`，因为它们包含部分高风险表象：

- 合法换卡
- 合法短时 burst
- 合法共享设备

### Suspicious

- `suspicious_card_theft`
- `suspicious_ato`
- `suspicious_device_ring`

这些模式分别强调：

- 换卡 + 换邮箱地址 + 高频大额
- 同卡但设备 / 邮箱 / 地址切换
- 同设备多卡轮转

---

## 3. 评估指标

本轮重点看五类指标：

1. `fraud_recall_at_top10pct`
2. `hard_negative_fp_rate`
3. `switch_attack_recall_at_0.5`
4. `result_sequence_accuracy`
5. `mse_same_device_distinct_card`

它们分别衡量：

- 高风险召回
- 对 hard negatives 的误报控制
- 对典型切换攻击的识别
- 自回归结果链学习效果
- 关系型 atomic target 学习效果

---

## 4. 实验配置

消融脚本：

- `scripts/run_ablation.py`

本轮比较四组配置：

1. `base`
2. `memory_only`
3. `bias_only`
4. `memory_plus_bias`

训练设置：

- train size: `2400`
- val size: `480`
- epochs: `2`
- `d_model=96`
- `layers=3`
- hard synthetic enabled

结果文件：

- `artifacts/ablations/ablation_summary.json`

---

## 5. 实验结果

### base

- `fraud_recall_at_top10pct = 0.2192`
- `hard_negative_fp_rate = 0.0000`
- `switch_attack_recall_at_0.5 = 0.0000`
- `result_sequence_accuracy = 0.3604`
- `mse_same_device_distinct_card = 0.1522`

### memory_only

- `fraud_recall_at_top10pct = 0.2192`
- `hard_negative_fp_rate = 0.0000`
- `switch_attack_recall_at_0.5 = 0.0000`
- `result_sequence_accuracy = 0.3604`
- `mse_same_device_distinct_card = 0.1578`

### bias_only

- `fraud_recall_at_top10pct = 0.2192`
- `hard_negative_fp_rate = 0.0000`
- `switch_attack_recall_at_0.5 = 0.0000`
- `result_sequence_accuracy = 0.3604`
- `mse_same_device_distinct_card = 0.1612`

### memory_plus_bias

- `fraud_recall_at_top10pct = 0.2192`
- `hard_negative_fp_rate = 0.0000`
- `switch_attack_recall_at_0.5 = 0.0000`
- `result_sequence_accuracy = 0.3604`
- `mse_same_device_distinct_card = 0.1603`

---

## 6. 结论

本轮实验的主要结论是：

1. 当前 synthetic data 已经比之前更复杂，但评估仍然不够敏感
2. 在当前实现和训练预算下，`entity memory token` 与 `typed attention bias` 还没有带来明确的下游收益
3. 少数原子任务指标存在波动，但不稳定，也未转化为风险识别指标上的一致提升

也就是说：

- 新结构已经接入并可训练
- 但当前还不能证明它们有价值

这不是坏结果，反而说明实验设计进入了可证伪状态。

---

## 7. 为什么暂时看不到收益

可能原因有四类：

1. 当前 synthetic data 仍可被较浅层模式解决，关系建模价值没有被真正逼出来
2. 训练轮数和样本规模偏小，结构收益还未显现
3. `fraud_label` 和 slice 任务仍然不够依赖复杂实体关系
4. 当前 `entity memory` 和 `relation bias` 只是第一版实现，归纳偏置还不够强

---

## 8. 下一轮建议

建议按以下顺序继续推进：

1. 继续增强 hard negatives，使“合法换卡 / 合法 burst / 合法共享设备”更像攻击样本
2. 增加更关系敏感的 atomic targets，例如：
   - same-email distinct card count
   - same-ip distinct card count
   - recent new-entity switch count
3. 将 `switch_attack_recall` 从固定阈值改成排序指标
4. 加大训练轮数和样本规模，再重新跑消融
5. 将 `entity memory` 与 event-memory membership bias 进一步做强

---

## 9. 结论摘要

这一轮不是“新结构有效”的结论，而是：

- 新结构已成功接入
- synthetic data 和消融流程已建立
- 当前实验还不足以证明收益

这是一个正常且有价值的阶段性结果，说明后续优化应该优先加强：

- 数据难度
- 指标敏感度
- 结构归纳偏置
