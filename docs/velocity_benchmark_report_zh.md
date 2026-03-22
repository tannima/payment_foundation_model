# velocity benchmark 结果记录

## 1. 评估目标

为了判断当前原型是否已经学会 `velocity` 类特征，仅看训练过程中的 MSE 不够。

因此新增了独立 benchmark，针对原子风险目标输出：

- `mse_log`
- `mae_log`
- `pearson`
- `spearman`
- `pairwise_ranking_acc`
- `bucket_accuracy`
- `exact_match_accuracy` 或 `within_xx_pct_accuracy`

其中：

- count 类目标看 `exact_match_accuracy` 和 `off_by_one_accuracy`
- amount 类目标看 `bucket_accuracy` 和相对误差

---

## 2. benchmark 代码

脚本位置：

- `scripts/velocity_benchmark.py`

输入：

- 训练好的 checkpoint

输出：

- 一个独立 test split 上的 benchmark JSON 结果

---

## 3. 评估对象

本轮评估了两版模型：

1. `velocity_base`
2. `velocity_memory_bias`

评估文件：

- `artifacts/velocity_benchmark_base.json`
- `artifacts/velocity_benchmark_memory_bias.json`

---

## 4. 结果摘要

### 4.1 base

#### tx_count_1d

- `exact_match_accuracy = 0.7725`
- `off_by_one_accuracy = 1.0000`
- `pearson = 0.9404`
- `pairwise_ranking_acc = 0.9906`

#### distinct_card_count_7d

- `exact_match_accuracy = 0.7813`
- `off_by_one_accuracy = 1.0000`
- `pearson = 0.8988`
- `pairwise_ranking_acc = 0.9764`

#### same_device_distinct_card_7d

- `exact_match_accuracy = 0.6769`
- `off_by_one_accuracy = 0.9937`
- `pearson = 0.8015`
- `pairwise_ranking_acc = 0.9185`

#### amount_sum_7d

- `bucket_accuracy = 0.5337`
- `within_10pct_accuracy = 0.1519`
- `within_20pct_accuracy = 0.2819`
- `pearson = 0.8022`
- `pairwise_ranking_acc = 0.9073`

### 4.2 memory_plus_bias

#### tx_count_1d

- `exact_match_accuracy = 0.9406`
- `off_by_one_accuracy = 1.0000`
- `pearson = 0.9504`
- `pairwise_ranking_acc = 0.9978`

#### distinct_card_count_7d

- `exact_match_accuracy = 0.8850`
- `off_by_one_accuracy = 1.0000`
- `pearson = 0.9036`
- `pairwise_ranking_acc = 0.9765`

#### same_device_distinct_card_7d

- `exact_match_accuracy = 0.6825`
- `off_by_one_accuracy = 0.9962`
- `pearson = 0.8668`
- `pairwise_ranking_acc = 0.9367`

#### amount_sum_7d

- `bucket_accuracy = 0.5994`
- `within_10pct_accuracy = 0.1606`
- `within_20pct_accuracy = 0.3237`
- `pearson = 0.8011`
- `pairwise_ranking_acc = 0.8966`

---

## 5. 结论

当前结果可以支持三个判断：

1. 对 count 类 velocity，模型已经学得比较强
2. 对更关系敏感的 distinct-count 目标，模型有明显能力，但还不算完美
3. 对 amount-sum 这类连续累加目标，当前还明显不够好

因此，不能说“已经完美预测 velocity”。

更准确的说法是：

- `tx_count_1d`：接近很好，尤其 `memory_plus_bias` 版本已很强
- `distinct_card_count_7d`：较强，但未完全解决
- `same_device_distinct_card_7d`：中等偏强，仍有提升空间
- `amount_sum_7d`：仍明显不足

---

## 6. 对 entity memory 和 typed attention bias 的判断

在这一轮专门针对 velocity benchmark 的评估里，`memory_plus_bias` 相比 `base` 的收益是清晰可见的：

- `tx_count_1d` exact match 从 `0.7725` 提升到 `0.9406`
- `distinct_card_count_7d` exact match 从 `0.7813` 提升到 `0.8850`
- `same_device_distinct_card_7d` pearson 从 `0.8015` 提升到 `0.8668`
- `amount_sum_7d` bucket accuracy 从 `0.5337` 提升到 `0.5994`

这说明：

- 对原子 `velocity` 预测本身，`entity memory + typed attention bias` 是有帮助的
- 只是这种收益还没有稳定体现在之前那轮更偏下游风险分类的 synthetic ablation 上

也就是说，结构收益首先体现在“更会算关系和统计”，而不一定立刻体现在最终 fraud score 上。

---

## 7. 下一步建议

下一步建议重点做两件事：

1. 扩充 benchmark target 集合
   - same-email distinct card count
   - same-ip distinct card count
   - new entity switch count
   - recent burst count

2. 在真实业务离线样本上做同口径 benchmark
   - 直接将模型预测值和特征平台真值对比
   - 用真实 velocity 评估替代 synthetic proxy
