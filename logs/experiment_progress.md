# Ours-FD baseline 实验进展记录

更新日期：2026-08-31

## 1. 研究目标

将 arXiv:2402.18211v1《Catastrophic Overfitting: A Potential Blessing in
Disguise》提出的 Ours-FD 作为 baseline，扩展原论文 Table 2，用于后续 PAMI
投稿。

当前阶段的目标实验矩阵：

- 数据集：CIFAR-10（完成后扩展至 CIFAR-100）。
- 模型：CIFAR ResNet18。
- 训练半径：`εT = 8/12/16/32/48/64`（单位均为 `/255`）。
- 方法：Ours-FD，以 FGSM-MEP 为训练底座。
- 第一轮参数筛选使用 seed 0；正式投稿结果至少需要 3 seeds 并报告均值和标准差。

## 2. 方法定义

当前实现的 Ours-FD 不是纯 FGSM-MEP，而是以 FGSM-MEP 生成训练扰动，并加入
论文的特征差异约束：

```text
L = smooth_CE
    + 10 × MSE(logits_adv, logits_initial)
    + λFD × MSE(feature_adv[B], feature_initial[B])
```

正式论文配置中：

- `λFD = 200`。
- 特征节点为 ResNet18 的 B 节点（`layer1` 输出）。
- label smoothing 的真实类别概率为 0.6。
- FGSM-MEP 为每个训练样本保存历史扰动和历史动量。
- MEP 状态在 epoch 0/40/80 重置。
- 动量衰减为 0.3。
- 参考实现式全 batch L1 梯度归一化。
- 默认训练步长 `α = εT`。

纯 FGSM-MEP 对照的目标为：

```text
L_MEP = smooth_CE + 10 × MSE(logits_adv, logits_initial)
```

## 3. 通用训练设置

- 数据增强：输入范围 `[0,1]`，Pad 4、随机裁剪、随机水平翻转。
- Batch size：128。
- Optimizer：SGD。
- 初始学习率：0.1。
- Momentum：0.9。
- Weight decay：`5e-4`。
- 正式训练：110 epochs。
- 学习率在 epoch 100、105 乘以 0.1。
- 第一轮正式 sweep：seed 0，`deterministic=false`（贴近参考实现）。
- Ours-FD checkpoint：按逐 epoch PGD-10 准确率保存 `best.pt`，并同时保留
  `final.pt` 和包含完整 MEP 状态的 `resume.pt`。

注意：当前实现使用 CIFAR-10 test loader 进行逐 epoch checkpoint 选择，这是为了
贴近论文/参考代码，但用于 PAMI 正式实验存在 test-set selection 风险。正式稿应考虑
固定训练集 validation split，或明确将该协议标为 faithful reproduction protocol。

## 4. 评估协议

### 4.1 原论文 Table 2 协议

- 训练半径 `εT` 随实验变化。
- 评估半径固定为 `εE = 16/255`。
- 攻击先针对无噪声确定性模型生成。
- 随后对 clean/adversarial input 加一次
  `U(-16/255, 16/255)`，并裁剪至 `[0,1]`。
- 这是论文的 non-adaptive attack-then-noise 协议，不使用 EOT。
- 指标：Clean、FGSM、PGD-10/20/50、C&W-20、APGD-T、完整 AutoAttack。
- PGD/C&W 步长：`2/255`，一次随机重启，执行完整迭代。

### 4.2 当前 matched-ε 诊断 sweep

- `εE = εT`。
- 无推理噪声。
- 指标：Clean、FGSM、PGD-10、PGD-50、C&W-20、AutoAttack。
- PGD/C&W 步长固定为 `2/255`。
- 使用各训练的 `best.pt`。

该协议用于观察模型随训练半径增加的真实无噪声鲁棒性和训练坍塌，不是严格意义上的
Table 2 扩展。PAMI 主表仍需把所有 checkpoint 统一补评为 `εE=16/255` 加论文噪声。

### 4.3 AutoAttack

- 使用 ConvergeSmooth/FGSM-PGI 参考仓库中的旧 AutoAttack 快照。
- 固定 SHA-256：
  `aeb3b5167a3e4971af0fb0192733cff9b8e5bba79ef5722dd1a1fe576db1afc0`。
- 已加入与 PyTorch 2.0.1 的兼容修复，但未修改攻击算法语义。

## 5. ε=12/255 论文结果复现

### 5.1 无推理噪声（Table 3 对照）

使用已验证的 Ours-FD + MEP checkpoint：

| Metric | Paper | Reproduced | Delta |
|---|---:|---:|---:|
| Clean | 74.08 | 71.41 | -2.67 |
| PGD-50 | 29.65 | 28.41 | -1.24 |
| C&W-20 | 25.32 | 24.18 | -1.14 |
| AutoAttack | 19.72 | 18.79 | -0.93 |

该次复现中额外得到：FGSM 38.43%，PGD-10 38.16%。

### 5.2 Table 2 non-adaptive noise 协议

| Metric | Paper | Reproduced | Delta |
|---|---:|---:|---:|
| Clean | 72.60 | 70.76 | -1.84 |
| FGSM | 49.08 | 38.66 | -10.42 |
| PGD-10 | 40.87 | 37.48 | -3.39 |
| PGD-20 | 33.06 | 29.87 | -3.19 |
| PGD-50 | 31.42 | 28.59 | -2.83 |
| C&W-20 | 28.16 | 24.25 | -3.91 |
| APGD-T | 23.33 | 21.21 | -2.12 |
| AutoAttack | 22.08 | 22.00 | -0.08 |

结论：AutoAttack 与论文几乎一致，PGD/C&W 相差约 2–4 个百分点，FGSM 差异较大。
整体上足以确认 Ours-FD + FGSM-MEP 的核心实现是可信的，但不能声称所有 Table 2
指标均已精确复现。

## 6. CIFAR-10 matched-ε 第一轮 sweep

共同设置：`λFD=200`、`α=εT`、110 epochs、seed 0、无推理噪声。下表的评估均使用
PGD-10 最佳 checkpoint。

### 6.1 已完成的正式攻击评估

| εT=εE | Best epoch | Best Clean | Best monitor PGD-10 | FGSM | Eval PGD-10 | PGD-50 | C&W-20 | AA |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 107 | 81.08 | 54.74 | 58.16 | 54.73 | 54.03 | 50.11 | 47.99 |
| 12 | 100 | 69.90 | 42.87 | 45.67 | 42.93 | 40.27 | 34.63 | 31.68 |
| 16 | 35 | 46.28 | 30.69 | 29.77 | 30.73 | 27.35 | 22.70 | 20.63 |

所有数值均为百分比。

### 6.2 best 与 final 的训练稳定性

| εT | Best epoch | Best Clean | Best PGD-10 | Final Clean | Final PGD-10 | 状态 |
|---:|---:|---:|---:|---:|---:|---|
| 8 | 107 | 81.08 | 54.74 | 81.04 | 54.52 | 稳定 |
| 12 | 100 | 69.90 | 42.87 | 85.19 | 10.80 | 后期发生 CO |
| 16 | 35 | 46.28 | 30.69 | 10.00 | 10.00 | epoch 35 后最终完全坍塌 |
| 32 | 109 | 10.00 | 10.00 | 10.00 | 10.00 | 从头到尾退化 |
| 48 | 109 | 10.00 | 10.00 | 10.00 | 10.00 | 从头到尾退化 |
| 64 | — | — | — | — | — | 尚未训练完成 |

ε=32 和 ε=48 的 matched-ε 正式攻击评估没有完成；对 clean accuracy 仅 10% 的恒定
分类器继续运行 AutoAttack 没有科学价值。ε=64 因顺序 manifest 被前面的评估阻塞而
未完成训练。

AutoAttack 对退化模型打印 `initial accuracy: 10.00%`、且 APGD 成功扰动数为 0，
不代表获得了有效的 10% 鲁棒准确率，而是模型接近恒定输出、仅随机命中一个类别。

## 7. 第一轮 matched-epsilon sweep 的阶段性结论

1. Ours-FD 的正确实现必须同时包含 FD feature MSE 和 FGSM-MEP 的 logit MSE；早期
   不含 logit MSE 的结果已经判定为错误配置，不用于论文结论。
2. 在 ε=8 下，固定论文参数训练稳定，matched-ε AA 达到 47.99%。
3. 第一轮 ε=12 sweep 中，best checkpoint 表现正常，但 final checkpoint 已发生明显
   CO；第 11 节使用的是另一条已核对配置的 ε=12 FD+MEP 训练轨迹，故两者不能混为
   同一 final 结果。
4. 在 ε=16 下，方法只能在早期 checkpoint 保持有效，最终完全坍塌。
5. 在冻结的原始配置下，`λFD=200, α=εT` 无法直接外推至 ε≥32；后续带
   `abort_on_nonfinite` 的审计确认其为数值失稳，而不是有效的 10% 模型。
6. best checkpoint 会掩盖 late-stage CO，因此正式报告必须同时展示 best/final 曲线，
   不能只报告最优 epoch。
7. 原先固定 `2/255` 的 PGD-10 monitor 对 ε=32/48/64 偏弱。高 ε 参数筛选改用
   `monitor_step_size=ε/4`；ε=32 pilot 使用 `8/255`。
8. 两次几乎相同的 ε=12 训练轨迹曾出现“稳定”和“后期 CO”分叉，说明该方法对 CUDA
   非确定性和随机轨迹敏感。正式投稿必须使用多 seeds，不能仅凭 seed 0 最佳结果。
9. 论文的随机推理噪声协议是 non-adaptive 的。PAMI 稿件除 faithful Table 2 外，
   应补充无噪声 AA 和 EOT/自适应噪声攻击，避免被质疑为随机预处理造成的虚假鲁棒性。

## 8. ε=32 第一轮稳定性 pilot

目的：判断大 ε 失败主要来自 FD 权重过大、训练步长过大，还是 FGSM-MEP 本身无法在
该半径学习。

缩放依据：若 feature MSE 近似随 ε² 增长，则从论文 ε=12 的权重 200 外推至 ε=32：

```text
200 × (12/32)^2 ≈ 28.1
```

因此第一轮选择 feature weight 25，并加入更小权重和纯 MEP 对照：

| Run | Objective | εT | Training α | FD weight | Monitor |
|---|---|---:|---:|---:|---|
| `pilot_fd_eps32_alpha32_fw25` | Ours-FD | 32 | 32 | 25 | PGD-10, step 8 |
| `pilot_fd_eps32_alpha16_fw25` | Ours-FD | 32 | 16 | 25 | PGD-10, step 8 |
| `pilot_fd_eps32_alpha16_fw10` | Ours-FD | 32 | 16 | 10 | PGD-10, step 8 |
| `pilot_mep_baseline_eps32_alpha16` | FGSM-MEP | 32 | 16 | 0 | PGD-10, step 8 |

共同设置：

- 40 epochs，seed 0。
- `deterministic=true`，减少高 ε 分叉造成的配置比较噪声。
- 完整 CIFAR-10 test monitor。
- `track_features=true`。
- 不运行 AutoAttack，只用于训练可行性筛选。
- 每 epoch 保存总 loss，并在 `loss_components.csv` 保存未加权 CE、logit MSE、
  feature MSE。
- 当前启动器使用物理 GPU 4/5/6：GPU 4 顺序运行第一组和纯 MEP 对照，GPU 5/6
  各运行一组。

结果筛选原则：首先排除 Clean/PGD 均约 10% 的退化模型，再按强 PGD monitor 表现、
训练稳定性和 loss 分项量级选择候选；不使用 AutoAttack 测试结果反向调参。

### 8.1 实验结果

四组训练均完成 40 epochs，但全部在 epoch 0 内产生 NaN，之后模型保持随机分类水平。
由于当时尚未启用 `abort_on_nonfinite`，无效训练继续到了 epoch 39；这些 checkpoint
不进入正式攻击评估。

| Run | Best epoch | Best Clean | Best PGD-10 | Final Clean | Final PGD-10 | Train loss |
|---|---:|---:|---:|---:|---:|---|
| `pilot_fd_eps32_alpha32_fw25` | 39 | 10.00 | 10.00 | 10.00 | 10.00 | epoch 0 起 NaN |
| `pilot_fd_eps32_alpha16_fw25` | 39 | 10.00 | 10.00 | 10.00 | 10.00 | epoch 0 起 NaN |
| `pilot_fd_eps32_alpha16_fw10` | 39 | 10.00 | 10.00 | 10.00 | 10.00 | epoch 0 起 NaN |
| `pilot_mep_baseline_eps32_alpha16` | 39 | 10.00 | 10.00 | 10.00 | 10.00 | epoch 0 起 NaN |

Ours-FD 三组的 CE、logit MSE、feature MSE 均为 NaN；纯 MEP 对照的 CE 和 logit
MSE 同样为 NaN，feature MSE 按定义为 0。由此可得：

1. 这不是普通的低准确率或 AutoAttack 异常，而是训练过程的数值故障。
2. 将 `λFD` 从 200 降至 25/10 仍不能避免 NaN。
3. 将训练步长从 32/255 降至 16/255 仍不能避免 NaN。
4. 纯 MEP 对照也失败，说明问题不能仅归因于 FD feature 项。
5. 在找到首个非有限 batch 的来源前，不应继续跑 110 epochs 或 AutoAttack。

## 9. ε=32 非有限值诊断

### 9.1 目的与约束

该实验只用于定位 NaN，不用于 Table 2，也不改变正式 Ours-FD 的方法定义。原始
FGSM-PGI 的 logit MSE 没有 detach，因此当前没有擅自加入 detach、梯度裁剪或其他
数值修复。

训练器新增 `abort_on_nonfinite=true`：在第一个异常 batch 立即终止并保存
`nonfinite_diagnostic.json`，分别检查：

- 初始 forward、initial loss 和 input gradient；
- adversarial forward、CE、logit MSE、feature MSE 和总 loss；
- backward 后的参数梯度；
- optimizer step 后的参数和 BatchNorm buffers。

每组最多运行 1 epoch，`εT=32/255`、deterministic、monitor PGD-10 step=8/255。

| Run | Objective | α | Logit weight | FD weight | LR | 变量用途 |
|---|---|---:|---:|---:|---:|---|
| `diagnostic_mep_eps32_alpha16_logit10_lr01` | MEP | 16 | 10 | 0 | 0.1 | 复现第一轮共同失败设置 |
| `diagnostic_mep_eps32_alpha8_logit10_lr01` | MEP | 8 | 10 | 0 | 0.1 | 检查更小训练步长 |
| `diagnostic_mep_ce_eps32_alpha8_lr01` | MEP CE 对照 | 8 | 0 | 0 | 0.1 | 仅作 logit 项因果诊断，不是 Ours-FD |
| `diagnostic_mep_eps32_alpha8_logit10_lr001` | MEP | 8 | 10 | 0 | 0.01 | 检查更低学习率 |

调度使用物理 GPU 1/5/6：GPU 1 顺序运行第一组和低学习率组，GPU 5/6 各运行一组。
实验已完成，确认了首个异常的位置：

- `diagnostic_mep_eps32_alpha16_logit10_lr01` 在 **epoch 0、batch 8** 的
  `initial_forward_or_input_gradient` 阶段中止。
- 初始扰动仍有限（最大绝对值为 `16/255`），但 `initial_loss`、`input_gradient` 和
  `logits_initial` 已非有限；此前 optimizer 更新后，参数量级已达约 `1e14`，多个
  BatchNorm running statistics 达 `1e33` 或非有限。
- 因此这是早期优化数值爆炸，而不是输入扰动越界。该诊断不能单独证明唯一根因是
  `alpha=16`、logit 项或学习率，后续的受控 α=8 实验用于进一步区分。

重要口径：去掉 logit 项的配置只能作为数值诊断/消融，不能作为论文 Ours-FD
baseline。正式 Ours-FD 仍定义为 CE + 非零 logit MSE + 非零 feature MSE。

## 10. ε=32 α=8 第二轮 pilot（已完成）

根据第一轮结果，将训练步长进一步降至 8/255，同时所有正式 Ours-FD 候选继续保留
论文 MEP logit 项及其权重 10；纯 MEP 仅作为对照。所有配置均为 40 epochs、
`εT=32/255`、lr=0.1、deterministic、monitor PGD-10 step=8/255、
`abort_on_nonfinite=true`。

| Run | Objective | α | Logit weight | FD weight |
|---|---|---:|---:|---:|
| `pilot_mep_eps32_alpha8_logit10` | FGSM-MEP 对照 | 8 | 10 | 0 |
| `pilot_fd_eps32_alpha8_fw5` | Ours-FD | 8 | 10 | 5 |
| `pilot_fd_eps32_alpha8_fw10` | Ours-FD | 8 | 10 | 10 |
| `pilot_fd_eps32_alpha8_fw25` | Ours-FD | 8 | 10 | 25 |

GPU 1 顺序运行 MEP 对照和 FD weight 25；GPU 5/6 分别运行 FD weight 5/10。
任何配置一旦出现 NaN 会立即停止，不会将部分训练结果当作正式结果。只有通过筛选的
Ours-FD 配置才会重新完整训练论文规定的 110 epochs。

### 10.1 一 epoch 受控诊断结果

在 alpha=8 下，下列三组均完成第一个 epoch 且保持有限；这些是一 epoch 数值诊断，
不是最终鲁棒性结果。数值为测试 clean/PGD-10（百分比）及训练末尾平均损失。

| Run | Clean | PGD-10 | Train acc. | Total loss | CE | Raw logit MSE |
|---|---:|---:|---:|---:|---:|---:|
| MEP, alpha=8, logit=10, lr=0.1 | 18.97 | 8.87 | 13.65 | 2.86045 | 2.64751 | 0.021293 |
| MEP CE-only, alpha=8, lr=0.1 | 35.22 | 5.45 | 18.98 | 2.28556 | 2.28556 | 0 |
| MEP, alpha=8, logit=10, lr=0.01 | 22.72 | 8.15 | 15.89 | 2.31211 | 2.29075 | 0.002136 |

这说明把 alpha 降至 `8/255` 足以避免第一 epoch 的该类数值爆炸；但一 epoch 的
clean/PGD 值不能用于选择论文 baseline，也不能据此断言降低学习率或移除 logit 项更优。

### 10.2 40-epoch alpha=8 pilot 结果

四组均完成 40 epochs、没有 `nonfinite_diagnostic.json`。共同设置为
`epsilon=32/255`、`alpha=8/255`、lr=0.1、seed 0、`deterministic=true`、
logit weight=10、PGD-10 monitor 使用 step `8/255`；指标均为百分比。

| Run | FD weight | Best epoch | Best clean | Best PGD-10 | Final clean | Final PGD-10 |
|---|---:|---:|---:|---:|---:|---:|
| `pilot_mep_eps32_alpha8_logit10` | 0 | 5 | 22.77 | 12.70 | 49.52 | 7.45 |
| `pilot_fd_eps32_alpha8_fw5` | 5 | 4 | 24.97 | 12.13 | 50.82 | 6.92 |
| `pilot_fd_eps32_alpha8_fw10` | 10 | 4 | 24.91 | 13.30 | 54.95 | 3.10 |
| `pilot_fd_eps32_alpha8_fw25` | 25 | 4 | 24.67 | 12.78 | 52.50 | 7.20 |

结论：训练不再 NaN，但鲁棒 PGD-10 仍在极早 epoch 达峰，之后退化。FD=10 的
13.30% 是这一轮最高值，但相对纯 MEP 仅高 0.60 个百分点；它是 seed-0 诊断结果，
不能据此主张显著提升。

### 10.3 更高 FD 权重筛选

为检查低权重范围是否遗漏候选，又在相同 alpha=8/epsilon=32/40-epoch 设置下运行
FD weight 50/100/200/400。结果仍全部在 epoch 4 达到最佳 monitor：

| FD weight | Best clean | Best PGD-10 | Final clean | Final PGD-10 |
|---:|---:|---:|---:|---:|
| 50 | 24.79 | 13.49 | 45.36 | 8.46 |
| 100 | 24.53 | 13.12 | 45.51 | 8.50 |
| 200 | 24.31 | 13.12 | 48.86 | 7.41 |
| 400 | 22.97 | 13.39 | 50.98 | 5.91 |

FD=50 的 13.49% 仅比 FD=10 高 0.19 个百分点、比纯 MEP 高 0.79 个百分点；单 seed
下不构成明确改进。因此停止继续扩展 FD 权重网格。所记录的加权 feature loss 约
`0.0004–0.0006`，但它不是梯度大小，不能据此推断特征约束无效。

## 10.4 ε=32、alpha=8 的 110-epoch 完整周期确认（已完成；机制诊断）

目的：检验 40-epoch pilot 在 epoch 4 的峰值是否仅由截断训练造成。三组均从保存的
epoch-39 `resume.pt` 在独立目录中恢复，完成 epoch 40/80 的 MEP 重置和 epoch
100/105 的学习率衰减；实际使用物理 GPU 5/6/7。共同设置为 `epsilon=32/255`、
`alpha=8/255`、lr=0.1、seed 0、`deterministic=true`、batch size 128、logit
weight=10、matched-epsilon 无噪声 PGD-10 monitor（步长 `8/255`）。

| Full run | FD weight | Best epoch | Best clean | Best PGD-10 | Final clean | Final PGD-10 | Final Vact-B |
|---|---:|---:|---:|---:|---:|---:|---:|
| `full_mep_eps32_alpha8_logit10` | 0 | 5 | 22.77 | 12.70 | 64.36 | 8.32 | 9.01448 |
| `full_fd_eps32_alpha8_fw50` | 50 | 4 | 24.79 | 13.49 | 63.62 | 8.77 | 2.26615 |
| `full_fd_eps32_alpha8_fw200` | 200 | 4 | 24.31 | 13.12 | 63.29 | 7.97 | 1.17642 |

所有数值为百分比（`Vact-B` 除外）。结论：完整周期没有恢复 ε=32 的鲁棒性；最佳
PGD-10 仍在 epoch 4/5。FD 确实降低了特征漂移，但没有转化为鲁棒准确率改善：FD=50
的最终 PGD-10 只比纯 MEP 高 0.45 个百分点，FD=200 更低。该组 `alpha=8` / FD=50/200
实验仅用于解释失败机制，**不进入正式 baseline 主表，也未运行 AA**。

## 11. CIFAR-10 Ours-FD：忠实性审计与 Table 2 baseline 扩展

### 11.1 原论文配置的忠实性审计

为定位原论文配置的适用边界，忠实性审计只改变训练半径
`epsilonT ∈ {8,12,16,32,48,64}/255`。其余设置冻结为：

- 单步训练攻击 `alpha=epsilonT`（checkpoint 中的 `alpha: None` 表示运行时自动采用
  `epsilonT`，并非 alpha=0）。
- MEP + `10 × MSE(logits_adv, logits_initial)`；节点 B 的 FD 权重 200。
- SGD lr=0.1、momentum=0.9、weight decay=`5e-4`、batch size=128、110 epochs。
- label smoothing 的真实类别概率 0.6；MEP 在 epoch 0/40/80 重置，动量衰减 0.3。
- CIFAR-10、CIFAR ResNet18、seed 0、`deterministic=false`，Pad 4/随机裁剪/水平翻转。

有效模型的评估统一采用论文 Table 2 的固定 `epsilonE=16/255`、PGD/C&W 步长
`2/255`、完整攻击迭代，以及对攻击后输入一次 `U(-16/255,16/255)` 裁剪噪声的
non-adaptive attack-then-noise 协议。

### 11.2 原论文配置下有效模型的 Table 2 评估

| epsilonT | Checkpoint epoch | Clean | FGSM | PGD-10 | PGD-20 | PGD-50 | C&W-20 | APGD-T | AA |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8/255 | 107 | 80.42 | 38.81 | 33.79 | 23.62 | 21.52 | 20.66 | 18.80 | 18.32 |
| 12/255 | 109 | 70.76 | 38.66 | 37.48 | 29.87 | 28.59 | 24.25 | 21.21 | 22.00 |
| 16/255 | 35 | 45.83 | 29.59 | 30.31 | 27.36 | 27.12 | 22.50 | 21.89 | 22.36 |

所有数值均为百分比。epsilon=12 是 Clean 与 AA 的较好平衡点；epsilon=16 的 AA
略高 0.36 个百分点，但 Clean 相比 epsilon=12 下降 24.93 个百分点。epsilon=16
的 checkpoint 来自其后期 CO 前的 epoch 35；对应 final 状态仍应作为 CO 曲线/附录
结果保留。

Checkpoint 来源：epsilon=8/16 使用
`/data/cjk/FGSM-MEP-cifar10-fd-sweep/ours_fd_eps{8,16}/best.pt`；epsilon=12 使用
`/data/cjk/FGSM-MEP-runs-v2/diagnostic_fd_plus_mep_eps12/best.pt`。三者的保存配置已
逐项核对为第 11.1 节的冻结设置。

### 11.3 原始配置的高 epsilon 失败审计（配置消融）

对 epsilon=32/48/64，使用同一冻结设置的 1-epoch 前缀（唯一诊断差异为
`epochs=1` 和 `abort_on_nonfinite=true`）进行审计。该 guard 不改变优化计算，只在
第一个非有限 batch 保存诊断并中止。三组均在 seed 0 下的 epoch 0 数值失稳：

| epsilonT | 正式 baseline 状态 | 首次失败位置 | 记录方式 |
|---:|---|---|---|
| 32/255 | `N/A (numerical divergence)` | epoch 0, batch 9, backward | 不运行攻击评估 |
| 48/255 | `N/A (numerical divergence)` | epoch 0, batch 8, backward | 不运行攻击评估 |
| 64/255 | `N/A (numerical divergence)` | epoch 0, batch 7, initial-forward/input-gradient | 不运行攻击评估 |

因此，32/48/64 不是上述冻结配置下的有效鲁棒模型，也不应把任何 NaN 后 checkpoint
纳入结果。结论的适用范围是：在本实现、CIFAR-10/ResNet18、上述冻结协议及 seed 0 下，
原始 `alpha=epsilonT, lr=0.1` 配方无法扩展至 `epsilonT >= 32/255`；这不是关于方法理论
能力的一般性证明，也不阻止为下游 Table 2 baseline 选择稳定的高 epsilon 配方。

### 11.4 当前结论与投稿口径

1. epsilon=8/12/16 已形成统一协议下的有效 baseline 行；epsilon=32/48/64 为相同
   配置下的数值失稳边界。
2. 原论文 epsilon=12 的复现（见第 5 节）与原表接近，支持核心 Ours-FD 实现可信。
3. alpha=8 的 epsilon=32 训练说明“避免立即 NaN”不足以恢复鲁棒学习，应只作为机制
   诊断或附录消融。
4. 当前 `best.pt` 由 test monitor 选择，适合 faithful reproduction 记录；PAMI 正式
   结果应以独立 validation 选择 epoch，并对有效半径使用多 seed 报告均值和标准差。
5. 主表之外应补充无噪声 AA 和 EOT/自适应随机噪声评估，以限定论文 non-adaptive
   attack-then-noise 结果的适用范围。

### 11.5 独立的高 epsilon 稳定性筛选（已完成；不属于正式 baseline）

为响应“避免数值发散、寻找可解释 CO 轨迹”的需求，固定 Ours-FD 的 MEP、logit
weight=10、FD(B)=200、seed 0 和 deterministic 训练，只把训练 `alpha/epsilon` 扫为
`{1/8,1/4,1/2}`、lr 扫为 `{0.01,0.03,0.1}`。每组仅训练 1 epoch，且
`abort_on_nonfinite=true`；这是数值筛选，不能用于模型选择或主表。

共有 27 组，其中 22 组完成有限前缀；5 组仍然数值失稳：

| epsilonT | alpha | lr | 首次失败位置 |
|---:|---:|---:|---|
| 32/255 | 16/255 | 0.10 | epoch 0, batch 11, initial-forward/input-gradient |
| 48/255 | 24/255 | 0.03 | epoch 0, batch 18, initial-forward/input-gradient |
| 48/255 | 24/255 | 0.10 | epoch 0, batch 8, initial-forward/input-gradient |
| 64/255 | 16/255 | 0.10 | epoch 0, batch 11, initial-forward/input-gradient |
| 64/255 | 32/255 | 0.10 | epoch 0, batch 7, backward |

基于“能正常学习”与“更大但仍有限的步长”两种用途，进入下一阶段的 40-epoch 轨迹候选为：

| epsilonT | alpha | lr | 1-epoch Clean | 1-epoch matched PGD-10 | 作用 |
|---:|---:|---:|---:|---:|---|
| 32/255 | 4/255 | 0.10 | 33.25 | 5.37 | 学习导向 |
| 32/255 | 8/255 | 0.10 | 23.56 | 10.90 | 较大有限步长 |
| 48/255 | 6/255 | 0.10 | 27.39 | 4.12 | 学习导向 |
| 48/255 | 12/255 | 0.03 | 15.67 | 7.38 | 较大有限步长 |
| 64/255 | 8/255 | 0.10 | 23.56 | 3.41 | 学习导向 |
| 64/255 | 16/255 | 0.03 | 15.85 | 6.22 | 较大有限步长 |

这些数值仅用于选择少量 finite candidates，绝不能解读为最终鲁棒性能。下一阶段每一组从头
训练 40 epochs，启用 A–E 特征统计，观察 PGD-10 的峰值、最终值和 `Vact-B`；有正常学习
且出现有限、可解释 CO 的候选才值得完整训练，并仍作为调参附录而非原始 baseline。

40-epoch 轨迹筛选已完成，全部 6 组保持数值有限：

| epsilonT | alpha | lr | PGD-10 最佳 epoch | 最佳 Clean | 最佳 PGD-10 | 最终 Clean | 最终 PGD-10 | PGD-10 降幅 | 最终 Vact-B |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32/255 | 4/255 | 0.10 | 0 | 33.25 | 5.29 | 59.90 | 1.65 | 3.64 | 4.47547 |
| 32/255 | 8/255 | 0.10 | 4 | 24.31 | 13.12 | 48.86 | 7.41 | 5.71 | 2.19455 |
| 48/255 | 6/255 | 0.10 | 19 | 30.96 | 5.42 | 36.94 | 2.96 | 2.46 | 2.87567 |
| 48/255 | 12/255 | 0.03 | 28 | 25.86 | 9.62 | 29.76 | 7.47 | 2.15 | 1.45435 |
| 64/255 | 8/255 | 0.10 | 19 | 23.67 | 5.98 | 26.10 | 2.91 | 3.07 | 2.74115 |
| 64/255 | 16/255 | 0.03 | 6 | 16.11 | 9.77 | 21.99 | 8.49 | 1.28 | 0.94656 |

所有准确率为百分比。结果显示，缩小 alpha 并在必要时降低学习率可避免原始高 epsilon
配置的直接数值崩坏；但有限训练不等于高鲁棒性。较大有限步长的三组
`(32,8,0.10)`、`(48,12,0.03)`、`(64,16,0.03)` 在各自 epsilon 下均有更高的
matched-epsilon PGD-10 峰值，并在 40 epoch 内呈现有限的 peak-to-final 下降，同时
`Vact-B` 小于各自的学习导向组。因此它们是下一阶段的“可解释/健康 CO”候选。epsilon=32
的该精确配方已存在完整 110-epoch FD=200 诊断（第 10.4 节）；epsilon=48/64 尚需从
保存的 epoch-39 状态精确继续至 epoch 109。三者始终属于调参诊断，不能取代第 11.3 节
原始 baseline 的 `N/A (numerical divergence)`。

epsilon=48/64 的两个候选已从各自 epoch-39 的 MEP/RNG/optimizer/scheduler 状态精确
续跑至 epoch 109，续跑期间均无 `nonfinite_diagnostic.json`：

| epsilonT | alpha | lr | Best epoch | Best Clean | Best PGD-10 | Final Clean | Final PGD-10 | Final Vact-B | 轨迹解释 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 32/255 | 8/255 | 0.10 | 4 | 24.31 | 13.12 | 63.29 | 7.97 | 1.17642 | 清晰、有限的 CO；来自第 10.4 节既有完整诊断 |
| 48/255 | 12/255 | 0.03 | 28 | 25.86 | 9.62 | 47.33 | 7.34 | 0.88555 | 温和 CO：clean 上升，matched-PGD 峰值后有限下降 |
| 64/255 | 16/255 | 0.03 | 99 | 23.88 | 10.46 | 27.80 | 10.19 | 0.51268 | 数值稳定、低性能；末期接近自身峰值，未显示明显 CO |

所有准确率均为百分比。该轨道可以在 `epsilonT=32/48/64` 得到有限训练轨迹，并提供
epsilon=32/48 的 CO 例子和 epsilon=64 的稳定低性能例子；它**没有**证明这些半径恢复了
与 epsilon=8/12/16 相当的鲁棒性。原始 `alpha=epsilonT, lr=0.1` 的失败仍作为配置消融
保留；下游 Table 2 baseline 则使用下节选出的稳定配方，并明确报告其 alpha/lr。

### 11.6 目标表格更正：AAER 论文的 Table 2（新套件已实现，尚待运行）

下游 baseline 的目标是 arXiv:2404.08154v2（AAER）的 Table 2，而非本记录此前采用的
arXiv:2402.18211 Table 2 协议。AAER Table 2 的 CIFAR-10 行使用 PreActResNet-18，报告
最终模型的 natural accuracy 和同半径 PGD-50-10 accuracy（50 steps、步长 `epsilon/4`、
10 restarts），并以 3 个随机 seed 的均值±标准差呈现。原表列为 8/12/16/32；本项目将其
扩展到 48/64。

因此，现有 8/12/16/32/48/64 结果不能直接填入该表：当前实现是 post-activation CIFAR
ResNet18，训练/选择使用 110-epoch step schedule 与 best checkpoint，评估则是另一论文的
固定 epsilon=16、attack-then-noise、AutoAttack 协议。它们保留为 Ours-FD 的复现和高
epsilon 机制诊断。AAER-compatible baseline 已作为独立套件实现，必须从头以
PreActResNet-18 训练六个半径、保存最终 epoch，并使用匹配半径 PGD-50-10；所有 18 个
（6 radii × 3 seeds）结果完成后再生成扩展 Table 2。

实现位置与冻结协议：

- 训练配置：`configs/train/aaer_ours_fd_cifar10_eps{8,12,16,32,48,64}_seed{0,1,2}.yaml`。
- 模型：新增 CIFAR `PreActResNet18`；节点 B 仍定义为 `layer1` 输出，保持 Ours-FD 的
  特征约束位置。BasicBlock 的投影分支、CIFAR-10 mean/std 与官方
  `related_code/2023_NeurIPS_AAER/CIFAR10/{preact_resnet.py,utils.py}` 对齐；归一化通过
  模型包装器实现，使 MEP 历史扰动和 epsilon 仍保留为 `[0,1]` 像素空间量。
- 训练：原 Ours-FD/MEP 的 110 epochs、0/40/80 reset、100/105 milestones、FD(B)=200、
  10×logit MSE；主结果强制使用 `final.pt`，不使用 PGD-best checkpoint。
- 高 epsilon：复用有限轨迹筛选所得、但须在 PreActResNet18 上重新验证的候选：32 使用
  `alpha=8/255, lr=0.1`，48 使用 `12/255, 0.03`，64 使用 `16/255, 0.03`。8/12/16 仍为
  `alpha=epsilon, lr=0.1`。这是逐行披露的稳定化扩展，不是声称高 epsilon 完全未调参。
- 评估：无随机推理噪声；同半径 PGD-50、步长 `epsilon/4`、10 restarts。为与 AAER 官方
  `utils.py` 一致，PGD 仅更新当前正确分类的样本；汇总器会拒绝非 `final.pt`、旧
  attack-then-noise、错误 PGD 设置或半径不匹配的 JSON。
- 调度：`run_aaer_ours_fd_cifar10_train.sh` 与
  `run_aaer_ours_fd_cifar10_eval.sh` 支持任意逗号分隔 GPU 列表，并对完成/中断运行分别跳过/恢复。

## 12. 后续实验顺序

1. 保留第 11.1–11.3 节的原始配置忠实性审计，作为“直接复用原论文配方在高 epsilon
   发散”的配置消融；它不是下游论文 Table 2 baseline 的唯一可选配方。
2. 运行并验证已实现的 AAER-compatible Ours-FD 套件：PreActResNet-18、最终 checkpoint、
   matching-epsilon PGD-50（步长 epsilon/4、10 restarts）、3 seeds；从头跑
   epsilon=8/12/16/32/48/64。高 epsilon 的稳定 alpha/lr 仍须与低 epsilon 一样在新模型/
   新协议下重新确认。
3. 当前高 epsilon 结果保留为选择候选 alpha/lr 和解释 CO/数值失稳的诊断，不用其
   checkpoint 生成 AAER Table 2 数值。
4. 在 CIFAR-10 协议冻结后，扩展同一 baseline 至 CIFAR-100；代码现已支持 CIFAR-100
   数据加载和 100 类 PreActResNet-18 输出，但尚未建立/运行独立的训练、失败审计与三 seed
   配置矩阵。
5. 保留 matched-epsilon 无噪声 AA、alpha=8 机制诊断和 CO/特征曲线作为补充材料。

## 13. 运行路径与代码版本

HPC 路径：

- 仓库：`/data/cjk/FGSM-MEP`
- CIFAR 数据：`/data/cjk/cifar-data`
- 第一轮 sweep：`/data/cjk/FGSM-MEP-cifar10-fd-sweep`
- ε=32 pilot：`/data/cjk/FGSM-MEP-cifar10-fd-pilots`
- ε=32 非有限值诊断：`/data/cjk/FGSM-MEP-cifar10-eps32-nonfinite`
- ε=32 α=8 pilot：`/data/cjk/FGSM-MEP-cifar10-eps32-alpha8-pilots`
- ε=32 α=8 full-cycle diagnostic：`/data/cjk/FGSM-MEP-cifar10-eps32-alpha8-full110`
- 原始高 epsilon 失败审计：`/data/cjk/FGSM-MEP-cifar10-native-high-eps-audit`
- 高 epsilon 1-epoch 稳定性网格：`/data/cjk/FGSM-MEP-cifar10-high-eps-stability-grid`
- 高 epsilon 40-epoch CO 轨迹筛选：`/data/cjk/FGSM-MEP-cifar10-high-eps-trajectory-screen`
- 高 epsilon 48/64 精确续跑：`/data/cjk/FGSM-MEP-cifar10-high-eps-healthy-co-full110`
- 冻结 CIFAR-10 Table 2 评估：`/data/cjk/FGSM-MEP-formal-baseline-table2`

关键提交：

- `80aeb55`：正式 Ours-FD 启用 FGSM-MEP logit 正则。
- `8870932`：加入 CIFAR-10 Ours-FD epsilon sweep 和通用汇总。
- `a08f26b`：加入 ε=32 stability pilots、强 monitor 和 loss 分项日志。
- `8b7a55f`：将 pilot 调度修改为 GPU 4/5/6。
- `612e785`：加入首个非有限 batch 的数值诊断与 GPU 1/5/6 调度。
- `c5156ba`：加入保留 logit weight=10 的 ε=32、α=8 第二轮 pilot。
- `1fc1ab1`：加入 ε=32、α=8 的高 FD weight 50/100/200/400 pilot。
- `5e697e2`：加入安全的 40→110 epoch continuation、三卡启动器、汇总器和测试。

当前 Python 目标环境：Python 3.10、PyTorch 2.0.1、torchvision 0.15.2、CUDA
11.8。每个完整 MEP `resume.pt` 约 1.3 GiB，应为大规模多 seed 实验预留足够磁盘。
