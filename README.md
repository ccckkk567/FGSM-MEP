# Catastrophic Overfitting: A Potential Blessing in Disguise — reproduction

这是 arXiv:2402.18211v1 的独立 CIFAR-10 / ResNet18 复现实现。正式主表使用
FGSM-MEP；FGSM-RS 只作为论文描述歧义的诊断后端。`related_code/` 不会被修改，
其中 ConvergeSmooth 自带的旧 AutoAttack 快照仅在 APGD-T/AA 评估时只读加载；其
SHA-256 固定为 `aeb3b5167a3e4971af0fb0192733cff9b8e5bba79ef5722dd1a1fe576db1afc0`，
与 FGSM-PGI 中的副本一致。快照变化时评估会直接报错，避免悄悄更换攻击版本。

## 环境

目标环境是 Linux、单张 RTX 3090：

```bash
conda env create -f environment.yml
conda activate co-blessing
```

锁定版本为 Python 3.10、PyTorch 2.0.1、torchvision 0.15.2 和 CUDA 11.8。
驱动版本只需满足 CUDA 11.8。每个 MEP resume checkpoint 包含两份 50,000 张图像
大小的状态缓冲，约 1.2 GiB；这是精确断点恢复所需，运行目录应预留足够磁盘。

## 训练一个模型

Ours-FD、训练 ε=12/255：

```bash
python -m co_blessing train \
  --config configs/train/ours_fd_eps12.yaml \
  --data-root /path/to/cifar-data \
  --output-root /path/to/runs \
  --device cuda:0
```

正式 Ours-FD 在 FGSM-MEP 损失上加入节点 B 的特征差异约束，训练目标为
`smooth CE + 10 × logit MSE + 200 × feature MSE(B)`。该组合在 CIFAR-10、
ε=12/255 上复现了论文 Table 3；所有 `ours_fd_eps*.yaml` 均已启用这一目标。

Ours-CO 使用相应的 `ours_co_eps*.yaml`。FD 的论文表格 checkpoint 是 `best.pt`；
CO 使用 `final.pt`。`resume.pt` 保存优化器、调度器、随机数和完整 MEP 状态：

```bash
python -m co_blessing train \
  --config configs/train/ours_fd_eps12.yaml \
  --resume /path/to/runs/ours_fd_eps12/resume.pt \
  --data-root /path/to/cifar-data \
  --output-root /path/to/runs \
  --device cuda:0
```

必要的 MEP 基线和 RS 诊断配置分别是 `mep_baseline_eps12.yaml`、
`mep_baseline_eps16.yaml`、
`rs_fd_eps12.yaml`、`rs_co_eps12.yaml`。
`diagnostic_fd_plus_mep_eps12.yaml` 是此前验证该损失组合时使用的历史诊断别名；
现在与正式 `ours_fd_eps12.yaml` 的训练目标一致，不属于主 manifest。

## 论文评估协议

```bash
python -m co_blessing evaluate \
  --config configs/eval/paper.yaml \
  --checkpoint /path/to/runs/ours_fd_eps12/best.pt \
  --data-root /path/to/cifar-data \
  --output-root /path/to/runs/evaluations \
  --device cuda:0
```

该命令依次执行 Clean、FGSM、PGD-10/20/50、C&W-20、APGD-T、AA。攻击针对
无噪声确定性模型生成，之后才加入一次 `U(-16/255,16/255)` 并裁剪至 `[0,1]`。
PGD/C&W 执行完整迭代，不按当前分类结果提前停止；FGSM 从零扰动开始。该协议没有
EOT。若要确认模型是否发生 CO，使用 `configs/eval/no_noise.yaml`。

首次定位复现偏差时可先运行 `configs/eval/diagnostic_iterative.yaml`，它跳过耗时的
APGD-T/AA，只计算 Clean、FGSM、PGD 和 C&W，并复刻 FGSM-PGI/ConvergeSmooth 的
旧评估语义：FGSM 随机起点，样本一旦被误分类就冻结扰动。论文没有说明加入随机
噪声后是否裁剪回 `[0,1]`；`diagnostic_iterative_no_clip.yaml` 保留越界值，用来与
裁剪版本做一次受控对照，正式 `paper.yaml` 仍按原复现计划执行裁剪。

比较一组评估结果与论文 Table 2：

```bash
python -m co_blessing compare \
  --results /path/to/eval_*/evaluation.json \
  --output /path/to/report
```

输出逐指标 paper/reproduced/delta，不使用人为通过阈值。

## 机制分析

单 checkpoint 的 A–E 特征/通道统计：

```bash
python -m co_blessing analyze --task features \
  --config configs/analysis/features.yaml \
  --checkpoint /path/to/runs/ours_co_eps12/final.pt \
  --output /path/to/analysis/co_features \
  --data-root /path/to/cifar-data --device cuda:0
```

ε=12 配置在训练期间记录了逐 epoch `Vact`。将 CO 和 FD 曲线画在一起：

```bash
python -m co_blessing analyze --task features \
  --runs /path/to/runs/ours_co_eps12 /path/to/runs/ours_fd_eps12 \
  --output /path/to/analysis/vact_curves
```

节点 A 的阈值掩码实验：

```bash
python -m co_blessing analyze --task mask \
  --config configs/analysis/masks.yaml \
  --checkpoint /path/to/runs/ours_co_eps12/final.pt \
  --output /path/to/analysis/masks \
  --data-root /path/to/cifar-data --device cuda:0
```

诱导 CO 的 p=1/3/10/20 曲线：

```bash
python -m co_blessing analyze --task induce \
  --runs /path/to/runs/induce_p1_eps8 /path/to/runs/induce_p3_eps8 \
         /path/to/runs/induce_p10_eps8 /path/to/runs/induce_p20_eps8 \
  --output /path/to/analysis/induce
```

## 一次运行完整清单

下面的命令顺序训练并评估 8 个主表模型，再训练 4 个诱导模型。已有 `final.pt` 或
`evaluation.json` 会跳过，只有 `resume.pt` 时会自动续训：

```bash
python -m co_blessing reproduce \
  --manifest configs/manifests/cifar10_resnet18.yaml \
  --data-root /path/to/cifar-data \
  --output-root /path/to/runs \
  --device cuda:0
```

论文报告每次训练约 92–104 分钟；单卡完成全部训练约需十余 GPU 小时，完整
AutoAttack 还会增加明显耗时。实际结果会受 PyTorch/cuDNN 和 GPU 算法选择影响，
所有版本、seed、配置与 checkpoint epoch 都会写入运行目录。

## CIFAR-10 Ours-FD epsilon sweep

`cifar10_fd_sweep.yaml` 固定已验证的 Ours-FD 目标，只扫描训练/评估半径
`8/12/16/32/48/64`。每组用对应半径、无推理噪声评估，并使用训练期间 PGD-10
准确率最高的 `best.pt`。评估包含 Clean、FGSM、PGD-10/50、C&W-20 和完整
AutoAttack；PGD/C&W 步长保持论文协议的 `2/255`。

两张 GPU 可分别运行互不重叠的分片；两个进程必须使用同一个全新输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 python -m co_blessing reproduce \
  --manifest configs/manifests/cifar10_fd_sweep_gpu0.yaml \
  --data-root /path/to/cifar-data \
  --output-root /path/to/cifar10-fd-sweep \
  --device cuda:0

CUDA_VISIBLE_DEVICES=1 python -m co_blessing reproduce \
  --manifest configs/manifests/cifar10_fd_sweep_gpu1.yaml \
  --data-root /path/to/cifar-data \
  --output-root /path/to/cifar10-fd-sweep \
  --device cuda:0
```

分片完成后运行完整 manifest。已有 checkpoint 和评估会被跳过，此命令只汇总六组
结果到 `cifar10_fd_sweep_report/sweep_summary.{csv,md}`：

```bash
CUDA_VISIBLE_DEVICES=0 python -m co_blessing reproduce \
  --manifest configs/manifests/cifar10_fd_sweep.yaml \
  --data-root /path/to/cifar-data \
  --output-root /path/to/cifar10-fd-sweep \
  --device cuda:0
```

若任务中断，原命令会从各运行目录的 `resume.pt` 自动恢复。不要复用此前采用旧损失
定义的运行目录，否则存在同名 `final.pt` 时会被当成已完成实验跳过。

### ε=32 stability pilots

当固定 `feature_weight=200` 的大 ε 训练发生坍塌时，使用四张卡并行运行 40-epoch
筛选。四组分别是 `(alpha, feature_weight)=(32,25),(16,25),(16,10)`，以及
`alpha=16` 的纯 MEP 对照；checkpoint 监控使用 ε=32、步长 8/255 的 PGD-10，
所有 pilot 开启确定性算法，且不执行正式攻击评估：

```bash
./run_cifar10_eps32_pilots.sh \
  /path/to/cifar-data \
  /path/to/cifar10-fd-pilots

python summarize_cifar10_eps32_pilots.py /path/to/cifar10-fd-pilots
```

启动器使用物理 GPU 4/5/6：GPU 4 顺序执行第一组和纯 MEP 对照，GPU 5/6 各执行
一组。它在当前终端等待所有进程结束，不创建 tmux session。已有 `final.pt` 会跳过，
只有 `resume.pt` 时自动续训。训练日志写到输出目录的 `logs/`，逐 epoch 的未加权
CE、logit MSE 和 feature MSE 写入 `loss_components.csv`。

若 pilot 在首个 epoch 就产生 NaN，运行数值诊断。它在第一个非有限 batch 立即中止，
并分别检查前向、输入梯度、反向梯度和优化器更新：

```bash
./run_cifar10_eps32_nonfinite_diagnostics.sh \
  /path/to/cifar-data \
  /path/to/eps32-nonfinite

python summarize_cifar10_eps32_nonfinite.py /path/to/eps32-nonfinite
```

### ε=32、α=8：从 40 epochs 续训到完整 110 epochs

只确认三组：纯 MEP（FD=0）、Ours-FD（FD=50）、Ours-FD（FD=200）。默认分别使用
物理 GPU **1、5、6**，每个子进程内部都是 `cuda:0`；不创建 tmux session，不自动评估 AA。
保留 `lr=0.1`、MEP 在 epoch 40/80 的重置，以及绝对里程碑 100/105 的学习率衰减。
40-epoch pilot 没有经历这些后续阶段；本轮是完整周期确认，不是继续增加权重搜索。

在 HPC 上更新代码、激活 `co-blessing` 后，直接在现有终端执行：

```bash
cd /data/cjk/FGSM-MEP
conda activate co-blessing
pytest -q

bash run_cifar10_eps32_alpha8_full110.sh \
  /data/cjk/cifar-data \
  /data/cjk/FGSM-MEP-cifar10-eps32-alpha8-full110
```

默认从以下两个根目录读取已完成的 pilot（只读，不覆盖）：

- `/data/cjk/FGSM-MEP-cifar10-eps32-alpha8-pilots`：`pilot_mep_eps32_alpha8_logit10`。
- `/data/cjk/FGSM-MEP-cifar10-eps32-alpha8-highfd`：`pilot_fd_eps32_alpha8_fw50`、`pilot_fd_eps32_alpha8_fw200`。

如源目录不同，可把这两个根目录依次作为脚本的第三、第四个参数。三份运行记录和
MEP checkpoint 的复制约需 4 GB；考虑原子保存时的临时副本，建议预留至少 10 GB。
准备阶段有 CPU 校验/复制时间。
也可以先执行 `python continue_cifar10_eps32_alpha8_full110.py --prepare-only`，只准备、不占 GPU。

准备器会核对 source `resume.pt/final.pt` 的 epoch=39、两份 CSV 的连续性、最佳模型、
优化器/调度器/MEP/RNG 状态及训练配置；复制完整运行到 `full_*` 子目录。旧 final 文件
移入副本中的 `source_pilot/`，旧配置/环境也在这里留档；源目录完全不变。新的
`continuation_config.yaml` 继承保存的配置，只变更名称、路径、设备和 `epochs=110`。
`continuation.json` 记录来源、目标及原 checkpoint/config 的 SHA256。

续训使用 `resume.pt`，保留复制来的 `best.pt`、0–39 的 CSV 和损失记录，日志追加到
新根目录的 `logs/full_*.log`。如果后续没有更好的 PGD-10，`best.pt` 仍可能是原 pilot
的 epoch 4/5，内部原始名称/40-epoch 配置也会保留；不要因此误认为续训没有发生。

重复执行同一命令会检查并恢复已保存的 epoch，只有完整 epoch 109 的最终 checkpoint
和 110 行连续日志通过核对才跳过。若中断发生在 CSV/checkpoint 分别写入的间隙，导致
epoch 不一致，或出现非有限诊断，准备器会拒绝续训并指出路径；不会擅自截断日志、
覆盖 checkpoint 或从零重新训练。不要同时手动向同一个 `full_*` 目录启动训练。

完成后汇总：

```bash
python summarize_cifar10_eps32_alpha8_full110.py \
  /data/cjk/FGSM-MEP-cifar10-eps32-alpha8-full110 \
  | tee /data/cjk/FGSM-MEP-cifar10-eps32-alpha8-full110/summary.md
```

本轮仍是 **εE=32/255、步长 8/255、无噪声 PGD-10 测试集监控**，不是完整 AA，也不是
原 Table 2 固定 εE=16/255、攻击后加噪的评估。`alpha=8` 是已调小步长的扩展配置，
不是原论文所有参数不变的复现。当前 checkpoint/权重选择用了测试集，结果只用于
探索；正式投稿需分开独立验证集选优与最终测试评估。

### 原始 Ours-FD 的高 epsilon 失败审计

正式半径扫描固定为 `εT={8,12,16,32,48,64}/255`，不额外加入 ε=24。对于尚未形成
有效模型的 32/48/64，先运行 1-epoch 前缀审计：除了 `epochs=1` 与
`abort_on_nonfinite=true` 外，配置严格保留 Ours-FD 的 `alpha=epsilon`、`lr=0.1`、
MEP + 10×logit MSE、节点 B 的 FD=200。GPU 5/6/7 分别对应 ε=32/48/64：

```bash
bash run_cifar10_native_high_eps_audit.sh \
  /data/cjk/cifar-data \
  /data/cjk/FGSM-MEP-cifar10-native-high-eps-audit

python summarize_cifar10_native_high_eps_audit.py \
  /data/cjk/FGSM-MEP-cifar10-native-high-eps-audit \
  | tee /data/cjk/FGSM-MEP-cifar10-native-high-eps-audit/summary.md
```

`NUMERICAL_DIVERGENCE` 记录为**原论文冻结配置审计**的
`N/A (numerical divergence)`，不对该 checkpoint 运行攻击评估。`FINITE_PREFIX` 不是
成功结论，必须以不改变任何训练超参的条件继续到 110 epochs；若之后鲁棒准确率崩塌，则
记录为 CO/鲁棒学习失败。它说明原始 `alpha=epsilon, lr=0.1` 配方的高 epsilon 边界；
下游论文的 Table 2 baseline 可另行选择完整训练且数值有限的稳定配方，但必须逐行披露
对应的 alpha 与 lr。

### 高 epsilon 稳定/健康 CO 调参轨道

若需要研究“不数值发散、但可观察正常 CO”的高 epsilon 训练，可使用独立的诊断网格。
该网格**不会修改或替代**上节冻结配置审计的 `N/A` 结论：它固定 Ours-FD 的 MEP、
logit weight=10、节点 B 的 FD=200，只扫描 `alpha/epsilon ∈ {1/8,1/4,1/2}` 和
`lr ∈ {0.01,0.03,0.1}`。共 27 个 deterministic、seed-0、1-epoch 前缀，物理 GPU
5/6/7 自动并行；每个发现 NaN 的候选都会保存诊断并继续下一个候选。

```bash
python run_cifar10_high_eps_stability_grid.py \
  --data-root /data/cjk/cifar-data \
  --output-root /data/cjk/FGSM-MEP-cifar10-high-eps-stability-grid \
  --gpus 5 6 7

python summarize_cifar10_high_eps_stability_grid.py \
  /data/cjk/FGSM-MEP-cifar10-high-eps-stability-grid \
  | tee /data/cjk/FGSM-MEP-cifar10-high-eps-stability-grid/summary.md
```

`FINITE_PREFIX` 只代表该超参对在一个 epoch 内数值有限。选择有限候选后，应固定候选
配置并运行完整 CO 轨迹，再分别报告 clean、强 PGD 和特征统计；不得把本筛选的结果
直接写入结果表；完成完整训练和统一攻击评估后，选定的稳定配方可作为下游论文 Table 2
baseline，但须在表注中报告其 alpha/lr。

### 高 epsilon 的 40-epoch CO 轨迹筛选

一 epoch 筛选完成后，不以其 clean 或 PGD-10 数值选 checkpoint。每个半径保留一个
学习导向的较小步长和一个较大但仍有限的步长：`(ε, α, lr)` 为
`(32,4,0.1)`、`(32,8,0.1)`、`(48,6,0.1)`、`(48,12,0.03)`、
`(64,8,0.1)`、`(64,16,0.03)`（均为 `/255`，lr 无单位）。它们均从头运行 40 epochs，
记录每 epoch 的 matched-epsilon PGD-10 与 A–E 特征差异。这个阶段用于区分正常学习、
可解释的 CO 和数值发散；它仍不是正式 baseline 或最终 checkpoint 选择。

```bash
python run_cifar10_high_eps_trajectory_screen.py \
  --data-root /data/cjk/cifar-data \
  --output-root /data/cjk/FGSM-MEP-cifar10-high-eps-trajectory-screen \
  --gpus 5 6 7

python summarize_cifar10_high_eps_trajectory_screen.py \
  /data/cjk/FGSM-MEP-cifar10-high-eps-trajectory-screen \
  | tee /data/cjk/FGSM-MEP-cifar10-high-eps-trajectory-screen/summary.md
```

汇总器中的 `Best→final PGD drop` 是 CO 曲线的诊断量，不是通过阈值。尤其不能把 40 epoch
期间的 test PGD-10 峰值直接作为投稿表格模型；有效候选之后应重新训练完整周期，并使用
独立 validation 选择 checkpoint。

### 选择的 48/64 轨迹精确续跑

40-epoch 结果中，ε=48 的 `(alpha, lr)=(12/255, 0.03)` 与 ε=64 的
`(16/255, 0.03)` 是较大但有限的候选；ε=32 的 `(8/255, 0.1)` 已有独立的完整周期
诊断。下面的续跑器只处理前两者：先将完整的 40-epoch 运行复制到新目录，并精确恢复
model、optimizer、scheduler、MEP state 及 RNG。除名称、输出目录、设备和总 epoch 数
从 40 改为 110 外，训练配方不得改变；源目录不会被修改。

先执行不占 GPU 的准备/完整性检查：

```bash
python continue_cifar10_high_eps_trajectory_full110.py \
  --source-root /data/cjk/FGSM-MEP-cifar10-high-eps-trajectory-screen \
  --output-root /data/cjk/FGSM-MEP-cifar10-high-eps-healthy-co-full110 \
  --data-root /data/cjk/cifar-data \
  --gpus 0 1 2 3 \
  --prepare-only
```

检查确认两条运行均显示 `READY: resume at epoch 40` 后，去掉 `--prepare-only` 才会使用
前两个 GPU 继续训练。续跑器会拒绝非连续 CSV、非有限数值、配方变更或过早的 final
checkpoint；输出仍是调参诊断，不能代替原始 Ours-FD baseline。

## 测试

```bash
pytest
```

测试不下载 CIFAR-10；集成测试使用合成小数据验证训练、checkpoint 和恢复流程。

## AAER Table 2 兼容的 Ours-FD baseline

`configs/manifests/aaer_ours_fd_cifar10.yaml` 是新的、独立的投稿 baseline 矩阵。它不复用
本 README 前文的旧 checkpoint：模型为 **PreActResNet-18**，每个半径运行 Ours-FD 的原始
110-epoch MEP 日程（reset 0/40/80，学习率里程碑 100/105），结果只使用 `final.pt`。

训练和评估半径为 `8/12/16/32/48/64 / 255`，每个半径使用 seeds `0/1/2`。低半径保留原始
`alpha=epsilon, lr=0.1`；为避免已审计的高半径数值发散，32/48/64 分别固定为
`(alpha, lr)=(8/255,0.1),(12/255,0.03),(16/255,0.03)`。这三个稳定化选择会写入每份
`config.yaml`，应在论文表注中披露。

AAER 评估不加本论文的 attack-then-noise：每个最终模型只报告 Clean 与同半径
PGD-50-10（步长 `epsilon/4`、10 random restarts）。新实现逐项对齐
`related_code/2023_NeurIPS_AAER/CIFAR10/`：网络输入采用其 CIFAR-10 标准化常数，
而 MEP 扰动仍以像素空间保存；PGD 按其 `utils.py` 仅更新当前仍被正确分类的样本。
汇总器仅接受这一协议和 `final.pt`，并输出三 seed 的 mean ± sample std。

在 HPC 激活 `co-blessing` 后，以下命令以四张物理 GPU 为例运行所有 18 个训练。脚本不新建
tmux，按 GPU 队列调度；已有 `final.pt` 会跳过，只有 `resume.pt` 时自动恢复。

```bash
cd /data/cjk/FGSM-MEP
conda activate co-blessing
pytest -q

bash run_aaer_ours_fd_cifar10_train.sh \
  /data/cjk/cifar-data \
  /data/cjk/FGSM-MEP-aaer-ours-fd-cifar10 \
  0,1,2,3
```

完成训练后再运行完整 PGD-50-10；不要与训练并发，以免评估占用训练 GPU：

```bash
bash run_aaer_ours_fd_cifar10_eval.sh \
  /data/cjk/cifar-data \
  /data/cjk/FGSM-MEP-aaer-ours-fd-cifar10 \
  0,1,2,3
```

最终表格写入
`/data/cjk/FGSM-MEP-aaer-ours-fd-cifar10/aaer_ours_fd_cifar10_table2/aaer_table2_summary.md`。
18 个 MEP `resume.pt` 按每份约 1.3 GiB 计就需约 24 GiB；考虑 final/best、原子保存和日志，
建议至少预留 60 GiB。

### AAER-PreAct 全半径稳定性筛选

若正式 `λFD=200` 在任一 seed 出现 feature-loss/BN 数值爆炸，先运行下面的独立筛选，
不要复用失败运行目录。它覆盖全部 `ε=8/12/16/32/48/64`，固定
`alpha=(8,12,16,8,12,16)/255`，扫描 `λFD={1,5,10,25}`、`lr={0.01,0.03}` 和 seeds
`0/1/2`，共 144 条 40-epoch 轨迹。每 epoch 仅监控 1,000 测试样本的 matched-ε PGD-10；
它用于筛除数值崩坏，**不是**最终模型选择或 Table 2 结果。

筛选不保存 1.3 GiB 的 `resume.pt`，完成项仅保留 final/best checkpoint 与曲线；中断项会从
epoch 0 重跑。因此 144 个筛选不需要约 190 GiB 的 MEP 状态空间。

```bash
python run_aaer_ours_fd_stability_screen.py \
  --data-root /data/cjk/cifar-data \
  --output-root /data/cjk/FGSM-MEP-aaer-ours-fd-cifar10-stability-screen \
  --gpus 0 1 2 3 4 5 6 7

python summarize_aaer_ours_fd_stability_screen.py \
  /data/cjk/FGSM-MEP-aaer-ours-fd-cifar10-stability-screen \
  | tee /data/cjk/FGSM-MEP-aaer-ours-fd-cifar10-stability-screen/summary.md
```

只有同一 `(epsilon, alpha, lambdaFD, lr)` 的三个 seed 均为 `ALL_FINITE`，才进入下一轮
110-epoch、保留 `resume.pt` 的最终训练。正式训练仍只报告 final checkpoint。

### 筛选后的 AAER-PreAct 最终训练与评估

40-epoch 筛选冻结的候选为：`epsilon=8/12/16/32/48/64` 对应
`(alpha, lambdaFD, lr)=(8,1,.03),(12,5,.01),(16,10,.01),(8,25,.01),
(12,5,.01),(16,10,.01)`。其中 epsilon=64 没有得到同时具有非随机 clean
精度且末期 PGD 稳定的候选；这里保留的是最终 PGD-10 最高的有限已学习配置，必须在论文
中标为高半径退化/CO-like regime，不能称为高 epsilon 的成功防御。

训练目录与筛选目录严格分开，配置和选择理由会写入 `selected_manifest.json`。训练只使用原
Ours-FD 110 epoch 日程和 `final.pt`，不按 PGD monitor 选择 best checkpoint：

```bash
python run_aaer_ours_fd_selected_full.py \
  --data-root /data/cjk/cifar-data \
  --output-root /data/cjk/FGSM-MEP-aaer-ours-fd-cifar10-selected \
  --screen-root /data/cjk/FGSM-MEP-aaer-ours-fd-cifar10-stability-screen \
  --gpus 0 1 2 3 4 5 6 7
```

完成且仅在 18 个 `final.pt` 都存在时，才运行 AAER Table-2 的 matched-epsilon PGD-50
（步长 epsilon/4、10 restarts）：

```bash
python run_aaer_ours_fd_selected_eval.py \
  --data-root /data/cjk/cifar-data \
  --output-root /data/cjk/FGSM-MEP-aaer-ours-fd-cifar10-selected \
  --gpus 0 1 2 3 4 5 6 7
```
