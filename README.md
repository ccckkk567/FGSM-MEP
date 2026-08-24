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

启动器固定使用物理 GPU 0/1/2/3，在当前终端等待所有进程结束，不创建 tmux
session。已有 `final.pt` 会跳过，只有 `resume.pt` 时自动续训。训练日志写到输出目录
的 `logs/`，逐 epoch 的未加权 CE、logit MSE 和 feature MSE 写入
`loss_components.csv`。

## 测试

```bash
pytest
```

测试不下载 CIFAR-10；集成测试使用合成小数据验证训练、checkpoint 和恢复流程。
