# Catastrophic Overfitting: A Potential Blessing in Disguise — reproduction

这是 arXiv:2402.18211v1 的独立 CIFAR-10 / ResNet18 复现实现。正式主表使用
FGSM-MEP；FGSM-RS 只作为论文描述歧义的诊断后端。`related_code/` 不会被修改，
其中 ConvergeSmooth 自带的旧 AutoAttack 快照仅在 APGD-T/AA 评估时只读加载；其
SHA-256 固定为 `25cd36d8e1b685755c7f988167daa71eb0ee95cbfff1c0add4b036542bfdb189`，
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
这严格对应论文的 non-adaptive protocol；第一版没有 EOT。若要确认模型是否发生
CO，使用 `configs/eval/no_noise.yaml`。

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

## 测试

```bash
pytest
```

测试不下载 CIFAR-10；集成测试使用合成小数据验证训练、checkpoint 和恢复流程。
