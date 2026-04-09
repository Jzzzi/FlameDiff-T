# Combustion 时间超分辨率框架使用说明

## 1. 项目目标

本项目面向 RealPDEBench combustion `real / observed` 单通道火焰序列。每条原始样本是一个长时序 trajectory，框架会在时间维上切出长度为 10 的滑动窗口，并按如下任务定义组织数据：

- 条件输入：第 `0` 帧和第 `9` 帧
- 恢复目标：中间 `8` 帧
- 方法主线：`Frame Autoencoder + Conditional Latent DiT/Diffusion`

## 2. 数据读取

默认数据根目录已经配置为：

```bash
/mnt/bn/embodied-lf3-data/realpdebench
```

combustion 数据接入位置：

```text
combustion/hf_dataset/real/data-*.arrow
```

可以先检查数据和切分：

```bash
python scripts/inspect_combustion_data.py --config configs/autoencoder_base.yaml
```

说明：

- 数据按 trajectory 级别划分 train / val / test，避免同一条长序列泄漏到不同 split。
- `window_size=10`、`stride=1` 默认在 YAML 中配置。
- 归一化由 `data.normalization` 控制，默认使用 `[0, 1]` min-max；如需自动统计训练集范围，可将 `auto_compute` 设为 `true`。

## 3. 训练 Autoencoder

单卡：

```bash
torchrun --standalone --nproc_per_node=1 scripts/train_autoencoder.py --config configs/autoencoder_base.yaml
```

多卡 DDP：

```bash
torchrun --nproc_per_node=4 scripts/train_autoencoder.py --config configs/autoencoder_base.yaml
```

如果要沿用集群脚本风格，可以直接使用：

```bash
EXP_NAME=combustion-ae-v0 \
RESULTS_DIR=results/autoencoder \
WANDB_ENABLED=true \
bash entrypoints/entry_train_autoencoder.sh \
  trainer.val_every_steps=500 \
  data.batch_size=4
```

训练输出默认写入 `outputs/autoencoder/`，包含：

- `checkpoints/best.pt` 和 `checkpoints/last.pt`
- `logs/autoencoder_train.jsonl`
- `visuals/ae_epoch_*.png`
- `visuals/ae_step_*.png` 或 `ae_epoch_*_step_*.png`，用于训练中途 validation 可视化
- WandB 中同步记录标量和 validation 图像

## 4. 训练 Diffusion / DiT

先把 `configs/diffusion_base.yaml` 中的 `model.autoencoder.checkpoint` 改成已训练 AE 权重，然后执行：

```bash
torchrun --standalone --nproc_per_node=1 scripts/train_diffusion.py --config configs/diffusion_base.yaml
```

DDP：

```bash
torchrun --nproc_per_node=4 scripts/train_diffusion.py --config configs/diffusion_base.yaml
```

集群脚本方式：

```bash
EXP_NAME=combustion-diffusion-v0 \
AUTOENCODER_CKPT=outputs/autoencoder/checkpoints/best.pt \
WANDB_ENABLED=true \
bash entrypoints/entry_train_diffusion.sh \
  trainer.val_every_steps=500 \
  data.batch_size=2
```

扩散只对中间 8 帧 latent 加噪，首尾帧始终作为固定条件输入。
训练过程中支持按步数插入 validation，并保存当前插值预览图，同时同步到 WandB。

## 5. 插值推理

基于测试集样本：

```bash
python scripts/infer_interpolation.py \
  --config configs/diffusion_base.yaml \
  --diffusion-checkpoint outputs/diffusion/checkpoints/best.pt \
  --sample-index 0
```

基于外部首尾帧：

```bash
python scripts/infer_interpolation.py \
  --config configs/diffusion_base.yaml \
  --diffusion-checkpoint outputs/diffusion/checkpoints/best.pt \
  --frame0 path/to/frame0.npy \
  --frame9 path/to/frame9.npy
```

推理输出包含 `prediction.npy`、序列图、GIF 和可选 MP4。

## 6. 评测

```bash
python scripts/evaluate.py \
  --config configs/diffusion_base.yaml \
  --diffusion-checkpoint outputs/diffusion/checkpoints/best.pt \
  --split test
```

评测指标覆盖中间 8 帧的：

- `MSE`
- `MAE`
- `PSNR`
- `SSIM`

输出包含：

- `summary.json`
- `per_sample_metrics.csv`
- `qualitative/case_*/comparison.png`

## 7. 主要配置项

- `data.dataset_root`: RealPDEBench 根目录
- `data.window_size` / `data.stride`: 切窗长度和步长
- `data.splits`: trajectory 级别切分比例
- `data.normalization`: 归一化模式和参数
- `model.autoencoder.*`: AE 编解码器结构
- `model.dit.*`: latent DiT 的 patch、宽度、层数、头数
- `diffusion.*`: DDPM 时间步和噪声日程
- `trainer.resume`: 从 `last.pt` 恢复训练
- `trainer.precision`: `fp32` / `fp16` / `bf16`
- `trainer.val_every_steps`: 每隔多少个训练 step 触发一次中途 validation，`0` 表示关闭
- `trainer.interval_val_batches`: 中途 validation 最多跑多少个 val batch，避免过于频繁时开销过大；epoch 末 validation 仍可跑完整验证集
- `trainer.full_val_at_epoch_end`: 是否在每个 epoch 结束时跑完整 validation
- `trainer.save_val_visuals`: 是否在 validation 时保存重建/插值可视化
- `experiment.output_dir`: 当前实验输出目录
- `wandb.*`: WandB 项目、实体、run name 和开关

## 8. 启动脚本环境变量

entrypoint 会识别并传递以下常用环境变量：

- `EXP_NAME`
- `CONFIG_PATH`
- `RESULTS_DIR`
- `LOG_ROOT`
- `PRECISION`
- `RESUME`
- `CKPT_PATH`
- `WANDB_ENABLED`
- `PROJECT`
- `ENTITY`
- `NPROC_PER_NODE`
- `NNODES`
- `NODE_RANK`
- `MASTER_ADDR`
- `MASTER_PORT`

Diffusion 启动脚本额外支持：

- `AUTOENCODER_CKPT`

除了环境变量，entrypoint 还支持把任意配置覆盖直接作为命令行参数追加在最后，例如：

```bash
bash entrypoints/entry_train_diffusion.sh \
  data.batch_size=2 \
  trainer.val_every_steps=250 \
  model.dit.depth=12
```

## 9. 当前注意事项

- 当前框架已经具备训练、推理、评测完整入口，但不会自动启动训练。
- 数据读取依赖 `pyarrow`；如果目标环境缺少该包，需要先补齐再正式运行。
- `wandb` 为可选依赖；如果 `WANDB_ENABLED=true`，建议环境中安装 `wandb` 并提前配置好 `WANDB_API_KEY`。
- 当前默认配置更偏向“框架可启动”，后续可以继续细化 batch size、日志频率和模型规模。
